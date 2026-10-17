"""Fault-injection regressions for session, media, and private recording recovery."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

from websockets.sync.client import connect

from tests.browser_smoke import ChromePage, wait_for_devtools_port


def run(url, executable):
    artifacts = Path("test-artifacts/resilience").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="chrome-", dir=artifacts))
    command = [executable, "--headless=new", "--no-first-run", "--no-default-browser-check",
               "--remote-debugging-port=0", "--user-data-dir=" + str(profile),
               "--autoplay-policy=no-user-gesture-required", "about:blank"]
    with (artifacts / "chrome.log").open("w", encoding="utf-8") as log:
        browser = subprocess.Popen(command, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            port = wait_for_devtools_port(profile, browser)
            with urllib.request.urlopen("http://127.0.0.1:" + port + "/json/list") as response:
                target = next(item for item in json.load(response) if item["type"] == "page")
            with connect(target["webSocketDebuggerUrl"]) as socket:
                page = ChromePage(socket)
                page.call("Runtime.enable")
                page.call("Page.enable")
                page.call("Emulation.setDeviceMetricsOverride", {"width":1280,"height":900,"deviceScaleFactor":1,"mobile":False})
                page.call("Page.navigate", {"url":url})
                page.wait_for("document.querySelector('#server-status')?.classList.contains('online')")
                page.click("#sample-button")
                page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                page.evaluate("""window.realFetch = window.fetch.bind(window); window.waitingWaves = []; window.waveAborts = 0;
                    window.fetch = (input, init) => !String(input).startsWith('/media/') ? realFetch(input, init)
                        : new Promise((resolve, reject) => {
                            const pending = {aborted:false, release:() => resolve(realFetch(input, init))};
                            waitingWaves.push(pending);
                            init?.signal?.addEventListener('abort', () => {pending.aborted=true; waveAborts++; reject(new DOMException('Aborted','AbortError'));},{once:true});
                        });""")
                page.click("#process-button")
                page.wait_for("waitingWaves.length === 2 && !busy && document.querySelector('#status').textContent === '處理完成'")
                assert page.evaluate("!document.querySelector('#reset-button').disabled && !document.querySelector('#process-button').disabled")
                page.evaluate("void drawWaveform(originalAudio.getAttribute('src'), 'original-waveform', trackColors.original, 50)")
                page.wait_for("document.querySelector('#original-track').dataset.waveformState === 'error'")
                assert page.evaluate("!document.querySelector('[data-retry=original]').hidden && !document.querySelector('[data-play=original]').disabled && !busy")
                page.evaluate("window.fetch = realFetch; waitingWaves.filter(p => !p.aborted).forEach(p => p.release())")
                page.click("[data-retry=original]")
                page.wait_for("document.querySelector('#original-track').dataset.waveformState === 'ready' && document.querySelector('#denoised-track').dataset.waveformState === 'ready'")
                assert page.evaluate("waveAborts >= 2")
                print("Pending waveform does not lock controls; timeout, cancellation and retry: passed")

                page.evaluate("""window.fetch = (input, init) => input === '/api/denoise' && init?.method === 'POST'
                    ? Promise.resolve(new Response(JSON.stringify({recordingId:'00000000000000000000000000000000',
                        originalUrl:'/media/00000000000000000000000000000001.wav',denoisedUrl:'/media/00000000000000000000000000000002.wav',
                        inputFormat:'wav',sampleRate:16000,channels:1,durationSeconds:1,strength:1,expiresAt:Math.floor(Date.now()/1000)+3600}),
                        {status:200,headers:{'Content-Type':'application/json'}})) : realFetch(input, init);""")
                page.click("#process-button")
                page.wait_for("!busy && playbackStates.original === 'error' && playbackStates.denoised === 'error'")
                assert page.evaluate("document.querySelector('#editor-state-label').textContent === '結果載入失敗' && document.querySelector('#status').classList.contains('error')")
                assert page.evaluate("!document.querySelector('#original-error').hidden && !document.querySelector('[data-retry=original]').hidden")
                for width, height in [(1280,900),(390,844),(320,740)]:
                    page.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
                    page.check_layout()
                    assert page.evaluate("getComputedStyle(document.querySelector('#original-error')).display !== 'none'")
                page.screenshot(artifacts / "mobile-visible-media-error.png")
                page.evaluate("window.fetch = realFetch")
                page.click("#reset-button")
                page.wait_for("!busy && !originalAudio.getAttribute('src')")
                print("404 media is visibly failed on desktop/mobile, never reported complete: passed")

                page.click("#sample-button")
                page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                page.evaluate("""window.posts = 0; window.sessionRefreshes = 0;
                    window.fetch = (input, init) => {
                        if (input === '/api/session') sessionRefreshes++;
                        if (input === '/api/denoise' && init?.method === 'POST' && ++posts === 1)
                            return Promise.resolve(new Response(JSON.stringify({detail:'Session expired'}),{status:401,headers:{'Content-Type':'application/json'}}));
                        return realFetch(input, init);
                    };""")
                page.click("#process-button")
                page.wait_for("!busy && document.querySelector('#status').textContent === '處理完成'")
                assert page.evaluate("posts === 2 && sessionRefreshes === 1")
                page.evaluate("window.fetch = realFetch")
                owned = page.evaluate("({id:currentResult.recordingId,url:currentResult.originalUrl})")
                page.call("Page.navigate", {"url":url})
                page.wait_for("document.querySelector('#server-status')?.classList.contains('online')")
                assert page.evaluate("savedRecordings.size === 0")
                page.click("#recordings-button")
                page.wait_for("!historyBusy && document.querySelectorAll('.recording-row').length === 1")
                assert page.evaluate("document.querySelector('.recording-row').dataset.recordingId === " + json.dumps(owned["id"]))
                for width, height in [(1280,900),(390,844),(320,740)]:
                    page.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
                    assert page.evaluate("document.querySelector('#recordings-dialog').scrollWidth <= document.querySelector('#recordings-dialog').clientWidth + 1")
                    assert page.evaluate("[...document.querySelectorAll('.recording-row button')].every(e => e.querySelector('svg'))")
                page.screenshot(artifacts / "mobile-private-recordings.png")
                page.click('.recording-row button[aria-label="載入這筆音訊"]')
                page.wait_for("!busy && currentResult?.recordingId === " + json.dumps(owned["id"]) + " && playbackStates.original === 'ready'")
                print("401 renews session once and retries; recordings survive reload and restore: passed")

                page.click("#sample-button")
                page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                page.evaluate("""window.realFetch = window.fetch.bind(window);
                    window.fetch = (input, init) => input === '/api/denoise' && init?.method === 'POST'
                        ? Promise.resolve(new Response(JSON.stringify({detail:'Quota full'}),{status:507,headers:{'Content-Type':'application/json'}}))
                        : realFetch(input, init);""")
                page.click("#process-button")
                page.wait_for("document.querySelector('#recordings-dialog').open && !historyBusy")
                assert page.evaluate("document.querySelectorAll('.recording-row').length === 1 && document.querySelector('#status').classList.contains('error')")
                page.evaluate("""window.fetch = realFetch; window.realFetch = window.fetch.bind(window);
                    window.fetch = (input, init) => String(input).startsWith('/api/recordings/') && init?.method === 'DELETE'
                        ? Promise.resolve(new Response('{}',{status:503,headers:{'Content-Type':'application/json'}})) : realFetch(input, init);""")
                page.click('.recording-row button[aria-label="刪除這筆音訊"]')
                page.wait_for("!historyBusy && !busy && document.querySelector('#recordings-status').classList.contains('error')")
                assert page.evaluate("document.querySelectorAll('.recording-row').length === 1")
                page.evaluate("window.fetch = realFetch")
                page.click('.recording-row button[aria-label="刪除這筆音訊"]')
                page.wait_for("!historyBusy && document.querySelectorAll('.recording-row').length === 0")
                assert page.evaluate("(async () => (await fetch(" + json.dumps(owned["url"]) + ")).status)()") == 404
                page.click("#recordings-close")
                print("Quota errors open private management; deletion failure preserves entry; retry reclaims it: passed")

                for unused in range(2):
                    page.click("#process-button")
                    page.wait_for("!busy && document.querySelector('#status').textContent === '處理完成'")
                page.click("#recordings-button")
                page.wait_for("!historyBusy && document.querySelectorAll('.recording-row').length === 2")
                page.click("#recordings-clear")
                assert page.evaluate("!document.querySelector('#recordings-confirmation').hidden")
                page.click("#recordings-cancel")
                assert page.evaluate("document.querySelector('#recordings-confirmation').hidden && document.querySelectorAll('.recording-row').length === 2")
                page.click("#recordings-clear")
                page.click("#recordings-confirm")
                page.wait_for("!historyBusy && !busy && document.querySelectorAll('.recording-row').length === 0")
                page.click("#recordings-close")
                print("Confirmed bulk deletion clears owned results and releases usage: passed")

                page.evaluate("""window.sessionRefreshes = 0; window.posts = 0;
                    window.fetch = (input, init) => {
                        if (input === '/api/session') sessionRefreshes++;
                        if (input === '/api/denoise' && init?.method === 'POST') {
                            posts++; return Promise.resolve(new Response(JSON.stringify({detail:'Session unavailable'}),{status:401,headers:{'Content-Type':'application/json'}}));
                        }
                        return realFetch(input, init);
                    };""")
                page.click("#process-button")
                page.wait_for("posts === 2 && !busy")
                assert page.evaluate("sessionRefreshes === 1 && document.querySelector('#status').classList.contains('error') && !document.querySelector('#process-button').disabled")
                page.evaluate("window.fetch = realFetch")
                assert not page.errors, page.errors
                print("Repeated 401 has a bounded retry and leaves controls usable: passed")
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--chrome", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    options = parser.parse_args()
    run(options.url.rstrip("/"), options.chrome)
