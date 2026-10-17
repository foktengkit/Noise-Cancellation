import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import httpx
from websockets.sync.client import connect

from backend.denoise import write_wav_bytes


def wait_for_devtools_port(profile, browser, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if browser.poll() is not None:
            break
        try:
            port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
            if port.isdigit():
                return port
        except (OSError, IndexError):
            pass
        time.sleep(0.1)
    raise RuntimeError("Chrome did not start; see test-artifacts/chrome.log.")


class ChromePage:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.errors = []

    def call(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv(timeout=15))
            if message.get("method") == "Runtime.exceptionThrown":
                self.errors.append(message["params"]["exceptionDetails"])
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait_for(self, expression, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(0.1)
        raise AssertionError("Browser condition timed out: " + expression + "; status=" +
                             str(self.evaluate("document.querySelector('#status')?.textContent")))

    def click(self, selector):
        self.evaluate("document.querySelector(" + json.dumps(selector) + ").click()")

    def wait_for_app(self, seconds=15):
        self.wait_for("typeof appReady !== 'undefined'", seconds=seconds)
        assert self.evaluate("appReady"), "Application configuration did not load"
        assert self.evaluate("!document.querySelector('#server-status, #reset-button, #recordings-refresh')")
        assert self.evaluate("document.querySelectorAll('.app-header button').length === 1 && !!document.querySelector('#recordings-button svg')")

    def clear_recordings(self):
        self.click("#recordings-button")
        self.wait_for("!historyBusy && !!document.querySelector('.recording-row')")
        self.click("#recordings-clear")
        self.click("#recordings-confirm")
        self.wait_for("!historyBusy && !busy && !document.querySelector('.recording-row') && !document.querySelector('#original-audio').getAttribute('src')")
        self.click("#recordings-close")

    def screenshot(self, path):
        data = self.call("Page.captureScreenshot", {"captureBeyondViewport": True})
        path.write_bytes(base64.b64decode(data["data"]))

    def check_layout(self):
        issues = self.evaluate("""(() => {
            const issues = [];
            if (document.documentElement.scrollWidth > innerWidth) issues.push('page overflow');
            for (const e of document.querySelectorAll('button,h1,h2,h3,.session-name,.track-title,.audio-metrics output,#strength-value,#file-label,#live-latency')) {
                if (e.getClientRects().length && e.scrollWidth > e.clientWidth + 1) issues.push(e.id || e.className || e.tagName);
            }
            const bounds = s => document.querySelector(s).getBoundingClientRect();
            const source = bounds('.input-panel'), editor = bounds('.editor');
            if (source.right > editor.left + 1 && source.bottom > editor.top + 1) issues.push('panels overlap');
            if (bounds('.brand').right > bounds('.header-actions').left - 4) issues.push('header overlap');
            for (const button of document.querySelectorAll('.app-header .icon-button')) {
                if (!button.querySelector('svg')) issues.push('missing header icon: ' + button.id);
            }
            return issues;
        })()""")
        assert not issues, issues


def run(url, executable):
    artifacts = Path("test-artifacts").resolve()
    artifacts.mkdir(exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="chrome-", dir=str(artifacts)))
    command = [
        executable, "--headless=new", "--no-first-run", "--no-default-browser-check",
        "--remote-debugging-port=0", "--user-data-dir=" + str(profile),
        "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        "--autoplay-policy=no-user-gesture-required", "about:blank",
    ]
    chrome_log = (artifacts / "chrome.log").open("w", encoding="utf-8")
    options = {"stdout": chrome_log, "stderr": chrome_log}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    browser = subprocess.Popen(command, **options)
    try:
        port = wait_for_devtools_port(profile, browser)
        with urllib.request.urlopen("http://127.0.0.1:" + port + "/json/list") as response:
            targets = json.load(response)
        target = next(target for target in targets if target["type"] == "page")
        with connect(target["webSocketDebuggerUrl"]) as socket:
            page = ChromePage(socket)
            page.call("Runtime.enable")
            page.call("Page.enable")
            for name, width, height in [("desktop", 1280, 900), ("mobile", 390, 844)]:
                page.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
                page.call("Page.navigate", {"url": url})
                page.wait_for_app()
                assert page.evaluate("document.title === 'Noise Control | 音訊工作台'")
                assert page.evaluate("document.querySelector('.brand h1').textContent === 'Noise Control'")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), name + " overflow"
                page.check_layout()
                page.screenshot(artifacts / (name + "-file.png"))
                page.click("#sample-button")
                page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                page.wait_for("!document.querySelector('[data-play=original]').disabled")
                assert page.evaluate("document.querySelector('#session-name').textContent === '合成人聲範例.wav'")
                assert page.evaluate("document.querySelector('#original-audio').duration >= 7.9")
                page.click('[data-strength="50"]')
                assert page.evaluate("document.querySelector('#strength').value === '50'")
                assert page.evaluate("document.querySelector('[data-strength=\"50\"]').getAttribute('aria-pressed') === 'true'")
                page.click('[data-play="original"]')
                page.wait_for("!document.querySelector('#original-audio').paused")
                page.click('[data-mute="original"]')
                assert page.evaluate("document.querySelector('#original-audio').muted")
                page.evaluate("(() => {const s = document.querySelector('[data-speed=original]'); s.value = '1.5'; s.dispatchEvent(new Event('change'));})()")
                assert page.evaluate("document.querySelector('#original-audio').playbackRate === 1.5")
                page.evaluate("(() => {const s = document.querySelector('[data-seek=original]'); s.value = 500; s.dispatchEvent(new Event('input'));})()")
                page.wait_for("document.querySelector('#original-audio').currentTime >= 3.9")
                page.click("#process-button")
                page.wait_for("document.querySelector('#status').textContent === '處理完成'")
                page.wait_for("!document.querySelector('[data-play=denoised]').disabled")
                page.click('[data-play="original"]')
                page.wait_for("!document.querySelector('#original-audio').paused")
                page.click('[data-play="denoised"]')
                page.wait_for("!document.querySelector('#denoised-audio').paused && document.querySelector('#original-audio').paused")
                page.click('[data-play="denoised"]')
                page.wait_for("document.querySelector('#denoised-audio').paused")
                assert page.evaluate("document.querySelector('#metric-rate').textContent.includes('16000')")
                overflow = page.evaluate("[...document.querySelectorAll('button,h1,h2,h3,.session-name,.track-title')].filter(e => e.getClientRects().length && e.scrollWidth > e.clientWidth + 1).map(e => e.id || e.className)")
                assert not overflow, overflow
                page.check_layout()
                page.screenshot(artifacts / (name + "-studio.png"))
                if name == "mobile":
                    page.call("Emulation.setDeviceMetricsOverride", {"width": 320, "height": 740, "deviceScaleFactor": 1, "mobile": False})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "narrow mobile overflow"
                    page.screenshot(artifacts / "mobile-narrow.png")
                    page.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
                previous_link = page.evaluate("document.querySelector('#original-download').getAttribute('href')")
                page.clear_recordings()
                assert page.evaluate("(async () => (await fetch(" + json.dumps(previous_link) + ")).status)()") == 404
                assert page.evaluate("document.querySelector('#file-input').files.length === 1")
                assert page.evaluate("!document.querySelector('#original-audio').getAttribute('src')")
                assert page.evaluate("document.querySelector('#strength').value === '50'")
                assert page.evaluate("document.querySelector('#hum-filter').value === 'off'")
                page.evaluate("document.querySelector('#hum-filter').value = '60'")
                assert page.evaluate("document.querySelector('[data-play=original]').disabled")
                page.click("#mic-tab")
                page.click("#record-button")
                page.wait_for("document.querySelector('#record-label').textContent === '停止收音'")
                assert page.evaluate("document.querySelector('#hum-filter').disabled")
                page.wait_for("document.querySelector('#recording-time').textContent !== '00:00'")
                page.wait_for("document.querySelector('#live-latency').textContent.includes('ms')")
                page.click("#monitor-toggle")
                page.click('input[name="monitor-mode"][value="original"]')
                page.click('input[name="monitor-mode"][value="denoised"]')
                page.evaluate("const slider = document.querySelector('#strength'); for(let i=0;i<80;i++){slider.value=String(i%101);slider.dispatchEvent(new Event('input'));} slider.value=60;slider.dispatchEvent(new Event('input'));")
                pixel_expression = "(() => { const c = document.querySelector('#denoised-waveform'); const p = c.getContext('2d').getImageData(0,0,c.width,c.height).data; let n = 0; for(let i=0;i<p.length;i+=4) if(p[i+3] && p[i+1] > p[i]*2) n++; return n; })()"
                page.wait_for(pixel_expression + " > 100")
                pixels = page.evaluate(pixel_expression)
                page.screenshot(artifacts / (name + "-live.png"))
                page.check_layout()
                page.click("#record-button")
                page.wait_for("document.querySelector('#status').textContent === '錄音已保存'")
                assert page.evaluate("document.querySelector('#denoised-meta').textContent.includes('60 Hz')")
                for kind in ["original", "denoised"]:
                    link = page.evaluate("document.querySelector('#" + kind + "-download').getAttribute('href')")
                    downloaded = page.evaluate("(async () => { const r = await fetch(" + json.dumps(link) + "); const b = new Uint8Array(await r.arrayBuffer()); return {status:r.status, magic:String.fromCharCode(...b.slice(0,4))}; })()")
                    assert downloaded == {"status": 200, "magic": "RIFF"}, downloaded
                page.screenshot(artifacts / (name + "-result.png"))
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                # A second session catches leaked microphone graphs and stale socket callbacks.
                page.click("#record-button")
                page.wait_for("document.querySelector('#record-label').textContent === '停止收音'")
                page.wait_for("document.querySelector('#recording-time').textContent !== '00:00'")
                page.click("#record-button")
                page.wait_for("document.querySelector('#status').textContent === '錄音已保存'")
                page.click("#file-tab")
                noise = np.random.default_rng(5).normal(0, 0.04, (16000, 1)).astype(np.float32)
                fixture = artifacts / "test.wav"
                fixture.write_bytes(write_wav_bytes(noise, 16000))
                root = page.call("DOM.getDocument")["root"]["nodeId"]
                node = page.call("DOM.querySelector", {"nodeId": root, "selector": "#file-input"})["nodeId"]
                page.call("DOM.setFileInputFiles", {"nodeId": node, "files": [str(fixture)]})
                page.click("#process-button")
                page.wait_for("document.querySelector('#status').textContent === '處理完成'")
                assert not page.errors, page.errors
                print(json.dumps({"viewport": name, "waveformPixels": pixels, "microphone": "passed", "upload": "passed", "sample": "passed", "transport": "passed", "presets": "passed", "privateDeletion": "passed", "simplifiedHeader": "passed"}))
            page.clear_recordings()
            page.click("#sample-button")
            page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
            page.evaluate("const longName = 'voice-recording-'.repeat(15) + '.wav'; document.querySelector('#file-label').textContent = longName; document.querySelector('#session-name').textContent = longName;")
            for width, height in [(1920,1080),(1440,900),(1024,768),(768,1024),(721,900),(720,900),(390,844),(320,740)]:
                page.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
                page.check_layout()
                if width in (1440, 768, 320):
                    page.screenshot(artifacts / ("layout-long-name-" + str(width) + ".png"))
            page.evaluate("document.querySelector('#mic-tab').dispatchEvent(new KeyboardEvent('keydown', {key:'End',bubbles:true}))")
            assert page.evaluate("document.querySelector('#mic-tab').getAttribute('aria-selected') === 'true' && !document.querySelector('#mic-panel').hidden")
            page.evaluate("document.querySelector('#mic-tab').dispatchEvent(new KeyboardEvent('keydown', {key:'Home',bubbles:true}))")
            assert page.evaluate("document.activeElement.id === 'file-tab' && !document.querySelector('#file-panel').hidden")
            page.evaluate("document.querySelector('#strength').value = '0'; document.querySelector('#strength').dispatchEvent(new Event('input', {bubbles:true}))")
            assert page.evaluate("document.querySelector('#strength').style.getPropertyValue('--range-fill') === '0%'")
            assert not page.errors, page.errors
            print("Responsive layouts 320-1920px, long filenames, keyboard tabs and slider theme: passed")
            if page.evaluate("settings.separationEnabled"):
                page.evaluate("document.querySelector('#file-engine').value='separate';document.querySelector('#file-engine').dispatchEvent(new Event('change'))")
                assert page.evaluate("document.querySelector('#strength').disabled && document.querySelector('#hum-filter').disabled")
                assert page.evaluate("document.querySelector('#file-limit').textContent.includes('15 秒')")
                page.click("#sample-button")
                page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                assert page.evaluate("document.querySelector('#file-label').textContent === '合成雙人聲範例.wav'")
                page.click("#process-button")
                page.wait_for("!document.body.classList.contains('is-busy')", seconds=160)
                page.wait_for("document.querySelector('#status').textContent === '處理完成' || document.querySelector('#status').classList.contains('error')")
                assert page.evaluate("!document.querySelector('#status').classList.contains('error')"), page.evaluate("document.querySelector('#status').textContent")
                assert page.evaluate("separatedResults.length === 2 && !document.querySelector('#source-option').hidden")
                first_url = page.evaluate("document.querySelector('#denoised-download').getAttribute('href')")
                page.evaluate("document.querySelector('#source-select').value='1';document.querySelector('#source-select').dispatchEvent(new Event('change'))")
                page.wait_for("document.querySelector('#denoised-meta').textContent.includes('人聲 2')")
                second_url = page.evaluate("document.querySelector('#denoised-download').getAttribute('href')")
                assert first_url != second_url
                source_ids = page.evaluate("separatedResults.map(result => result.recordingId)")
                with httpx.Client(base_url=url, headers={"origin": url}, timeout=15) as other:
                    other.get("/api/session").raise_for_status()
                    assert other.get("/api/recordings").json()["recordings"] == []
                    for link, recording_id in zip([first_url, second_url], source_ids):
                        assert other.get(link).status_code == 404
                        assert other.head(link).status_code == 404
                        assert other.get(link, headers={"range":"bytes=0-15"}).status_code == 404
                        assert other.delete("/api/recordings/" + recording_id).status_code == 404
                page.wait_for("!document.querySelector('[data-play=denoised]').disabled")
                page.click('[data-play="denoised"]')
                page.wait_for("!document.querySelector('#denoised-audio').paused")
                for width, height in [(1280,900),(390,844),(320,740)]:
                    page.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
                    page.check_layout()
                    page.screenshot(artifacts / ("separation-" + str(width) + ".png"))
                page.clear_recordings()
                page.wait_for("!document.body.classList.contains('is-busy') && document.querySelector('#source-option').hidden")
                for link in [first_url, second_url]:
                    assert page.evaluate("(async()=> (await fetch(" + json.dumps(link) + ")).status)()") == 404
                print("Private two-speaker separation, source selection, playback, 320-1280px layouts and cleanup: passed")
            # Exercise the worklet's limit and partial final chunk without waiting two minutes.
            # Only this synthetic VM uses eval; all real UI flows above run under production CSP.
            page.call("Page.setBypassCSP", {"enabled": True})
            page.call("Page.navigate", {"url": url})
            page.wait_for_app()
            result = page.evaluate("""(async () => {
                const code = await (await fetch('/capture-worklet.js')).text();
                let Capture;
                const messages = [];
                class Processor { constructor() { this.port = {postMessage: m => messages.push(m)}; } }
                new Function('AudioWorkletProcessor', 'sampleRate', 'registerProcessor', code)(Processor, 8000, (name, cls) => Capture = cls);
                const p = new Capture({processorOptions: {chunkSeconds: 0.25, maxDurationSeconds: 0.2}});
                for (let i=0;i<20;i++) p.process([[new Float32Array(128).fill(0.1)]]);
                p.stop();
                const chunks = messages.filter(m => m.type === 'chunk');
                return {samples: chunks.reduce((n,m) => n + m.buffer.byteLength/4, 0),
                        stops: messages.filter(m => m.type === 'stopped').length};
            })()""")
            assert result == {"samples": 1600, "stops": 1}, result
            print("Worklet duration limit and final partial chunk: passed")
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()
            browser.wait()
        chrome_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--chrome", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    arguments = parser.parse_args()
    run(arguments.url.rstrip("/"), arguments.chrome)
