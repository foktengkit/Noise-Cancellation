"""Verify every public demo through the real selection and processing UI."""

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
    artifacts = Path("test-artifacts/samples").resolve()
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
                page.call("Page.navigate", {"url": url})
                page.wait_for_app(seconds=90)
                choices = page.evaluate("[...sampleSelect.options].filter(o => !o.disabled).map(o => ({id:o.value,name:o.dataset.filename}))")
                assert len(choices) == 6, choices
                try:
                    for index, choice in enumerate(choices):
                        width, height = [(1280, 900), (390, 844), (320, 740)][index % 3]
                        page.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height,
                                  "deviceScaleFactor": 1, "mobile": False})
                        page.evaluate("sampleSelect.value = " + json.dumps(choice["id"]))
                        page.click("#sample-button")
                        page.wait_for("document.querySelector('#status').textContent === '範例已載入'", seconds=30)
                        page.wait_for("!document.querySelector('[data-play=original]').disabled", seconds=30)
                        assert page.evaluate("fileInput.files[0].name") == choice["name"]
                        assert page.evaluate("originalAudio.duration > 10 && originalAudio.duration < 20")
                        page.click('[data-play="original"]')
                        page.wait_for("!originalAudio.paused")
                        page.click('[data-play="original"]')
                        page.click("#process-button")
                        page.wait_for("document.querySelector('#status').textContent === '處理完成'", seconds=90)
                        page.click('[data-play="denoised"]')
                        page.wait_for("!denoisedAudio.paused")
                        page.click('[data-play="denoised"]')
                        assert page.evaluate("sampleSelect.value") == choice["id"]
                        page.check_layout()
                        page.screenshot(artifacts / (choice["id"] + "-" + str(width) + ".png"))
                        assert not page.errors, page.errors
                        page.clear_recordings()
                        print(json.dumps({"sample": choice["id"], "viewport": width,
                                          "originalPlayback": True, "denoisedPlayback": True,
                                          "privateDeletion": True}), flush=True)
                    page.evaluate("""window.realFetch = window.fetch.bind(window);
                        window.fetch = (input, init) => String(input).startsWith('/assets/sample-')
                            ? Promise.resolve(new Response('', {status:503})) : realFetch(input, init);""")
                    previous = page.evaluate("fileInput.files[0].name")
                    page.click("#sample-button")
                    page.wait_for("!busy && document.querySelector('#status').classList.contains('error')")
                    assert page.evaluate("!sampleSelect.disabled && !document.querySelector('#sample-button').disabled")
                    assert page.evaluate("fileInput.files[0].name") == previous
                    page.evaluate("window.fetch = realFetch")
                    page.click("#sample-button")
                    page.wait_for("document.querySelector('#status').textContent === '範例已載入'")
                    if page.evaluate("settings.separationEnabled"):
                        page.evaluate("fileEngine.value='separate'; fileEngine.dispatchEvent(new Event('change'))")
                        assert page.evaluate("sampleSelect.value === 'speakers' && [...sampleSelect.options].filter(o => !o.disabled).length === 1")
                        page.evaluate("fileEngine.value='rnnoise'; fileEngine.dispatchEvent(new Event('change'))")
                        assert page.evaluate("sampleSelect.value === 'fan' && [...sampleSelect.options].filter(o => !o.disabled).length === 6")
                    assert not page.errors, page.errors
                    print("All six demos, responsive layout, load failure recovery and engine filtering: passed", flush=True)
                finally:
                    status = page.evaluate("(async () => (await fetch('/api/recordings', {method:'DELETE'})).status)()")
                    assert status == 204, status
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--chrome", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    options = parser.parse_args()
    run(options.url.rstrip("/"), options.chrome)
