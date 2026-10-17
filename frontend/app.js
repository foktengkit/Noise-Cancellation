const $ = (selector) => document.querySelector(selector);
const form = $("#upload-form");
const fileInput = $("#file-input");
const strength = $("#strength");
const humFilter = $("#hum-filter");
const fileEngine = $("#file-engine");
const sampleSelect = $("#sample-select");
let separatedResults = [];
const statusText = $("#status");
const recordButton = $("#record-button");
const monitorToggle = $("#monitor-toggle");
const originalAudio = $("#original-audio");
const denoisedAudio = $("#denoised-audio");

let audioContext;
let workletLoaded = false;
let mode = "file";
let busy = false;
let micSession = null;
let settings = { maxUploadBytes: 25 * 1024 * 1024, maxDurationSeconds: 120, maxSeparationDurationSeconds: 15, chunkSeconds: 0.25 };
let previewUrl = null;
let selectionVersion = 0;
let sessionPromise = null;
let sessionPending = false;
let currentResult = null;
let historyBusy = false;
const waveformVersions = {};
const waveformControllers = {};
const mediaTimers = {};
const playbackStates = { original: "empty", denoised: "empty" };
const trackKinds = ["original", "denoised"];
const trackColors = { original: "#758bb3", denoised: "#168477" };

function formatTime(seconds) {
  const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return String(Math.floor(value / 60)).padStart(2, "0") + ":" + String(value % 60).padStart(2, "0");
}

function updateRuler(kind, duration) {
  $("#" + kind + "-ruler").querySelectorAll("span").forEach((tick, index) => {
    tick.textContent = duration > 0
      ? duration < 4 ? (duration * index / 4).toFixed(1) + "s" : formatTime(duration * index / 4)
      : index === 0 ? "00:00" : "--:--";
  });
}

function updateFileLimit() {
  const limit = fileEngine.value === "separate" ? settings.maxSeparationDurationSeconds : settings.maxDurationSeconds;
  $("#file-limit").textContent = Math.round(settings.maxUploadBytes / 1024 / 1024) + " MB · 最長 "
    + (limit < 60 ? limit + " 秒" : limit / 60 + " 分鐘");
}

function updateEditorState(isError = false) {
  const failed = trackKinds.some((kind) => playbackStates[kind] === "error");
  isError = isError || (!!currentResult && failed);
  const hasResult = !!currentResult && trackKinds.every((kind) => playbackStates[kind] === "ready");
  const label = isError ? currentResult ? "結果載入失敗" : "處理未完成" : busy ? "處理中" : micSession
    ? micSession.stopping ? "正在保存" : micSession.ready ? "即時收音" : "連線中"
    : hasResult ? "已完成" : currentResult ? "結果載入中" : originalAudio.getAttribute("src") ? "原音已就緒" : "等待音訊";
  if ($("#editor-state-label").textContent !== label) $("#editor-state-label").textContent = label;
  $("#editor-state").classList.toggle("ready", !isError && (hasResult || !!micSession?.ready));
  $("#editor-state").classList.toggle("error", isError);
}

function syncPlayer(kind) {
  const player = $("#" + kind + "-audio");
  const hasDuration = player.readyState >= 1 && Number.isFinite(player.duration) && player.duration > 0 && !!player.getAttribute("src");
  const ready = hasDuration && !player.error && playbackStates[kind] !== "error" && !busy && !micSession;
  const playing = hasDuration && !player.paused && !player.ended;
  const track = $("#" + kind + "-track");
  const play = $('[data-play="' + kind + '"]');
  const seek = $('[data-seek="' + kind + '"]');
  const mute = $('[data-mute="' + kind + '"]');
  const speed = $('[data-speed="' + kind + '"]');
  play.disabled = seek.disabled = mute.disabled = speed.disabled = !ready;
  play.classList.toggle("is-playing", playing);
  track.classList.toggle("is-playing", playing);
  play.setAttribute("aria-label", (playing ? "暫停" : "播放") + (kind === "original" ? "原始音訊" : "降噪音訊"));
  play.title = play.getAttribute("aria-label");
  mute.classList.toggle("is-muted", player.muted);
  mute.setAttribute("aria-pressed", String(player.muted));
  mute.setAttribute("aria-label", (player.muted ? "取消靜音" : "靜音") + (kind === "original" ? "原始音訊" : "降噪音訊"));
  mute.title = mute.getAttribute("aria-label");
  speed.value = String(player.playbackRate);
  const progress = hasDuration ? Math.min(1, Math.max(0, player.currentTime / player.duration)) : 0;
  seek.value = String(Math.round(progress * 1000));
  seek.style.setProperty("--range-fill", progress * 100 + "%");
  $("#" + kind + "-current").textContent = formatTime(player.currentTime);
  $("#" + kind + "-duration").textContent = formatTime(hasDuration ? player.duration : 0);
  const playhead = $("#" + kind + "-playhead");
  playhead.hidden = !hasDuration;
  playhead.style.left = progress * 100 + "%";
  $("#" + kind + "-state").textContent = playbackStates[kind] === "error" ? "載入失敗"
    : playbackStates[kind] === "loading" ? "載入中" : playing ? "播放中" : hasDuration
    ? player.ended ? "已播放" : player.currentTime > 0 ? "已暫停" : "待播放"
    : micSession?.ready ? "串流中" : kind === "original" ? "ORIGINAL" : "DENOISED";
  if (hasDuration) updateRuler(kind, player.duration);
}

function updateStrength() {
  $("#strength-value").replaceChildren(document.createTextNode(strength.value), Object.assign(document.createElement("span"), { textContent: "%" }));
  let selected = "自訂";
  document.querySelectorAll("[data-strength]").forEach((button) => {
    const active = button.dataset.strength === strength.value;
    button.setAttribute("aria-pressed", String(active));
    if (active) selected = button.textContent;
  });
  $("#strength-mode").textContent = selected;
  strength.style.setProperty("--range-fill", strength.value + "%");
}

function setStatus(message, isError = false) {
  if (statusText.textContent !== message) statusText.textContent = message;
  statusText.classList.toggle("error", isError);
  updateEditorState(isError);
}

function updateControls() {
  const locked = busy || !!micSession;
  document.body.classList.toggle("is-busy", busy);
  $("#sample-button").disabled = locked;
  sampleSelect.disabled = locked;
  for (const option of sampleSelect.options) {
    option.hidden = option.disabled = option.dataset.engine !== fileEngine.value;
  }
  if (!sampleSelect.selectedOptions.length || sampleSelect.selectedOptions[0].disabled) {
    sampleSelect.value = [...sampleSelect.options].find((option) => !option.disabled).value;
  }
  $("#recordings-button").disabled = locked;
  document.querySelectorAll("[data-retry]").forEach((button) => { button.disabled = locked; });
  const separating = mode === "file" && fileEngine.value === "separate";
  $("#process-label").textContent = busy ? "處理中" : separating ? "分離人聲" : "開始降噪";
  document.querySelectorAll("[data-strength]").forEach((button) => { button.disabled = busy || separating || (!!micSession && !micSession.ready); });
  $("#file-tab").disabled = locked;
  $("#mic-tab").disabled = locked;
  fileInput.disabled = locked;
  $("#process-button").disabled = locked;
  strength.disabled = busy || separating || (!!micSession && !micSession.ready);
  humFilter.disabled = locked || separating;
  fileEngine.disabled = locked;
  $("#source-select").disabled = locked;
  recordButton.disabled = busy || !!(micSession && micSession.stopping);
  $("#record-label").textContent = micSession
    ? micSession.stopping ? "正在保存" : micSession.ready ? "停止收音" : "取消連線"
    : "開始收音";
  recordButton.classList.toggle("recording", !!micSession);
  $("#recording-indicator").classList.toggle("active", !!(micSession && micSession.ready && !micSession.stopping));
  $("#monitor-modes").disabled = !monitorToggle.checked;
  trackKinds.forEach(syncPlayer);
  updateEditorState(statusText.classList.contains("error"));
  refreshResultStatus();
}

function refreshResultStatus() {
  if (!currentResult || busy || micSession) return;
  if (trackKinds.some((kind) => playbackStates[kind] === "error")) {
    setStatus("音訊載入失敗或已過期。", true);
  } else if (trackKinds.every((kind) => playbackStates[kind] === "ready")) {
    setStatus(currentResult.inputFormat === "microphone" ? "錄音已保存" : "處理完成");
  } else {
    setStatus("正在載入音訊…");
  }
}

function updateTrackFeedback(kind) {
  const track = $("#" + kind + "-track");
  track.dataset.mediaState = playbackStates[kind];
  const failed = playbackStates[kind] === "error";
  const waveFailed = track.dataset.waveformState === "error";
  $("#" + kind + "-error").hidden = !failed;
  $('[data-retry="' + kind + '"]').hidden = !failed && !waveFailed;
  $("#" + kind + "-wave-message").textContent = failed ? "音訊無法載入"
    : waveFailed ? "波形載入失敗" : track.dataset.waveformState === "loading" ? "波形載入中"
    : kind === "original" ? "等待音訊" : "等待降噪結果";
  syncPlayer(kind);
  refreshResultStatus();
}

function loadTrack(kind, url, timeoutMs = 15000) {
  clearTimeout(mediaTimers[kind]);
  playbackStates[kind] = "loading";
  const player = $("#" + kind + "-audio");
  player.src = url;
  player.load();
  mediaTimers[kind] = setTimeout(() => {
    if (playbackStates[kind] === "loading") {
      playbackStates[kind] = "error";
      updateTrackFeedback(kind);
    }
  }, timeoutMs);
  updateTrackFeedback(kind);
}

function retryTrack(kind) {
  if (busy || micSession) return;
  const url = $("#" + kind + "-audio").getAttribute("src");
  if (!url) return;
  if (playbackStates[kind] === "error") loadTrack(kind, url);
  void drawWaveform(url, kind + "-waveform", trackColors[kind]);
}

function setMode(next) {
  if (busy || micSession) return;
  mode = next;
  for (const item of ["file", "mic"]) {
    const selected = item === mode;
    $("#" + item + "-tab").setAttribute("aria-selected", String(selected));
    $("#" + item + "-tab").tabIndex = selected ? 0 : -1;
    $("#" + item + "-panel").hidden = !selected;
  }
  $("#process-button").hidden = mode === "mic";
  updateFileLimit();
  if (!originalAudio.getAttribute("src")) $("#session-name").textContent = mode === "mic" ? "麥克風錄音" : fileInput.files[0]?.name || "未命名音訊";
  setStatus(mode === "file" ? "等待音訊" : "等待麥克風");
  updateControls();
}

function getAudioContext() {
  if (!audioContext) {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error("瀏覽器不支援音訊處理。");
    try {
      audioContext = new Context({ sampleRate: 16000, latencyHint: "interactive" });
    } catch {
      audioContext = new Context();
    }
  }
  return audioContext;
}

function drawSamples(samples, canvasId, color) {
  const canvas = $("#" + canvasId);
  const ctx = canvas.getContext("2d");
  const mid = canvas.height / 2;
  canvas.closest(".audio-track").classList.toggle("has-waveform", samples.length > 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e2e9ed";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(canvas.width, mid);
  ctx.stroke();
  if (!samples.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let x = 0; x < canvas.width; x += 1) {
    const start = Math.floor(x * samples.length / canvas.width);
    const end = Math.min(samples.length, Math.max(start + 1, Math.floor((x + 1) * samples.length / canvas.width)));
    let min = 1;
    let max = -1;
    for (let index = start; index < end; index += 1) {
      min = Math.min(min, samples[index]);
      max = Math.max(max, samples[index]);
    }
    ctx.moveTo(x, mid - max * mid * 0.9);
    ctx.lineTo(x, mid - min * mid * 0.9);
  }
  ctx.stroke();
}

async function drawWaveform(url, canvasId, color, timeoutMs = 15000) {
  waveformControllers[canvasId]?.abort();
  const controller = new AbortController();
  waveformControllers[canvasId] = controller;
  const version = (waveformVersions[canvasId] || 0) + 1;
  waveformVersions[canvasId] = version;
  const kind = canvasId.replace("-waveform", "");
  const track = $("#" + kind + "-track");
  track.dataset.waveformState = "loading";
  updateTrackFeedback(kind);
  let timeout;
  try {
    const loading = (async () => {
      const response = await fetch(url, { signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error("Audio unavailable");
      return getAudioContext().decodeAudioData(await response.arrayBuffer());
    })();
    const buffer = await Promise.race([loading, new Promise((resolve, reject) => {
      timeout = setTimeout(() => { controller.abort(); reject(new Error("Waveform timeout")); }, timeoutMs);
    })]);
    if (version !== waveformVersions[canvasId]) return;
    drawSamples(buffer.getChannelData(0), canvasId, color);
    track.dataset.waveformState = "ready";
    updateRuler(canvasId.replace("-waveform", ""), buffer.duration);
  } catch {
    if (version !== waveformVersions[canvasId]) return;
    drawSamples(new Float32Array(0), canvasId, color);
    track.dataset.waveformState = "error";
  } finally {
    clearTimeout(timeout);
    if (version === waveformVersions[canvasId]) {
      delete waveformControllers[canvasId];
      updateTrackFeedback(kind);
    }
  }
}

function clearResult() {
  currentResult = null;
  separatedResults = [];
  $("#source-option").hidden = true;
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  for (const kind of ["original", "denoised"]) {
    waveformVersions[kind + "-waveform"] = (waveformVersions[kind + "-waveform"] || 0) + 1;
    waveformControllers[kind + "-waveform"]?.abort();
    delete waveformControllers[kind + "-waveform"];
    clearTimeout(mediaTimers[kind]);
    playbackStates[kind] = "empty";
    $("#" + kind + "-track").dataset.waveformState = "empty";
    const player = $("#" + kind + "-audio");
    player.pause();
    player.removeAttribute("src");
    player.load();
    $("#" + kind + "-download").hidden = true;
    $("#" + kind + "-download").removeAttribute("href");
    drawSamples(new Float32Array(0), kind + "-waveform", "#65717a");
    updateRuler(kind, 0);
    updateTrackFeedback(kind);
  }
  $("#original-meta").textContent = "尚未載入";
  $("#denoised-meta").textContent = "尚未處理";
  $("#metric-duration").textContent = "--:--";
  $("#metric-rate").textContent = "-- Hz";
  $("#result-expiry").hidden = true;
}

async function ensureSession(force = false) {
  if (force && !sessionPending) sessionPromise = null;
  if (!sessionPromise) {
    sessionPending = true;
    sessionPromise = fetch("/api/session", { cache: "no-store", signal: AbortSignal.timeout(90000) })
      .then(async (response) => {
        if (!response.ok) throw new Error("無法建立私人工作階段，請重新載入頁面。");
        return response.json();
      }).catch((error) => { sessionPromise = null; throw error; })
      .finally(() => { sessionPending = false; });
  }
  return sessionPromise;
}

async function apiFetch(url, options = {}) {
  await ensureSession();
  const request = { ...options, signal: options.signal || AbortSignal.timeout(20000) };
  let response = await fetch(url, request);
  if (response.status === 401) {
    await ensureSession(true);
    response = await fetch(url, request);
  }
  return response;
}

function setHistoryBusy(value) {
  historyBusy = value;
  const dialog = $("#recordings-dialog");
  dialog.setAttribute("aria-busy", String(value));
  dialog.querySelectorAll("button:not(#recordings-close)").forEach((button) => { button.disabled = value; });
  if (!value) $("#recordings-clear").disabled = !dialog.querySelector(".recording-row");
}

function historyAction(icon, label, tag = "button") {
  const action = document.createElement(tag);
  if (tag === "button") action.type = "button";
  action.className = "icon-button";
  action.title = label;
  action.setAttribute("aria-label", label);
  const symbol = document.createElement("i");
  symbol.setAttribute("data-lucide", icon);
  action.append(symbol);
  return action;
}

async function loadRecordings() {
  if (historyBusy) return;
  setHistoryBusy(true);
  $("#recordings-status").textContent = "載入中";
  $("#recordings-status").classList.remove("error");
  $("#recordings-confirmation").hidden = true;
  try {
    const response = await apiFetch("/api/recordings");
    if (!response.ok) throw new Error("音訊紀錄載入失敗，請稍後重試。");
    const data = await response.json();
    const list = $("#recordings-list");
    list.replaceChildren();
    $("#recordings-usage").textContent = (data.usedBytes / 1024 / 1024).toFixed(1) + " / " + (data.maxBytes / 1024 / 1024).toFixed(1) + " MiB";
    $("#recordings-status").textContent = data.recordings.length + " 筆音訊";
    for (const entry of data.recordings) {
      const row = document.createElement("li");
      row.className = "recording-row";
      row.dataset.recordingId = entry.recordingId;
      const info = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = (entry.engine === "sepformer" ? "人聲 " + entry.sourceIndex : entry.inputFormat === "microphone" ? "麥克風錄音" : "音訊檔")
        + (Number.isFinite(entry.durationSeconds) ? " · " + formatTime(entry.durationSeconds) : "");
      const detail = document.createElement("p");
      detail.textContent = (entry.sizeBytes / 1024 / 1024).toFixed(1) + " MiB · 保留至 "
        + new Date(entry.expiresAt * 1000).toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" });
      info.append(title, detail);
      const actions = document.createElement("div");
      actions.className = "recording-actions";
      if (entry.canRestore) {
        const restore = historyAction("folder-open", "載入這筆音訊");
        restore.addEventListener("click", async () => {
          if (busy || micSession || historyBusy) return;
          $("#recordings-dialog").close();
          setMode("file");
          busy = true;
          updateControls();
          clearResult();
          $("#session-name").textContent = entry.inputFormat === "microphone" ? "麥克風錄音" : "暫存音訊";
          strength.value = String(Math.round(entry.strength * 100));
          humFilter.value = ["50", "60"].includes(entry.humFilter) ? entry.humFilter : "off";
          updateStrength();
          try { await displayResult(entry); }
          finally { busy = false; updateControls(); }
        });
        actions.append(restore);
      }
      for (const kind of trackKinds) {
        const download = historyAction("download", kind === "original" ? "下載原始 WAV" : "下載降噪 WAV", "a");
        download.href = entry[kind + "Url"];
        download.download = kind + ".wav";
        actions.append(download);
      }
      const remove = historyAction("trash-2", "刪除這筆音訊");
      remove.addEventListener("click", () => { void deleteRecording(entry.recordingId); });
      actions.append(remove);
      row.append(info, actions);
      list.append(row);
    }
    if (!data.recordings.length) {
      const empty = document.createElement("li");
      empty.className = "recordings-empty";
      empty.textContent = "尚無暫存音訊";
      list.append(empty);
    }
    if (window.lucide) window.lucide.createIcons();
  } catch (error) {
    $("#recordings-status").textContent = error.name === "TimeoutError" ? "紀錄載入逾時，請重試。" : error.message;
    $("#recordings-status").classList.add("error");
  } finally {
    setHistoryBusy(false);
  }
}

async function openRecordings() {
  if (busy || micSession) return;
  if (!$("#recordings-dialog").open) $("#recordings-dialog").showModal();
  await loadRecordings();
}

async function deleteRecording(id = null) {
  if (historyBusy || busy || micSession) return;
  busy = true;
  setHistoryBusy(true);
  updateControls();
  let deleted = false;
  try {
    const response = await apiFetch("/api/recordings" + (id ? "/" + id : ""), { method: "DELETE" });
    if (!response.ok && response.status !== 404) throw new Error("刪除失敗，請稍後重試。");
    if (separatedResults.some((entry) => entry.recordingId === id)) {
      separatedResults = [];
      $("#source-option").hidden = true;
    }
    if (!id || currentResult?.recordingId === id) clearResult();
    setStatus("音訊已刪除");
    deleted = true;
  } catch (error) {
    $("#recordings-status").textContent = error.name === "TimeoutError" ? "刪除逾時，請重試。" : error.message;
    $("#recordings-status").classList.add("error");
  } finally {
    busy = false;
    setHistoryBusy(false);
    updateControls();
  }
  if (deleted) await loadRecordings();
}

async function loadSample() {
  if (busy || micSession) return;
  busy = true;
  updateControls();
  setStatus("正在載入範例…");
  try {
    const sample = sampleSelect.selectedOptions[0];
    if (!sample || sample.disabled) throw new Error("請選擇可用的範例。");
    const response = await fetch(sample.dataset.url, { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error("範例音訊無法載入，請稍後重試。");
    const transfer = new DataTransfer();
    transfer.items.add(new File([await response.blob()], sample.dataset.filename, { type: "audio/wav" }));
    fileInput.files = transfer.files;
    await showFileName();
    setStatus("範例已載入");
  } catch (error) {
    setStatus(error.name === "TimeoutError" ? "載入範例逾時，請稍後重試。" : error.message, true);
  } finally {
    busy = false;
    updateControls();
  }
}

async function displayResult(result) {
  currentResult = result;
  loadTrack("original", result.originalUrl);
  loadTrack("denoised", result.denoisedUrl);
  $("#metric-duration").textContent = formatTime(result.durationSeconds);
  $("#metric-rate").replaceChildren(document.createTextNode(String(result.sampleRate) + " "), Object.assign(document.createElement("small"), { textContent: "Hz" }));
  const format = result.inputFormat === "microphone" ? "麥克風" : result.inputFormat.toUpperCase();
  $("#original-meta").textContent = format + " · " + result.durationSeconds + " 秒 · " + (result.originalSampleRate || result.sampleRate) + " Hz · " + (result.originalChannels || result.channels) + " 聲道";
  $("#denoised-meta").textContent = (result.inputFormat === "microphone" ? "AI 人聲 · 最後強度 " : "AI 人聲 · 強度 ") + Math.round(result.strength * 100) + "%";
  if (result.strength > 0 && ["50", "60"].includes(result.humFilter)) $("#denoised-meta").textContent += " · " + result.humFilter + " Hz 抑制";
  if (result.engine === "sepformer") $("#denoised-meta").textContent = "人聲 " + result.sourceIndex + " · " + result.sampleRate + " Hz · 單聲道";
  for (const kind of ["original", "denoised"]) {
    $("#" + kind + "-download").href = result[kind + "Url"];
    $("#" + kind + "-download").hidden = false;
  }
  $("#result-expiry").textContent = "音訊暫存至 " + new Date(result.expiresAt * 1000).toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" });
  $("#result-expiry").hidden = false;
  void drawWaveform(result.originalUrl, "original-waveform", trackColors.original);
  void drawWaveform(result.denoisedUrl, "denoised-waveform", trackColors.denoised);
}

async function submitAudio(event) {
  event.preventDefault();
  if (busy || micSession || mode !== "file") return;
  const file = fileInput.files[0];
  if (!file) return setStatus("請先選擇 WAV、MP3 或 M4A 音訊。", true);
  if (!/\.(wav|mp3|m4a)$/i.test(file.name)) return setStatus("支援 WAV、MP3 和 M4A 格式。", true);
  const separating = fileEngine.value === "separate";
  if (separating && Number.isFinite(originalAudio.duration) && originalAudio.duration > settings.maxSeparationDurationSeconds) {
    return setStatus("人聲分離每次最長 " + settings.maxSeparationDurationSeconds + " 秒。", true);
  }
  if (file.size > settings.maxUploadBytes) return setStatus("檔案超過 25 MB。", true);
  const data = new FormData();
  data.append("file", file);
  data.append("strength", String(Number(strength.value) / 100));
  data.append("humFilter", humFilter.value);
  busy = true;
  updateControls();
  clearResult();
  setStatus("上傳並處理中…");
  let quotaFull = false;
  try {
    const response = await apiFetch(separating ? "/api/separate" : "/api/denoise", { method: "POST", body: data, signal: AbortSignal.timeout(separating ? 240000 : 180000) });
    quotaFull = response.status === 507;
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "音訊處理失敗。");
    if (separating) {
      if (!Array.isArray(result.sources) || result.sources.length !== 2) throw new Error("人聲分離結果不完整。");
      separatedResults = result.sources;
      $("#source-select").value = "0";
      $("#source-option").hidden = false;
      await displayResult(separatedResults[0]);
    } else await displayResult(result);
    setStatus("處理完成");
  } catch (error) {
    setStatus(quotaFull ? "儲存空間不足，請刪除舊音訊。" : error.name === "TimeoutError" ? "處理逾時，請稍後重試。" : error.message, true);
  } finally {
    busy = false;
    updateControls();
  }
  if (quotaFull) await openRecordings();
}

function stopMonitors(session) {
  if (!session) return;
  for (const source of session.playing) {
    try { source.stop(); } catch { /* Source may already have ended. */ }
  }
  session.playing.clear();
  session.nextPlayTime = 0;
}

function monitorChunk(session, original, denoised) {
  if (!monitorToggle.checked || session.stopping) return;
  const selected = $('input[name="monitor-mode"]:checked').value;
  const samples = selected === "original" ? original : denoised;
  const ctx = getAudioContext();
  if (session.nextPlayTime > ctx.currentTime + 0.75) stopMonitors(session);
  const buffer = ctx.createBuffer(1, samples.length, session.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  session.playing.add(source);
  source.onended = () => { session.playing.delete(source); source.disconnect(); };
  const start = Math.max(ctx.currentTime + 0.06, session.nextPlayTime);
  source.start(start);
  session.nextPlayTime = start + buffer.duration;
}

function disposeMic(session) {
  if (!session || session.disposed) return;
  session.disposed = true;
  clearTimeout(session.timeout);
  clearTimeout(session.flushTimeout);
  clearTimeout(session.strengthTimeout);
  stopMonitors(session);
  if (session.worklet) {
    session.worklet.port.onmessage = null;
    session.worklet.disconnect();
  }
  if (session.source) session.source.disconnect();
  if (session.stream) session.stream.getTracks().forEach((track) => track.stop());
  if (session.socket && session.socket.readyState < WebSocket.CLOSING) session.socket.close();
  if (micSession === session) micSession = null;
  updateControls();
}

function failMic(session, message) {
  if (session.disposed || session.complete) return;
  disposeMic(session);
  $("#live-latency").textContent = "已停止";
  setStatus(message, true);
}

function finalizeMic(session) {
  if (session.finalized || session.disposed) return;
  session.finalized = true;
  session.stopping = true;
  clearTimeout(session.flushTimeout);
  if (session.stream) session.stream.getTracks().forEach((track) => track.stop());
  stopMonitors(session);
  updateControls();
  setStatus("正在保存錄音…");
  if (session.socket.readyState !== WebSocket.OPEN) return failMic(session, "連線已中斷。");
  session.socket.send(JSON.stringify({ type: "stop" }));
  session.timeout = setTimeout(() => failMic(session, "保存逾時，請重新收音。"), 30000);
}

function stopMic() {
  const session = micSession;
  if (!session || session.stopping) return;
  if (!session.ready) {
    disposeMic(session);
    setStatus("已取消");
    return;
  }
  session.stopping = true;
  stopMonitors(session);
  updateControls();
  setStatus("正在結束收音…");
  session.worklet.port.postMessage({ type: "stop" });
  session.flushTimeout = setTimeout(() => finalizeMic(session), 1000);
}

async function handleLiveMessage(session, event) {
  if (session.disposed) return;
  try {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        clearTimeout(session.timeout);
        session.worklet = new AudioWorkletNode(getAudioContext(), "microphone-capture", {
          processorOptions: {
            chunkSeconds: settings.chunkSeconds,
            maxDurationSeconds: message.maxDurationSeconds,
          },
        });
        session.worklet.port.onmessage = (capture) => {
          if (session.disposed || session.finalized) return;
          if (capture.data.type === "stopped") return finalizeMic(session);
          const samples = new Float32Array(capture.data.buffer);
          if (session.pending.length >= 8 || session.socket.bufferedAmount > 128000) {
            return failMic(session, "連線跟不上收音，請確認網路後重新開始。");
          }
          if (session.socket.readyState !== WebSocket.OPEN) return failMic(session, "伺服器連線已中斷。");
          session.pending.push({ samples, sentAt: performance.now() });
          session.socket.send(samples.buffer);
          session.totalSamples += samples.length;
          const seconds = Math.floor(session.totalSamples / session.sampleRate);
          $("#recording-time").value = String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0");
          $("#metric-duration").textContent = formatTime(seconds);
          drawSamples(samples, "original-waveform", "#758bb3");
        };
        session.source = getAudioContext().createMediaStreamSource(session.stream);
        // The capture worklet emits silence; monitoring uses returned server audio.
        session.source.connect(session.worklet);
        session.worklet.connect(getAudioContext().destination);
        session.ready = true;
        $("#metric-rate").textContent = session.sampleRate + " Hz";
        $("#original-meta").textContent = "麥克風 · " + session.sampleRate + " Hz";
        $("#denoised-meta").textContent = "收音中";
        updateControls();
        setStatus("收音中");
      } else if (message.type === "complete") {
        if (session.pending.length) throw new Error("音訊回傳不完整，請重新收音。");
        session.complete = true;
        busy = true;
        disposeMic(session);
        $("#live-latency").textContent = "錄音已保存";
        await displayResult(message);
        busy = false;
        updateControls();
        setStatus("錄音已保存");
      } else if (message.type === "error") {
        failMic(session, message.detail);
      }
    } else {
      const cleaned = new Float32Array(event.data);
      let offset = 0;
      // Neural inference buffers a short look-ahead; replies need not match input chunks.
      while (offset < cleaned.length) {
        const pending = session.pending[0];
        if (!pending) throw new Error("音訊串流不完整。");
        const used = pending.offset || 0;
        const count = Math.min(pending.samples.length - used, cleaned.length - offset);
        const output = cleaned.subarray(offset, offset + count);
        $("#live-latency").textContent = "往返 " + Math.round(performance.now() - pending.sentAt) + " ms · 最長 " + settings.maxDurationSeconds / 60 + " 分鐘";
        drawSamples(output, "denoised-waveform", "#168477");
        monitorChunk(session, pending.samples.subarray(used, used + count), output);
        pending.offset = used + count;
        offset += count;
        if (pending.offset === pending.samples.length) session.pending.shift();
      }
    }
  } catch (error) {
    busy = false;
    if (session.complete) { updateControls(); setStatus(error.message, true); }
    else failMic(session, error.message);
  }
}

async function startMic() {
  if (busy || micSession) return;
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return setStatus("麥克風需要 HTTPS 網址或 localhost。", true);
  }
  const session = { pending: [], playing: new Set(), totalSamples: 0, nextPlayTime: 0 };
  micSession = session;
  updateControls();
  clearResult();
  $("#session-name").textContent = "麥克風錄音";
  $("#recording-time").value = "00:00";
  $("#live-latency").textContent = "正在連線 · 最長 " + settings.maxDurationSeconds / 60 + " 分鐘";
  setStatus("正在連線並取得麥克風…");
  try {
    const ctx = getAudioContext();
    await ctx.resume();
    await ensureSession(true);
    if (!ctx.audioWorklet) throw new Error("瀏覽器不支援即時收音，請使用新版 Chrome、Edge 或 Safari。");
    if (!workletLoaded) {
      await ctx.audioWorklet.addModule("/capture-worklet.js");
      workletLoaded = true;
    }
    if (session.disposed) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    if (session.disposed) { stream.getTracks().forEach((track) => track.stop()); return; }
    session.stream = stream;
    stream.getAudioTracks()[0].addEventListener("ended", () => {
      if (!session.disposed && !session.stopping) failMic(session, "麥克風已中斷，請重新開始。");
    });
    session.sampleRate = ctx.sampleRate;
    const url = new URL("/api/live", window.location.href);
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    session.socket = new WebSocket(url);
    session.socket.binaryType = "arraybuffer";
    session.socket.onopen = () => session.socket.send(JSON.stringify({
      type: "start", sampleRate: session.sampleRate, strength: Number(strength.value) / 100,
      humFilter: humFilter.value,
    }));
    session.socket.onmessage = (event) => { void handleLiveMessage(session, event); };
    session.socket.onerror = () => failMic(session, "無法連到收音伺服器，請稍後重試。");
    session.socket.onclose = () => {
      if (!session.complete && !session.disposed) failMic(session, "連線已中斷，請重新收音。");
    };
    session.timeout = setTimeout(() => failMic(session, "伺服器連線逾時，請稍後重試。"), 30000);
  } catch (error) {
    const messages = {
      NotAllowedError: "麥克風權限未開啟，請在瀏覽器允許此網站使用麥克風。",
      NotFoundError: "找不到麥克風。",
      NotReadableError: "麥克風無法使用，請確認是否被其他程式占用。",
    };
    failMic(session, messages[error.name] || error.message);
  }
}

async function showFileName() {
  const version = ++selectionVersion;
  const file = fileInput.files[0];
  $("#file-label").textContent = file ? file.name : "選擇音訊檔";
  $("#dropzone").classList.toggle("has-file", !!file);
  clearResult();
  if (!file) {
    $("#session-name").textContent = "未命名音訊";
    $("#file-detail").textContent = "WAV · MP3 · M4A";
    return;
  }
  const format = file.name.split(".").pop().toUpperCase();
  const size = file.size < 1024 * 1024 ? Math.round(file.size / 1024) + " KB" : (file.size / 1024 / 1024).toFixed(1) + " MB";
  $("#file-detail").textContent = format + " · " + size;
  $("#session-name").textContent = file.name;
  if (!/\.(wav|mp3|m4a)$/i.test(file.name) || file.size > settings.maxUploadBytes) {
    setStatus("請選擇 25 MB 以內的 WAV、MP3 或 M4A 音訊。", true);
    return;
  }
  previewUrl = URL.createObjectURL(file);
  loadTrack("original", previewUrl);
  $("#original-meta").textContent = format + " · " + size;
  setStatus("音訊已載入");
  void drawWaveform(previewUrl, "original-waveform", trackColors.original);
  if (version !== selectionVersion) return;
  if (Number.isFinite(originalAudio.duration)) $("#metric-duration").textContent = formatTime(originalAudio.duration);
}

for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
  $("#dropzone").addEventListener(name, (event) => {
    event.preventDefault();
    $("#dropzone").classList.toggle("dragging", name === "dragenter" || name === "dragover");
    if (name !== "drop" || busy || micSession || !event.dataTransfer.files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    fileInput.files = transfer.files;
    showFileName();
  });
}
for (const name of ["file", "mic"]) {
  $("#" + name + "-tab").addEventListener("click", () => setMode(name));
  $("#" + name + "-tab").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? "file" : event.key === "End" ? "mic" : mode === "file" ? "mic" : "file";
    setMode(next);
    $("#" + next + "-tab").focus();
  });
}
fileInput.addEventListener("change", showFileName);
form.addEventListener("submit", submitAudio);
recordButton.addEventListener("click", () => micSession ? stopMic() : void startMic());
strength.addEventListener("input", () => {
  updateStrength();
  if (micSession && micSession.ready && !micSession.stopping && micSession.socket.readyState === WebSocket.OPEN) {
    const session = micSession;
    session.nextStrength = Number(strength.value) / 100;
    if (!session.strengthTimeout) session.strengthTimeout = setTimeout(() => {
      session.strengthTimeout = null;
      if (!session.disposed && !session.stopping && session.socket.readyState === WebSocket.OPEN) {
        session.socket.send(JSON.stringify({ type: "strength", value: session.nextStrength }));
      }
    }, 100);
  }
});
monitorToggle.addEventListener("change", () => { stopMonitors(micSession); updateControls(); });
document.querySelectorAll('input[name="monitor-mode"]').forEach((input) => {
  input.addEventListener("change", () => stopMonitors(micSession));
});
for (const [player, other] of [[originalAudio, denoisedAudio], [denoisedAudio, originalAudio]]) {
  player.addEventListener("play", () => { other.pause(); stopMonitors(micSession); });
}

for (const kind of trackKinds) {
  const player = $("#" + kind + "-audio");
  const play = $('[data-play="' + kind + '"]');
  const seek = $('[data-seek="' + kind + '"]');
  const mute = $('[data-mute="' + kind + '"]');
  const speed = $('[data-speed="' + kind + '"]');

  for (const name of ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended", "volumechange", "ratechange", "emptied"]) {
    player.addEventListener(name, () => {
      if (name === "loadedmetadata" && !player.error && player.getAttribute("src")) {
        clearTimeout(mediaTimers[kind]);
        playbackStates[kind] = "ready";
        if (kind === "original" && !currentResult) $("#metric-duration").textContent = formatTime(player.duration);
      }
      updateTrackFeedback(kind);
    });
  }
  player.addEventListener("error", () => {
    if (!player.getAttribute("src")) return;
    clearTimeout(mediaTimers[kind]);
    playbackStates[kind] = "error";
    updateTrackFeedback(kind);
  });
  $('[data-retry="' + kind + '"]').addEventListener("click", () => retryTrack(kind));
  play.addEventListener("click", async () => {
    if (play.disabled || busy || micSession) return;
    if (!player.paused && !player.ended) {
      player.pause();
      return;
    }
    try {
      if (player.ended) player.currentTime = 0;
      await player.play();
    } catch (error) {
      if (error.name !== "AbortError") setStatus("音訊無法播放，請重新載入音訊。", true);
    }
  });
  seek.addEventListener("input", () => {
    if (seek.disabled || !Number.isFinite(player.duration)) return;
    player.currentTime = Number(seek.value) / 1000 * player.duration;
    syncPlayer(kind);
  });
  mute.addEventListener("click", () => {
    if (mute.disabled) return;
    player.muted = !player.muted;
    syncPlayer(kind);
  });
  speed.addEventListener("change", () => {
    if (speed.disabled) return;
    player.playbackRate = Number(speed.value);
    syncPlayer(kind);
  });
  $("#" + kind + "-waveform").addEventListener("click", (event) => {
    if (seek.disabled || !Number.isFinite(player.duration)) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width) return;
    player.currentTime = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)) * player.duration;
    syncPlayer(kind);
  });
}

document.querySelectorAll("[data-strength]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.disabled) return;
    strength.value = button.dataset.strength;
    strength.dispatchEvent(new Event("input", { bubbles: true }));
  });
});
$("#sample-button").addEventListener("click", () => { void loadSample(); });
fileEngine.addEventListener("change", () => {
  updateFileLimit();
  updateControls();
});
$("#source-select").addEventListener("change", () => {
  if (busy || micSession) return;
  const selected = separatedResults[Number($("#source-select").value)];
  if (selected) void displayResult(selected);
});
$("#recordings-button").addEventListener("click", () => { void openRecordings(); });
$("#recordings-close").addEventListener("click", () => $("#recordings-dialog").close());
$("#recordings-clear").addEventListener("click", () => { $("#recordings-confirmation").hidden = false; });
$("#recordings-cancel").addEventListener("click", () => { $("#recordings-confirmation").hidden = true; });
$("#recordings-confirm").addEventListener("click", () => { void deleteRecording(); });
window.addEventListener("pagehide", () => disposeMic(micSession));
document.addEventListener("visibilitychange", () => { if (document.hidden && micSession) stopMic(); });

async function loadConfig() {
  try {
    await ensureSession();
    const response = await fetch("/api/config", { signal: AbortSignal.timeout(90000) });
    if (!response.ok) throw new Error("Server unavailable");
    settings = await response.json();
    $("#engine-option").hidden = !settings.separationEnabled;
    updateFileLimit();
    return true;
  } catch {
    if (!busy && !micSession && !currentResult) setStatus("暫時無法連線，請稍後再試。", true);
    return false;
  }
}
if (window.lucide) window.lucide.createIcons();
clearResult();
updateStrength();
updateControls();
const appReady = loadConfig();
