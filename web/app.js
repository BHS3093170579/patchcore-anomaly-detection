/* ============================================================
   无监督缺陷检测系统 · 前端逻辑
   每个检测任务在前端也保持独立状态（lastInput / lastResult / history），
   切换任务只切换视图，不共享任何结果对象。
   ============================================================ */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const API = {
  tasks: () => fetchJSON("/api/tasks"),
  state: (t) => fetchJSON(`/api/tasks/${t}/state`),
  history: (t, limit = 50) => fetchJSON(`/api/tasks/${t}/history?limit=${limit}`),
  model: (t) => fetchJSON(`/api/tasks/${t}/model`),
  metrics: (t) => fetchJSON(`/api/tasks/${t}/metrics`),
  anomalies: (t, limit = 18) => fetchJSON(`/api/tasks/${t}/anomalies?limit=${limit}`),
  clear: (t) => fetchJSON(`/api/tasks/${t}/clear`, { method: "POST" }),
  setThreshold: (t, body) => fetchJSON(`/api/tasks/${t}/threshold`, { method: "POST", body }),
  reloadModel: (t) => fetchJSON(`/api/tasks/${t}/model/load`, { method: "POST" }),
  buildModel: (t) => fetchJSON(`/api/tasks/${t}/model/build`, { method: "POST" }),
  infer: (t, file, saveAnomaly = true) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("save_anomaly", String(saveAnomaly));
    return fetchJSON(`/api/infer/${t}`, { method: "POST", body: fd });
  },
  frame: (t, image, quality = 72) =>
    fetchJSON(`/api/infer/${t}/frame`, { method: "POST", body: { image, overlay_quality: quality } }),
  uploadVideo: (t, file, fps) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("target_fps", String(fps));
    return fetchJSON(`/api/tasks/${t}/video`, { method: "POST", body: fd });
  },
  video: (id) => fetchJSON(`/api/video/${id}`),
  cancelVideo: (id) => fetchJSON(`/api/video/${id}/cancel`, { method: "POST" }),
};

async function fetchJSON(url, opts = {}) {
  const init = { ...opts };
  if (opts.body !== undefined && !(opts.body instanceof FormData)) {
    init.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch(url, init);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`;
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ---------------- 全局状态 ---------------- */
const S = {
  tasks: [],
  task: null,          // 当前任务名
  source: "image",
  view: "overlay",
  perTask: {},         // task -> { lastInput, lastResult, history, threshold }
  cam: { running: false, busy: false, stream: null, frames: 0, tPrev: 0, emaDt: 0, timer: null },
  video: { file: null, jobId: null, poll: null, frames: [] },
  stats: { count: 0, anomaly: 0, latSum: 0, latN: 0, fps: 0, e2e: 0 },
  regionFilter: "all",
};

function perTask(t = S.task) {
  if (!S.perTask[t]) S.perTask[t] = { lastInput: null, lastResult: null, history: [], threshold: null, model: null };
  return S.perTask[t];
}

function activeTaskInfo() {
  return S.tasks.find((x) => x.task === S.task) || null;
}

/* ---------------- Toast ---------------- */
let toastSeq = 0;
function toast(msg, kind = "info", ms = 3200) {
  const wrap = $("#toastWrap");
  const el = document.createElement("div");
  el.className = `toast ${kind === "error" ? "is-error" : kind === "ok" ? "is-ok" : kind === "warn" ? "is-warn" : ""}`;
  el.textContent = msg;
  el.dataset.id = String(++toastSeq);
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .25s ease, transform .25s ease";
    el.style.opacity = "0";
    el.style.transform = "translateY(6px)";
    setTimeout(() => el.remove(), 260);
  }, ms);
  return el;
}

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(d);
}

/* ---------------- 任务切换 ---------------- */
async function loadTasks() {
  const data = await API.tasks();
  S.tasks = data.tasks || [];
  renderTaskSwitch();
  const urlTask = new URLSearchParams(location.search).get("task");
  const saved = (urlTask && S.tasks.some((x) => x.task === urlTask)) ? urlTask : localStorage.getItem("quex.task");
  const pick = S.tasks.find((x) => x.task === saved) || S.tasks[0];
  if (pick) await selectTask(pick.task);
  else setEmptyAll("没有可用任务配置（configs/*.json）");
}

function renderTaskSwitch() {
  const nav = $("#taskSwitch");
  nav.innerHTML = "";
  for (const t of S.tasks) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "task-btn";
    btn.dataset.task = t.task;
    btn.style.setProperty("--swatch", t.accent || "#94a3b8");
    const sw = document.createElement("span");
    sw.className = "swatch";
    const name = document.createElement("span");
    name.className = "task-name";
    name.textContent = t.display_name || t.task;
    btn.append(sw, name);
    btn.addEventListener("click", () => selectTask(t.task));
    nav.appendChild(btn);
  }
}

function applyAccent(hex) {
  const root = document.documentElement;
  const color = hex || "#2563eb";
  root.style.setProperty("--accent", color);
  root.style.setProperty("--accent-ink", shade(color, -22));
  root.style.setProperty("--accent-soft", mix(color, "#ffffff", 0.88));
}

function hexToRgb(h) {
  const s = h.replace("#", "");
  const v = s.length === 3 ? s.split("").map((c) => c + c).join("") : s;
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}
function rgbToHex(r, g, b) {
  const f = (x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0");
  return `#${f(r)}${f(g)}${f(b)}`;
}
function shade(hex, pct) {
  const [r, g, b] = hexToRgb(hex);
  const k = 1 + pct / 100;
  return rgbToHex(r * k, g * k, b * k);
}
function mix(a, b, ratio) {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  return rgbToHex(r1 * (1 - ratio) + r2 * ratio, g1 * (1 - ratio) + g2 * ratio, b1 * (1 - ratio) + b2 * ratio);
}

async function selectTask(task) {
  if (!task || task === S.task) {
    if (S.task) refreshTaskView();
    return;
  }
  stopCamera(true);
  resetVideoUi();
  S.task = task;
  localStorage.setItem("quex.task", task);
  const info = activeTaskInfo();
  applyAccent(info ? info.accent : "#2563eb");

  $$("#taskSwitch .task-btn").forEach((b) => b.classList.toggle("is-active", b.dataset.task === task));
  $("#thrNote").textContent = `阈值写入该任务自己的 threshold.json 与 configs/${task}.json，另一任务不受影响。`;

  // 恢复该任务自己的最近输入/结果
  const pt = perTask();
  if (!pt.lastResult) {
    try {
      const st = await API.state(task);
      pt.lastResult = st.last_result || null;
      pt.lastInput = st.last_input || null;
      if (st.params) pt.params = st.params;
    } catch (e) {
      toast(`读取任务状态失败：${e.message}`, "error");
    }
  }
  try {
    const h = await API.history(task, 40);
    pt.history = h.items || [];
  } catch { pt.history = []; }

  renderThresholdInputs();
  renderModelPanel();
  renderHistory();
  renderViewer();
  renderResult();
  renderStats();
  refreshModelPill();
  loadMetrics();
  loadAnomalies();

  // 后台拉模型信息（不阻塞界面）
  API.model(task).then((m) => { perTask().model = m; renderModelPanel(); refreshModelPill(); })
    .catch(() => {});
}

function refreshTaskView() {
  renderThresholdInputs();
  renderModelPanel();
  renderHistory();
  renderViewer();
  renderResult();
  renderStats();
}

function setEmptyAll(msg) {
  $("#viewerEmpty").hidden = false;
  $("#viewerImg").hidden = true;
  $("#viewerBadge").hidden = true;
  $("#regionList").innerHTML = `<div class="muted-line">${msg || "暂无数据"}</div>`;
}

/* ---------------- 输入源 tabs ---------------- */
function bindSourceTabs() {
  $$("#sourceTabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const src = tab.dataset.source;
      if (src === S.source) return;
      if (S.source === "camera" && src !== "camera") stopCamera();
      S.source = src;
      $$("#sourceTabs .tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      $$(".pane").forEach((p) => p.classList.toggle("is-active", p.dataset.pane === src));
      $("#modePill").textContent = src === "image" ? "图片" : src === "video" ? "视频" : "摄像头";
      $("#sourceHint").textContent = src === "image" ? "单张图片推理" : src === "video" ? "抽帧批量推理" : "实时帧推理";
    });
  });
}

/* ---------------- 图片 ---------------- */
function bindImage() {
  const input = $("#imageInput");
  const drop = $("#imageDrop");

  drop.addEventListener("click", (e) => {
    if (e.target.tagName !== "INPUT") input.click();
  });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) runImage(input.files[0]);
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) runImage(f);
  });
  $("#imageClearBtn").addEventListener("click", () => {
    const pt = perTask();
    pt.lastInput = null;
    pt.lastResult = null;
    renderViewer();
    renderResult();
    toast("已清除当前任务的图片与结果", "ok");
  });
}

async function runImage(file) {
  if (!S.task) return;
  if (!/^image\//.test(file.type) && !/\.(png|jpe?g|bmp|webp)$/i.test(file.name)) {
    toast("请选择图片文件（PNG / JPG / BMP / WebP）", "error");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    toast("图片超过 20MB 上限", "error");
    return;
  }
  $("#viewerLoading").hidden = false;
  const t0 = performance.now();
  try {
    const res = await API.infer(S.task, file, $("#saveAnomaly").checked);
    const dt = performance.now() - t0;
    const pt = perTask();
    pt.lastInput = { name: `图片: ${file.name}`, thumb: res.overlay, localUrl: URL.createObjectURL(file) };
    pt.lastResult = res;
    S.stats.count += 1;
    S.stats.latSum += res.latency_ms;
    S.stats.latN += 1;
    if (res.is_anomaly) S.stats.anomaly += 1;
    S.stats.e2e = dt;
    renderViewer();
    renderResult();
    renderStats();
    await reloadHistory();
    if (res.saved_to) loadAnomalies();
    toast(res.is_anomaly
      ? `检出异常：${res.defect_count} 个区域${res.saved_to ? "，已存档" : ""}`
      : "未检出异常",
      res.is_anomaly ? "warn" : "ok");
  } catch (e) {
    toast(`推理失败：${e.message}`, "error", 5000);
  } finally {
    $("#viewerLoading").hidden = true;
  }
}

/* ---------------- 摄像头 ---------------- */
function bindCamera() {
  $("#camStartBtn").addEventListener("click", startCamera);
  $("#camStopBtn").addEventListener("click", () => stopCamera());
}

async function startCamera() {
  if (!S.task) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("当前浏览器不支持摄像头，或页面不是 localhost/HTTPS", "error", 5200);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
      audio: false,
    });
    const video = $("#camVideo");
    video.srcObject = stream;
    await video.play();
    S.cam.stream = stream;
    S.cam.running = true;
    S.cam.busy = false;
    S.cam.frames = 0;
    S.cam.tPrev = 0;
    S.cam.emaDt = 0;
    $(".camera-wrap").classList.add("is-on");
    $("#cameraHud").hidden = false;
    $("#camStartBtn").disabled = true;
    $("#camStopBtn").disabled = false;
    toast("摄像头已启动，实时推理中", "ok");
    camLoop();
  } catch (e) {
    toast(`摄像头启动失败：${e.message}`, "error", 5200);
  }
}

function stopCamera(silent = false) {
  S.cam.running = false;
  if (S.cam.stream) {
    S.cam.stream.getTracks().forEach((t) => t.stop());
    S.cam.stream = null;
  }
  const video = $("#camVideo");
  if (video) video.srcObject = null;
  $(".camera-wrap")?.classList.remove("is-on");
  $("#cameraHud").hidden = true;
  $("#camStartBtn").disabled = false;
  $("#camStopBtn").disabled = true;
  if (!silent && S.source === "camera") toast("摄像头已停止", "info", 1800);
}

const camCanvas = document.createElement("canvas");

async function camLoop() {
  if (!S.cam.running || S.source !== "camera") return;
  if (S.cam.busy) { setTimeout(camLoop, 30); return; }
  const video = $("#camVideo");
  if (!video.videoWidth) { setTimeout(camLoop, 60); return; }

  S.cam.busy = true;
  const now = performance.now();
  try {
    const maxSide = 720;
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    camCanvas.width = Math.round(video.videoWidth * scale);
    camCanvas.height = Math.round(video.videoHeight * scale);
    const ctx = camCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, camCanvas.width, camCanvas.height);
    const dataUrl = camCanvas.toDataURL("image/jpeg", 0.8);

    const res = await API.frame(S.task, dataUrl, 70);
    const dt = performance.now() - now;

    // 只写本任务视图（其他任务状态不受影响）
    const pt = perTask();
    pt.lastResult = res;
    pt.lastInput = { name: "摄像头实时帧", thumb: res.overlay };

    S.stats.count += 1;
    S.stats.latSum += res.latency_ms;
    S.stats.latN += 1;
    S.stats.e2e = dt;
    if (res.is_anomaly) S.stats.anomaly += 1;

    // FPS：帧完成率的指数滑动平均
    if (S.cam.tPrev) {
      const gap = now - S.cam.tPrev;
      S.cam.emaDt = S.cam.emaDt ? S.cam.emaDt * 0.8 + gap * 0.2 : gap;
    }
    S.cam.tPrev = now;
    S.cam.frames += 1;
    S.stats.fps = S.cam.emaDt ? 1000 / S.cam.emaDt : 0;

    renderViewerLive(res);
    renderResultLive(res);
    $("#hudFps").textContent = `${fmt(S.stats.fps, 1)} fps`;
    $("#hudLatency").textContent = `${Math.round(dt)} ms`;
    renderStats();
  } catch (e) {
    $("#hudFps").textContent = "错误";
    $("#hudLatency").textContent = e.message.slice(0, 18);
    S.stats.fps = 0;
  } finally {
    S.cam.busy = false;
    if (S.cam.running && S.source === "camera") setTimeout(camLoop, 0);
  }
}

/* ---------------- 视频 ---------------- */
function resetVideoUi() {
  $("#videoProgress").hidden = true;
  $("#videoCancelBtn").hidden = true;
  $("#videoStartBtn").disabled = true;
  $("#frameStrip").hidden = true;
  $("#frameStrip").innerHTML = "";
  if (S.video.poll) { clearInterval(S.video.poll); S.video.poll = null; }
  S.video = { file: null, jobId: null, poll: null, frames: [] };
}

function bindVideo() {
  const input = $("#videoInput");
  const drop = $("#videoDrop");
  drop.addEventListener("click", (e) => { if (e.target.tagName !== "INPUT") input.click(); });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      S.video.file = input.files[0];
      $("#videoStartBtn").disabled = false;
      $("#frameStrip").hidden = true;
      drop.querySelector("strong").textContent = input.files[0].name;
      drop.querySelector("small").textContent = `${(input.files[0].size / 1024 / 1024).toFixed(1)} MB · 点击可重新选择`;
    }
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) {
      S.video.file = f;
      $("#videoStartBtn").disabled = false;
      drop.querySelector("strong").textContent = f.name;
    }
  });

  $("#videoStartBtn").addEventListener("click", startVideo);
  $("#videoCancelBtn").addEventListener("click", async () => {
    if (!S.video.jobId) return;
    try { await API.cancelVideo(S.video.jobId); toast("已请求取消视频分析", "warn"); }
    catch (e) { toast(`取消失败：${e.message}`, "error"); }
  });
}

async function startVideo() {
  if (!S.task || !S.video.file) { toast("请先选择视频文件", "error"); return; }
  const fps = Math.max(0.5, Math.min(30, Number($("#videoFps").value) || 3));
  $("#videoStartBtn").disabled = true;
  $("#videoProgress").hidden = false;
  $("#videoBar").style.width = "2%";
  $("#videoProgressText").textContent = "上传视频…";
  try {
    const r = await API.uploadVideo(S.task, S.video.file, fps);
    S.video.jobId = r.job.id;
    $("#videoCancelBtn").hidden = false;
    $("#videoProgressText").textContent = "已提交，等待开始…";
    S.video.poll = setInterval(pollVideo, 600);
    pollVideo();
  } catch (e) {
    $("#videoStartBtn").disabled = false;
    $("#videoProgressText").textContent = `失败：${e.message}`;
    toast(`视频上传失败：${e.message}`, "error", 5200);
  }
}

async function pollVideo() {
  if (!S.video.jobId) return;
  try {
    const job = await API.video(S.video.jobId);
    const done = job.frames_done || 0;
    const total = job.frames_total || 0;
    const pct = total ? Math.min(100, (done / total) * 100) : 8;
    $("#videoBar").style.width = `${Math.max(2, pct)}%`;
    $("#videoProgressText").textContent =
      `${job.status === "running" ? "分析中" : job.status === "done" ? "完成" : job.status} · ${done}/${total || "?"} 帧` +
      (job.p95_ms ? ` · P95 ${job.p95_ms} ms` : "");

    if (job.status === "done" || job.status === "error" || job.status === "canceled") {
      clearInterval(S.video.poll);
      S.video.poll = null;
      $("#videoCancelBtn").hidden = true;
      $("#videoStartBtn").disabled = false;
      if (job.status === "error") { toast(`视频分析失败：${job.message}`, "error", 5200); return; }
      if (job.status === "canceled") { toast("视频分析已取消", "warn"); return; }
      S.video.frames = job.frames || [];
      renderFrameStrip();
      toast(`视频分析完成：${job.frames_done} 帧，${job.anomaly_frames} 帧异常`, "ok");
      if (S.video.frames.length) showVideoFrame(0);
    }
  } catch (e) {
    clearInterval(S.video.poll);
    S.video.poll = null;
    toast(`查询视频进度失败：${e.message}`, "error");
  }
}

function renderFrameStrip() {
  const strip = $("#frameStrip");
  strip.innerHTML = "";
  if (!S.video.frames.length) { strip.hidden = true; return; }
  strip.hidden = false;
  const base = `/media/${S.task}/videos/${S.video.jobId}`;
  S.video.frames.forEach((f, i) => {
    if (!f.overlay) return;
    const img = document.createElement("img");
    img.src = `${base}/${f.overlay}`;
    img.alt = `帧 ${f.t_s}s`;
    img.title = `${f.t_s}s · 分数 ${f.score}${f.is_anomaly ? " · 异常" : ""}`;
    img.addEventListener("click", () => showVideoFrame(i));
    strip.appendChild(img);
  });
}

function showVideoFrame(i) {
  const f = S.video.frames[i];
  if (!f) return;
  $$("#frameStrip img").forEach((im, k) => im.classList.toggle("is-active", k === i));
  const overlayUrl = `/media/${S.task}/videos/${S.video.jobId}/${f.overlay}`;
  const pt = perTask();
  pt.lastResult = { ...f, overlay: overlayUrl, source: `视频帧 ${f.t_s}s`, image_size: null };
  pt.lastInput = { name: `视频帧 ${f.t_s}s`, thumb: overlayUrl };
  S.stats.count += 1;
  S.stats.latSum += f.latency_ms || 0;
  S.stats.latN += 1;
  renderViewer();
  renderResult();
  renderStats();
}

/* ---------------- 视图渲染 ---------------- */
function renderViewerLive(res) { renderViewer(res); }

function renderViewer(resOverride) {
  const pt = perTask();
  const res = resOverride || pt.lastResult;
  const img = $("#viewerImg");
  const empty = $("#viewerEmpty");
  const badge = $("#viewerBadge");

  if (!res) {
    img.hidden = true;
    empty.hidden = false;
    badge.hidden = true;
    $("#viewer").classList.add("is-empty");
    $("#viewerMeta").textContent = "—";
    $("#viewerSize").textContent = "";
    return;
  }
  const overlaySrc = res.overlay || (pt.lastInput && pt.lastInput.thumb) || "";
  const origSrc = res.original || (pt.lastInput && (pt.lastInput.localUrl || pt.lastInput.thumb)) || overlaySrc;
  const src = S.view === "overlay" ? overlaySrc : origSrc;
  if (!src) { img.hidden = true; empty.hidden = false; $("#viewer").classList.add("is-empty"); return; }
  img.src = src;
  img.hidden = false;
  empty.hidden = true;
  $("#viewer").classList.remove("is-empty");
  badge.hidden = false;
  badge.textContent = res.is_anomaly ? `异常 · ${res.defect_count ?? 0} 区域` : "正常";
  badge.className = "viewer-badge " + (res.is_anomaly ? "is-bad" : "is-ok");
  $("#viewerMeta").textContent = res.source || (pt.lastInput && pt.lastInput.name) || "—";
  const sz = res.image_size;
  $("#viewerSize").textContent = sz ? `${sz[0]} × ${sz[1]}` : "";
}

function renderResultLive(res) { renderResult(res); }

function renderResult(override) {
  const pt = perTask();
  const res = override || pt.lastResult;
  const verdict = $("#verdict");
  verdict.classList.remove("is-bad", "is-ok");
  if (!res) {
    $("#verdictLabel").textContent = "等待检测";
    $("#verdictScore").textContent = "—";
    $("#verdictThr").textContent = "";
    $("#mRegions").textContent = "—";
    $("#mLatency").textContent = "—";
    $("#mInputSize").textContent = "—";
    $("#resultTime").textContent = "—";
    $("#regionFilter").hidden = true;
    $("#regionList").innerHTML = `<div class="muted-line">暂无区域数据</div>`;
    return;
  }
  verdict.classList.add(res.is_anomaly ? "is-bad" : "is-ok");
  $("#verdictLabel").textContent = res.is_anomaly ? "检出异常" : "判定正常";
  $("#verdictScore").textContent = fmt(res.score, 3);
  $("#verdictThr").textContent = res.threshold !== undefined ? `阈值 ${fmt(res.threshold, 3)}` : "";
  $("#mRegions").textContent = res.is_anomaly
    ? (res.defect_count ?? 0)
    : (res.defect_count ? `${res.defect_count}（疑似）` : 0);
  $("#mLatency").textContent = res.latency_ms !== undefined ? `${Math.round(res.latency_ms)} ms` : "—";
  $("#mInputSize").textContent = res.input_size ? `${res.input_size}²` : (res.image_size ? res.image_size.join("×") : "—");
  $("#resultTime").textContent = res.timestamp || (res.t_s !== undefined ? `视频 ${res.t_s}s` : "—");
  $("#regionFilter").hidden = !(res.regions && res.regions.length);
  renderRegions(res);
}

function regionLevel(r, res) {
  const thr = res.pixel_threshold || res.threshold || 1;
  const peak = r.peak || 0;
  if (peak >= thr * 1.35) return "high";
  if (peak >= thr * 1.0) return "mid";
  return "low";
}

function renderRegions(res) {
  const list = $("#regionList");
  const regions = res.regions || [];
  if (!regions.length) {
    list.innerHTML = `<div class="muted-line">未检出超阈值区域</div>`;
    return;
  }
  const shown = regions.filter((r) => S.regionFilter === "all" || regionLevel(r, res) === S.regionFilter);
  if (!shown.length) {
    list.innerHTML = `<div class="muted-line">当前筛选下没有区域</div>`;
    return;
  }
  list.innerHTML = "";
  if (!res.is_anomaly) {
    const hint = document.createElement("div");
    hint.className = "muted-line";
    hint.textContent = "图像级判定为正常：以下是像素级疑似点，未达到图像级阈值，可结合阈值调节使用。";
    list.appendChild(hint);
  }
  shown.forEach((r, i) => {
    const lv = regionLevel(r, res);
    const row = document.createElement("div");
    row.className = "region-row";
    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = String(i + 1);
    const bbox = document.createElement("span");
    bbox.className = "bbox";
    bbox.textContent = `x${r.bbox[0]}–${r.bbox[2]}  y${r.bbox[1]}–${r.bbox[3]}`;
    const peak = document.createElement("span");
    peak.className = "peak";
    peak.textContent = fmt(r.peak, 3);
    const tag = document.createElement("span");
    tag.className = `tag ${lv}`;
    tag.textContent = lv === "high" ? "高分" : lv === "mid" ? "中分" : "低分";
    row.append(idx, bbox, peak, tag);
    list.appendChild(row);
  });
}

function renderHistory() {
  const pt = perTask();
  const list = $("#historyList");
  const items = pt.history || [];
  $("#historyCount").textContent = `${items.length} 条`;
  if (!items.length) {
    list.innerHTML = `<div class="muted-line">暂无历史记录</div>`;
    return;
  }
  list.innerHTML = "";
  items.forEach((it) => {
    const row = document.createElement("div");
    row.className = "history-row" + (it.is_anomaly ? " is-bad" : "");
    const mark = document.createElement("span");
    mark.className = "mark";
    const main = document.createElement("div");
    main.className = "history-main";
    const src = document.createElement("span");
    src.className = "src";
    src.textContent = it.source || "—";
    const tm = document.createElement("span");
    tm.className = "tm";
    tm.textContent = `${it.time || ""}${it.defect_count ? ` · ${it.defect_count} 区` : ""}`;
    main.append(src, tm);
    const sc = document.createElement("span");
    sc.className = "sc";
    sc.textContent = fmt(it.score, 3);
    row.append(mark, main, sc);
    list.appendChild(row);
  });
}

function renderStats() {
  const root = $("#statsKv");
  const avg = S.stats.latN ? S.stats.latSum / S.stats.latN : null;
  root.innerHTML = `
    <div><dt>推理次数</dt><dd>${S.stats.count}</dd></div>
    <div><dt>异常次数</dt><dd>${S.stats.anomaly}</dd></div>
    <div><dt>平均耗时</dt><dd>${avg ? fmt(avg, 1) + " ms" : "—"}</dd></div>
    <div><dt>实时链路 FPS</dt><dd>${S.stats.fps ? fmt(S.stats.fps, 1) : "—"}</dd></div>
    <div><dt>端到端延迟</dt><dd>${S.stats.e2e ? fmt(S.stats.e2e, 0) + " ms" : "—"}</dd></div>`;
}

async function loadMetrics() {
  const task = S.task;
  const grid = $("#metricGrid");
  const cls = $("#classList");
  try {
    const m = await API.metrics(task);
    if (task !== S.task) return;
    if (!m.available) {
      grid.innerHTML = `<div class="muted-line">暂无评估数据，运行 python -m det.eval --task ${task} 后显示</div>`;
      cls.hidden = true;
      $("#metricsHint").textContent = "离线评估";
      return;
    }
    const items = [
      ["图像 AUROC", m.image_auroc, 4],
      ["图像 AP", m.image_ap, 4],
      ["F1 最大", m.image_f1max, 4],
      ["像素 AUROC", m.pixel_auroc, 4],
      ["AUPRO", m.pixel_pro, 4],
      ["测试样本", m.n_images, 0],
    ];
    grid.innerHTML = items.map(([k, v, d]) =>
      `<div class="m-item"><span>${k}</span><b>${v == null ? "—" : fmt(v, d)}</b></div>`).join("");
    const cs = m.classes || [];
    cls.hidden = !cs.length;
    cls.innerHTML = cs.map((c) =>
      `<div class="class-row"><span>${c.name}</span><b>${c.detected ?? "—"}/${c.total ?? "—"}</b></div>`).join("");
    $("#metricsHint").textContent = `${m.n_images} 张测试图`;
  } catch (e) {
    grid.innerHTML = `<div class="muted-line">指标读取失败：${e.message}</div>`;
  }
}

async function loadAnomalies() {
  const task = S.task;
  const grid = $("#anomalyGrid");
  try {
    const d = await API.anomalies(task, 8);
    if (task !== S.task) return;
    $("#anomalyHint").textContent = `${d.total} 张`;
    const items = d.items || [];
    if (!items.length) {
      grid.innerHTML = `<div class="muted-line">暂无存档</div>`;
      return;
    }
    grid.innerHTML = "";
    items.forEach((it) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "anomaly-cell";
      cell.title = `${it.time || ""} · 峰值 ${fmt(it.score, 3)}`;
      const im = document.createElement("img");
      im.src = it.overlay_url || it.url;
      im.alt = it.name;
      im.loading = "lazy";
      cell.appendChild(im);
      cell.addEventListener("click", () => {
        const img = $("#viewerImg");
        const badge = $("#viewerBadge");
        img.src = it.overlay_url || it.url;
        img.hidden = false;
        $("#viewerEmpty").hidden = true;
        $("#viewer").classList.remove("is-empty");
        badge.hidden = false;
        badge.className = "viewer-badge is-bad";
        badge.textContent = `存档 · 峰值 ${fmt(it.score, 3)}`;
        $("#viewerMeta").textContent = `存档 ${it.time || it.name}`;
        $("#viewerSize").textContent = "";
        grid.querySelectorAll(".anomaly-cell.is-active").forEach((el) => el.classList.remove("is-active"));
        cell.classList.add("is-active");
      });
      grid.appendChild(cell);
    });
  } catch (e) {
    grid.innerHTML = `<div class="muted-line">存档读取失败：${e.message}</div>`;
  }
}

function renderThresholdInputs() {
  const info = activeTaskInfo();
  const pt = perTask();
  const cfgThr = (info && info.threshold) || {};
  $("#thrImage").value = cfgThr.image_score ?? "";
  $("#thrPixel").value = cfgThr.pixel_score ?? "";
  renderSensitivity();
}

function currentPresets() {
  const info = activeTaskInfo();
  return (info && info.threshold_presets) || {};
}

function renderSensitivity() {
  const presets = currentPresets();
  const sel = $("#sensitivity");
  const pt = perTask();
  const want = pt.sensitivity || "balanced";
  sel.value = presets[want] ? want : "balanced";
  const p = presets[sel.value];
  $("#sensMeta").textContent = p
    ? `召回 ${(p.recall * 100).toFixed(1)}% · 误报 ${p.false_alarm}/${p.n_normal}`
    : "—";
}

function renderModelPanel() {
  const info = activeTaskInfo();
  const pt = perTask();
  const m = pt.model || {};
  const hasModel = info ? info.model_ready : false;
  const kv = $("#modelKv");
  const modelFile = info && info.model_path ? String(info.model_path).split(/[\\/]/).pop() : "";
  const rows = [
    ["任务", info ? `${info.display_name}（${info.task}）` : S.task],
    ["模型文件", hasModel ? modelFile || "已就绪" : "缺失（需建模）"],
    ["特征库条数", m.bank_size ?? "—"],
    ["输入尺寸", info ? `${info.input_size}×${info.input_size}` : "—"],
    ["骨干网络", m.meta ? `${m.meta.backbone} · ${(m.meta.layers || []).join("+")}` : "wide_resnet50_2 · layer2+layer3"],
    ["建模样本", m.meta ? `${m.meta.n_train_images} 张（正常）` : "—"],
  ];
  kv.innerHTML = rows.map(([k, v]) => {
    const title = k === "模型文件" && info ? info.model_path : v;
    return `<div><dt>${k}</dt><dd title="${title}">${v}</dd></div>`;
  }).join("");
}

function refreshModelPill() {
  const info = activeTaskInfo();
  const pill = $("#modelPill");
  const statePill = $("#modelStatePill");
  const ok = info && info.model_ready;
  const st = info && info.state ? info.state.state : "idle";
  pill.className = "pill pill-dot" + (st === "building" || st === "loading" ? " is-busy" : ok ? " is-ok" : " is-bad");
  pill.innerHTML = `<i class="dot"></i>${st === "building" ? "建模中" : st === "loading" ? "加载中" : ok ? "模型就绪" : "模型缺失"}`;
  statePill.className = "pill pill-dot sm" + (ok ? " is-ok" : " is-bad");
  statePill.innerHTML = `<i class="dot"></i>${ok ? "已加载" : "未就绪"}`;
}

/* ---------------- 阈值 / 模型 / 历史操作 ---------------- */
function bindControls() {
  $("#thrApplyBtn").addEventListener("click", async () => {
    const img = Number($("#thrImage").value);
    const pix = Number($("#thrPixel").value);
    if (!Number.isFinite(img) || !Number.isFinite(pix) || img < 0 || pix < 0) {
      toast("阈值必须是 ≥ 0 的数字", "error");
      return;
    }
    try {
      const r = await API.setThreshold(S.task, { image_score: img, pixel_score: pix });
      toast(`阈值已保存：图像 ${fmt(r.threshold.image_score, 3)} / 像素 ${fmt(r.threshold.pixel_score, 3)}`, "ok");
      const info = activeTaskInfo();
      if (info) info.threshold = r.threshold;
      const pt = perTask();
      if (pt.lastResult) {
        pt.lastResult.threshold = r.threshold.image_score;
        renderResult();
        renderViewer();
      }
    } catch (e) {
      toast(`阈值保存失败：${e.message}`, "error", 5000);
    }
  });

  $("#thrResetBtn").addEventListener("click", () => {
    perTask().lastResult = perTask().lastResult;
    renderThresholdInputs();
    toast("已恢复为配置文件中的阈值（未改动服务器）", "info", 2600);
  });

  $("#sensitivity").addEventListener("change", async () => {
    const name = $("#sensitivity").value;
    const p = currentPresets()[name];
    if (!p) return;
    perTask().sensitivity = name;
    $("#thrImage").value = p.image;
    $("#thrPixel").value = p.pixel;
    try {
      const r = await API.setThreshold(S.task, {
        image_score: p.image,
        pixel_score: p.pixel,
        calibrated_from: `preset:${name}`,
      });
      const info = activeTaskInfo();
      if (info) info.threshold = r.threshold;
      const pt = perTask();
      if (pt.lastResult) {
        pt.lastResult.threshold = r.threshold.image_score;
        renderResult();
        renderViewer();
      }
      renderSensitivity();
      toast(`已切到「${p.label}」：召回 ${(p.recall * 100).toFixed(1)}%，误报 ${p.false_alarm}/${p.n_normal}`,
        name === "high_recall" ? "info" : "ok", 4200);
    } catch (e) {
      toast(`切换失败：${e.message}`, "error", 5000);
    }
  });

  $("#modelReloadBtn").addEventListener("click", async () => {
    try {
      toast("正在重新加载模型…", "info", 1500);
      const r = await API.reloadModel(S.task);
      toast(`模型已加载：特征库 ${r.bank_size} 条，耗时 ${r.load_ms} ms`, "ok", 4200);
      const m = await API.model(S.task);
      perTask().model = m;
      renderModelPanel();
      refreshModelPill();
    } catch (e) {
      toast(`模型加载失败：${e.message}`, "error", 6000);
    }
  });

  $("#modelBuildBtn").addEventListener("click", async () => {
    const ok = window.confirm(
      "将只用该任务的 train/good 正常样本重新建模，并覆盖现有模型与阈值。\n建模期间该任务无法推理（另一任务不受影响）。继续？"
    );
    if (!ok) return;
    try {
      await API.buildModel(S.task);
      toast("已开始后台建模，进度见右上角模型状态", "ok", 5000);
      setTimeout(async () => {
        const d = await API.tasks();
        S.tasks = d.tasks || [];
        renderTaskSwitch();
        $$("#taskSwitch .task-btn").forEach((b) => b.classList.toggle("is-active", b.dataset.task === S.task));
        refreshModelPill();
      }, 1500);
    } catch (e) {
      toast(`建模启动失败：${e.message}`, "error", 5000);
    }
  });

  $("#exportCsvBtn").addEventListener("click", () => download(`/api/tasks/${S.task}/export?fmt=csv`));
  $("#exportJsonBtn").addEventListener("click", () => download(`/api/tasks/${S.task}/export?fmt=json`));

  $("#clearBtn").addEventListener("click", async () => {
    if (!window.confirm(`确定清空「${activeTaskInfo()?.display_name || S.task}」的最近输入、结果与历史？\n模型与阈值不会被删除。`)) return;
    try {
      await API.clear(S.task);
      const pt = perTask();
      pt.lastResult = null;
      pt.lastInput = null;
      pt.history = [];
      renderViewer();
      renderResult();
      renderHistory();
      toast("已清空该任务的输入、结果与历史", "ok");
    } catch (e) {
      toast(`清空失败：${e.message}`, "error");
    }
  });

  $$("#viewToggle .seg-btn").forEach((b) => b.addEventListener("click", () => {
    S.view = b.dataset.view;
    $$("#viewToggle .seg-btn").forEach((x) => x.classList.toggle("is-active", x === b));
    renderViewer();
  }));

  $("#regionFilter").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    S.regionFilter = chip.dataset.filter;
    $$("#regionFilter .chip").forEach((c) => c.classList.toggle("is-active", c === chip));
    renderResult();
  });
}

function download(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
  toast("已开始下载", "ok", 1800);
}

async function reloadHistory() {
  if (!S.task) return;
  try {
    const h = await API.history(S.task, 40);
    perTask().history = h.items || [];
    renderHistory();
  } catch { /* 历史读取失败不阻塞主流程 */ }
}

/* ---------------- 状态轮询（模型建模进度） ---------------- */
function startStatusPolling() {
  setInterval(async () => {
    if (!S.task) return;
    try {
      const d = await API.tasks();
      const before = S.tasks.map((t) => t.state && t.state.state).join(",");
      S.tasks = d.tasks || [];
      const after = S.tasks.map((t) => t.state && t.state.state).join(",");
      if (before !== after) {
        refreshModelPill();
        const cur = activeTaskInfo();
        if (cur && cur.state && cur.state.state === "idle" && $$("#taskSwitch .task-btn").length !== S.tasks.length) {
          renderTaskSwitch();
          $$("#taskSwitch .task-btn").forEach((b) => b.classList.toggle("is-active", b.dataset.task === S.task));
        }
      }
    } catch { /* 网络抖动忽略 */ }
  }, 4000);
}

/* ---------------- 初始化 ---------------- */
/* 超宽屏下把「异常存档」挪进判定结果卡，利用结果卡下方的空白，避免右栏溢出 */
function layoutAnomalyCard() {
  const card = $("#anomalyCard");
  if (!card) return;
  const wide = window.matchMedia("(min-width: 1900px)").matches;
  const target = wide ? document.querySelector(".result-card") : document.querySelector(".col-right");
  if (!target) return;
  if (wide) {
    if (card.parentElement !== target) target.appendChild(card);
    card.classList.add("card--inner");
  } else {
    if (card.parentElement !== target) target.appendChild(card);
    card.classList.remove("card--inner");
  }
}

async function init() {
  bindSourceTabs();
  bindImage();
  bindCamera();
  bindVideo();
  bindControls();
  layoutAnomalyCard();
  window.addEventListener("resize", layoutAnomalyCard);
  try {
    await loadTasks();
  } catch (e) {
    toast(`初始化失败：${e.message}`, "error", 8000);
    $("#taskSkeleton").textContent = "任务加载失败，请检查后端服务";
  }
  startStatusPolling();
}

document.addEventListener("DOMContentLoaded", init);
