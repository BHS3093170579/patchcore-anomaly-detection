"""FastAPI 应用入口。

任务隔离原则：每个检测任务拥有独立的模型、阈值、数据路径与运行时状态；
本文件只负责进程级装配（静态资源、路由、日志），不含任何具体任务的名字。
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.runtime import RuntimeManager
from app.video import VideoManager
from det.config import PROJECT_ROOT, available_tasks, ensure_task_dirs, paths, task_summary
from det.patchcore import normalize_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("app")

WEB_DIR = PROJECT_ROOT / "web"
RUNS_DIR = PROJECT_ROOT / "runs"
START_TS = time.time()

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 800 * 1024 * 1024

app = FastAPI(title="无监督缺陷检测系统", version="1.0.0")
runtime = RuntimeManager()
videos = VideoManager(get_state=runtime.get, max_workers=1)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _require_task(task: str) -> None:
    if task not in available_tasks():
        raise HTTPException(status_code=404, detail=f"未知任务: {task}")


def _thumb(img: Image.Image, side: int = 320) -> str:
    """生成最近输入的小缩略图（JPEG base64），用于切换任务后恢复显示。"""
    im = img.copy()
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _open_image_upload(data: bytes, name: str) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"图片过大（上限 {MAX_IMAGE_BYTES // 1024 // 1024}MB）")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = normalize_image(img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片 {name}: {exc}") from exc
    return img


_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _decode_data_url(s: str) -> bytes:
    s = _DATA_URL.sub("", s.strip())
    try:
        return base64.b64decode(s, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"帧数据不是合法 base64: {exc}") from exc


def _infer(task: str, img: Image.Image, *, save_anomaly: bool,
           record: bool, source: str, overlay_quality: int = 82) -> dict[str, Any]:
    state = runtime.get(task)
    det = state.load_model()
    with state.lock:
        result = det.predict(img, with_overlay=True, save_anomaly=save_anomaly,
                             overlay_quality=overlay_quality)
    state.infer_count += 1
    state.last_latency_ms = float(result.get("latency_ms", 0.0))
    result["source"] = source
    if record:
        item = {
            "id": uuid.uuid4().hex[:10],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "score": result["score"],
            "threshold": result["threshold"],
            "is_anomaly": result["is_anomaly"],
            "defect_count": result["defect_count"],
            "latency_ms": result["latency_ms"],
        }
        state.push_history(item)
        result["record_id"] = item["id"]
        state.last_result = result
        state.last_input = {"name": source, "thumb": _thumb(img)}
    return result


# --------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    tasks = available_tasks()
    log.info("发现任务配置: %s", tasks)
    for t in tasks:
        ensure_task_dirs(t)
    log.info("运行时就绪，Web 目录: %s", WEB_DIR)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START_TS, 1),
        "tasks": available_tasks(),
        "version": app.version,
    }


@app.get("/api/tasks")
def api_tasks() -> dict:
    out = []
    for t in available_tasks():
        s = task_summary(t)
        st = runtime.get(t)
        det = st.detector
        s["has_model"] = bool(det and det.bank)
        s["state"] = st.snapshot_status()
        s["threshold_runtime"] = {
            "image_score": st.params.get("threshold_image"),
            "pixel_score": st.params.get("threshold_pixel"),
        }
        out.append(s)
    return {"tasks": out}


# --------------------------------------------------------------------------
# 任务状态 / 参数
# --------------------------------------------------------------------------
@app.get("/api/tasks/{task}/state")
def api_task_state(task: str) -> dict:
    _require_task(task)
    st = runtime.get(task)
    return {
        "task": task,
        "status": st.snapshot_status(),
        "params": dict(st.params),
        "last_result": st.last_result,
        "last_input": st.last_input,
        "counters": {"infer_count": st.infer_count, "error_count": st.error_count},
    }


@app.post("/api/tasks/{task}/clear")
def api_clear(task: str) -> dict:
    _require_task(task)
    st = runtime.get(task)
    st.clear()
    return {"ok": True, "message": "已清空该任务的输入与结果（模型与阈值不变）"}


@app.get("/api/tasks/{task}/history")
def api_history(task: str, limit: int = Query(50, ge=1, le=200)) -> dict:
    _require_task(task)
    st = runtime.get(task)
    with st.lock:
        items = list(st.history)[:limit]
    return {"task": task, "total": len(st.history), "items": items}


@app.get("/api/tasks/{task}/metrics")
def api_metrics(task: str) -> dict:
    """返回该任务最近一次离线评估的指标（供界面展示，不参与推理）。"""
    _require_task(task)
    p = paths(task)
    eval_path = Path(p["runs_dir"]) / f"eval_{task}.json"
    if not eval_path.exists():
        return {"task": task, "available": False}
    try:
        d = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"task": task, "available": False, "error": str(exc)}
    per_class = d.get("per_class") or {}
    return {
        "task": task,
        "available": True,
        "n_images": d.get("n_images"),
        "image_auroc": d.get("image_auroc"),
        "image_ap": d.get("image_ap"),
        "image_f1max": d.get("image_f1max"),
        "pixel_auroc": d.get("pixel_auroc"),
        "pixel_pro": d.get("pixel_pro"),
        "threshold": d.get("threshold"),
        "classes": [
            {"name": k, "detected": v.get("detected"), "total": v.get("n"),
             "auroc": v.get("auroc_vs_good")}
            for k, v in per_class.items()
        ],
    }


@app.get("/api/tasks/{task}/anomalies")
def api_anomalies(task: str, limit: int = Query(12, ge=1, le=60)) -> dict:
    """列出该任务已存档的异常图（最新在前）。"""
    _require_task(task)
    p = paths(task)
    folder = Path(p["anomalies_dir"])
    items: list[dict] = []
    if folder.exists():
        for img in sorted(folder.glob("*_image.png"), reverse=True):
            if len(items) >= limit:
                break
            stem = img.name[: -len("_image.png")]
            meta_path = folder / f"{stem}_meta.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
            items.append({
                "name": stem,
                "url": f"/media/{task}/anomalies/{img.name}",
                "overlay_url": (f"/media/{task}/anomalies/{stem}_overlay.jpg"
                                if (folder / f"{stem}_overlay.jpg").exists() else None),
                "score": meta.get("map_max"),
                "time": meta.get("saved_at"),
                "regions": len(meta.get("regions") or []),
            })
    return {"task": task, "total": len(list(folder.glob("*_image.png"))) if folder.exists() else 0,
            "items": items}


@app.post("/api/tasks/{task}/threshold")
async def api_set_threshold(task: str, request: Request) -> dict:
    _require_task(task)
    body = await request.json()
    st = runtime.get(task)
    det = st.load_model()
    img_t = body.get("image_score")
    pix_t = body.get("pixel_score")
    if img_t is None and pix_t is None:
        raise HTTPException(status_code=400, detail="至少提供 image_score 或 pixel_score")
    try:
        thr = det.set_threshold(
            image_score=None if img_t is None else float(img_t),
            pixel_score=None if pix_t is None else float(pix_t),
            persist=True,
            origin=body.get("calibrated_from") or None,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"阈值必须是数字: {exc}") from exc
    with st.lock:
        st.params["threshold_image"] = float(thr.get("image_score") or 0.0)
        st.params["threshold_pixel"] = float(thr.get("pixel_score") or 0.0)
    return {"ok": True, "threshold": thr,
            "note": "阈值已写入该任务的 models/<task>/threshold.json 与 configs/<task>.json；已生成的历史结果不会自动重算，可重新推理刷新。"}


@app.get("/api/tasks/{task}/model")
def api_model(task: str) -> dict:
    _require_task(task)
    st = runtime.get(task)
    p = paths(task)
    info: dict[str, Any] = {
        "task": task,
        "bank_path": str(p["model_dir"] / "memory_bank.npz"),
        "threshold_path": str(p["model_dir"] / "threshold.json"),
        "has_model": (p["model_dir"] / "memory_bank.npz").is_file(),
        "state": st.snapshot_status(),
    }
    det = st.detector
    if det and det.bank:
        info["meta"] = det.bank.meta
        info["bank_size"] = int(det.bank.size)
        info["threshold"] = det.threshold
    elif info["has_model"]:
        # 模型还没懒加载进内存时，直接从 npz 读元信息，避免为了显示参数而加载 263MB 权重
        try:
            import numpy as np

            with np.load(p["model_dir"] / "memory_bank.npz", allow_pickle=False) as z:
                info["meta"] = json.loads(str(z["meta"].item()))
                info["bank_size"] = int(z["bank"].shape[0])
        except (OSError, ValueError, KeyError):
            pass
    return info


@app.post("/api/tasks/{task}/model/load")
def api_model_load(task: str) -> dict:
    _require_task(task)
    st = runtime.get(task)
    t0 = time.perf_counter()
    try:
        det = st.load_model(force=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"模型加载失败: {exc}") from exc
    return {"ok": True, "task": task, "bank_size": int(det.bank.size if det.bank else 0),
            "load_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/api/tasks/{task}/model/build")
def api_model_build(task: str) -> dict:
    """后台重新建模（仅用 train/good）。返回 202，进度看 /api/tasks/{task}/model。"""
    _require_task(task)
    st = runtime.get(task)
    with st.lock:
        if st.status.get("state") == "building":
            raise HTTPException(status_code=409, detail="该任务正在建模中")
        st.status = {"state": "building", "message": "建模启动", "updated_at": time.time()}

    def _work() -> None:
        try:
            det = st.load_model()
            det.build(progress=None, verbose=False)
            st.load_model(force=True)
            st.set_status("idle", "建模完成")
        except Exception as exc:  # noqa: BLE001
            log.exception("建模失败: %s", exc)
            st.set_status("error", f"建模失败: {exc}")

    threading.Thread(target=_work, daemon=True).start()
    return {"ok": True, "task": task, "message": "已开始后台建模（只用 train/good）"}


# --------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------
@app.post("/api/infer/{task}")
async def api_infer(task: str, file: UploadFile = File(...),
                    save_anomaly: bool = Form(True)) -> dict:
    _require_task(task)
    data = await file.read()
    img = _open_image_upload(data, file.filename or "upload")
    return _infer(task, img, save_anomaly=save_anomaly, record=True,
                  source=f"图片: {file.filename or 'upload'}")


@app.post("/api/infer/{task}/frame")
async def api_infer_frame(task: str, request: Request) -> dict:
    """实时链路入口：接收 base64 帧，返回结果 + 叠加图，不写历史（由前端按需标记）。"""
    _require_task(task)
    body = await request.json()
    raw = body.get("image") or body.get("image_base64") or ""
    if not raw:
        raise HTTPException(status_code=400, detail="缺少 image 字段（dataURL 或 base64）")
    data = _decode_data_url(raw)
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="帧数据过大")
    img = _open_image_upload(data, "frame")
    rec = bool(body.get("record", False))
    overlay_q = max(40, min(int(body.get("overlay_quality", 70)), 95))
    result = _infer(task, img, save_anomaly=False, record=rec,
                    source="摄像头" if rec else "实时帧", overlay_quality=overlay_q)
    return result


@app.post("/api/infer/{task}/batch")
async def api_infer_batch(task: str, files: list[UploadFile] = File(...)) -> dict:
    _require_task(task)
    items = []
    for f in files:
        try:
            data = await f.read()
            img = _open_image_upload(data, f.filename or "upload")
            r = _infer(task, img, save_anomaly=False, record=False, source=f"批量: {f.filename}")
            items.append({
                "name": f.filename, "ok": True, "score": r["score"],
                "is_anomaly": r["is_anomaly"], "defect_count": r["defect_count"],
                "latency_ms": r["latency_ms"], "overlay": r.get("overlay"),
            })
        except Exception as exc:  # noqa: BLE001
            items.append({"name": f.filename, "ok": False, "error": str(exc)})
    return {"task": task, "count": len(items), "items": items}


# --------------------------------------------------------------------------
# 视频
# --------------------------------------------------------------------------
@app.post("/api/tasks/{task}/video")
async def api_video_upload(task: str, file: UploadFile = File(...),
                           target_fps: float = Form(3.0)) -> dict:
    _require_task(task)
    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="视频过大（上限 800MB）")
    p = paths(task)
    up_dir = p["runs_dir"] / "videos" / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.\-]+", "_", file.filename or "video.mp4")
    dst = up_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    dst.write_bytes(data)
    job = videos.submit(task, dst, target_fps=target_fps)
    return {"ok": True, "job": job.summary()}


@app.get("/api/video/{job_id}")
def api_video_get(job_id: str) -> dict:
    job = videos.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="视频任务不存在")
    return job.summary()


@app.get("/api/videos")
def api_video_list(task: str | None = None) -> dict:
    if task is not None:
        _require_task(task)
    return {"items": videos.list(task)}


@app.post("/api/video/{job_id}/cancel")
def api_video_cancel(job_id: str) -> dict:
    if not videos.cancel(job_id):
        raise HTTPException(status_code=404, detail="视频任务不存在")
    return {"ok": True}


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
@app.get("/api/tasks/{task}/export")
def api_export(task: str, fmt: str = Query("csv", pattern="^(csv|json)$")) -> Response:
    _require_task(task)
    st = runtime.get(task)
    with st.lock:
        items = list(st.history)
    if fmt == "json":
        payload = json.dumps({"task": task, "count": len(items), "items": items},
                             ensure_ascii=False, indent=2)
        return Response(payload, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{task}_history.json"'})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["时间", "来源", "异常分", "阈值", "是否异常", "区域数", "耗时ms"])
    for it in items:
        w.writerow([it.get("time"), it.get("source"), it.get("score"), it.get("threshold"),
                    "是" if it.get("is_anomaly") else "否", it.get("defect_count"),
                    it.get("latency_ms")])
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{task}_history.csv"'})


# --------------------------------------------------------------------------
# 静态资源
# --------------------------------------------------------------------------
if (PROJECT_ROOT / "runs").is_dir():
    app.mount("/media", StaticFiles(directory=str(RUNS_DIR)), name="media")

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index() -> Response:
        """注入静态资源版本号，避免浏览器拿旧 CSS/JS 导致界面与接口不一致。"""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        stamp = 0
        for name in ("style.css", "app.js"):
            f = WEB_DIR / name
            if f.exists():
                stamp = max(stamp, int(f.stat().st_mtime))
        html = html.replace("__V__", str(stamp))
        return Response(html, media_type="text/html; charset=utf-8",
                        headers={"Cache-Control": "no-cache"})


@app.exception_handler(Exception)
async def _unhandled(request, exc: Exception):
    log.exception("未处理异常: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"内部错误: {exc}"})
