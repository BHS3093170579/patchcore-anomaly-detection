"""视频批处理任务。

每个视频任务只服务一个检测任务（task），使用该任务自己的 Detector 实例，
不与其他任务的模型或阈值发生任何交叉。
"""
from __future__ import annotations

import base64
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
from PIL import Image

from det.config import paths


@dataclass
class VideoJob:
    id: str
    task: str
    src: Path
    out_dir: Path
    status: str = "queued"          # queued | running | done | error | canceled
    message: str = ""
    src_fps: float = 0.0
    n_frames: int = 0
    duration_s: float = 0.0
    target_fps: float = 3.0
    processed: int = 0
    total: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)
    anomaly_frames: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status,
            "message": self.message,
            "file": self.src.name,
            "src_fps": round(self.src_fps, 2),
            "duration_s": round(self.duration_s, 2),
            "target_fps": self.target_fps,
            "frames_total": self.total,
            "frames_done": self.processed,
            "anomaly_frames": self.anomaly_frames,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0.0,
            "frames": self.frames,
        }


class VideoManager:
    def __init__(self, get_state: Callable[[str], Any], max_workers: int = 1) -> None:
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()
        self._get_state = get_state
        self._sem = threading.Semaphore(max_workers)

    # ---- 查询 -------------------------------------------------------------
    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, task: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if task is None or j.task == task]
        return [j.summary() for j in sorted(jobs, key=lambda x: x.started_at, reverse=True)]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job._cancel.set()
        return True

    # ---- 提交 -------------------------------------------------------------
    def submit(self, task: str, src: Path, target_fps: float = 3.0) -> VideoJob:
        job_id = uuid.uuid4().hex[:12]
        out_dir = paths(task)["runs_dir"] / "videos" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        job = VideoJob(id=job_id, task=task, src=src, out_dir=out_dir,
                       target_fps=max(0.2, min(float(target_fps), 30.0)))
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    # ---- 执行 -------------------------------------------------------------
    def _run(self, job: VideoJob) -> None:
        with self._sem:
            job.started_at = time.time()
            job.status = "running"
            job.message = "解析视频"
            cap = cv2.VideoCapture(str(job.src))
            if not cap.isOpened():
                job.status = "error"
                job.message = "无法打开视频文件"
                job.finished_at = time.time()
                return
            try:
                job.src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                job.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                job.duration_s = job.n_frames / job.src_fps if job.src_fps > 0 else 0.0
                stride = max(1, int(round(job.src_fps / job.target_fps)))
                job.total = (job.n_frames + stride - 1) // stride if job.n_frames else 0

                state = self._get_state(job.task)
                det = state.load_model()
                state.set_status("running", f"视频推理: {job.src.name}")

                idx = 0
                while True:
                    if job._cancel.is_set():
                        job.status = "canceled"
                        job.message = "已取消"
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % stride != 0:
                        idx += 1
                        continue
                    t_frame = idx / job.src_fps if job.src_fps > 0 else 0.0
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    with state.lock:
                        r = det.predict(img, with_overlay=True, save_anomaly=False)
                    overlay_b64 = r.pop("overlay", None)
                    overlay_name = None
                    if overlay_b64:
                        overlay_name = f"f{idx:06d}.jpg"
                        (job.out_dir / overlay_name).write_bytes(
                            base64.b64decode(overlay_b64.split(",", 1)[-1]))
                    item = {
                        "frame_index": idx,
                        "t_s": round(t_frame, 3),
                        "score": r["score"],
                        "is_anomaly": r["is_anomaly"],
                        "defect_count": r["defect_count"],
                        "regions": r["regions"][:10],
                        "latency_ms": r["latency_ms"],
                        "overlay": overlay_name,
                    }
                    job.frames.append(item)
                    job.latencies_ms.append(float(r["latency_ms"]))
                    if r["is_anomaly"]:
                        job.anomaly_frames += 1
                    job.processed += 1
                    job.message = f"处理中 {job.processed}/{job.total or '?'}"
                    idx += 1
                if job.status == "running":
                    job.status = "done"
                    job.message = "完成"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.message = f"处理失败: {exc}"
            finally:
                cap.release()
                job.finished_at = time.time()
                try:
                    self._get_state(job.task).set_status("idle", "就绪")
                except Exception:  # noqa: BLE001
                    pass
