"""任务运行时：每个检测任务一份独立的模型、阈值与状态。

隔离要求：任何任务的状态（最近一次结果、参数、阈值、模型实例、历史）
都只存在于自己的 TaskState 里，跨任务访问只能通过 RuntimeManager.get(task)，
不存在共享的可变全局状态。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from det.config import load_task_config


@dataclass
class TaskState:
    task: str
    cfg: dict[str, Any] = field(default_factory=dict)

    lock: threading.RLock = field(default_factory=threading.RLock)
    _detector: Any = None

    params: dict[str, Any] = field(default_factory=dict)
    last_result: dict[str, Any] | None = None
    last_input: dict[str, Any] | None = None     # 最近输入（名称 + 小缩略图），切回任务时恢复显示
    status: dict[str, Any] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=200))

    infer_count: int = 0
    error_count: int = 0
    last_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.cfg:
            self.cfg = load_task_config(self.task)
        self.status = {
            "state": "idle",
            "message": "就绪",
            "updated_at": time.time(),
        }
        self.params = {
            "threshold_image": float(self.cfg["threshold"].get("image_score") or 0.0),
            "threshold_pixel": float(self.cfg["threshold"].get("pixel_score") or 0.0),
            "score_mode": "max",
            "save_anomaly": True,
        }

    # ---- 状态 -------------------------------------------------------------
    def set_status(self, state: str, message: str) -> None:
        with self.lock:
            self.status = {"state": state, "message": message, "updated_at": time.time()}

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status)

    # ---- 模型 -------------------------------------------------------------
    @property
    def detector(self):
        with self.lock:
            return self._detector

    def load_model(self, force: bool = False):
        """懒加载模型；失败时抛异常并把状态标记为 error。"""
        from det.engine import Detector  # 延迟导入，避免启动即加载 torch

        with self.lock:
            if self._detector is not None and not force:
                return self._detector
            self.status = {
                "state": "loading",
                "message": "正在加载模型",
                "updated_at": time.time(),
            }
        try:
            det = Detector(self.task)
            det.load()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = {
                    "state": "error",
                    "message": f"模型加载失败: {exc}",
                    "updated_at": time.time(),
                }
            raise
        with self.lock:
            self._detector = det
            self.status = {
                "state": "idle",
                "message": "就绪",
                "updated_at": time.time(),
            }
        return self._detector

    # ---- 历史 -------------------------------------------------------------
    def push_history(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.history.appendleft(item)

    def clear(self) -> None:
        """清空本任务的最近结果与历史（不影响模型与阈值）。"""
        with self.lock:
            self.last_result = None
            self.last_input = None
            self.history.clear()
            self.status = {
                "state": "idle",
                "message": "已清空",
                "updated_at": time.time(),
            }


class RuntimeManager:
    def __init__(self) -> None:
        self._states: dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def get(self, task: str) -> TaskState:
        with self._lock:
            if task not in self._states:
                self._states[task] = TaskState(task=task)
            return self._states[task]

    def known(self) -> list[str]:
        with self._lock:
            return sorted(self._states)
