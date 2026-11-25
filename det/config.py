"""任务配置加载。

每个检测任务一份独立配置（``configs/<task>.json``），文件内只出现本任务自己的
数据路径、模型路径与阈值，不引用其他任务。加载器只按 task 名读取对应文件，
因此两个任务在配置层面天然隔离。
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}


def available_tasks() -> list[str]:
    """扫描 configs 目录，返回可用任务名（按文件名）。"""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def config_path(task: str) -> Path:
    path = CONFIG_DIR / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"任务配置不存在: {path}")
    return path


def load_task_config(task: str) -> dict[str, Any]:
    """读取任务配置（深拷贝，调用方可安全修改）。"""
    with _lock:
        if task not in _cache:
            path = config_path(task)
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("task") != task:
                raise ValueError(f"配置内 task 字段({cfg.get('task')}) 与文件名({task}) 不一致")
            _cache[task] = cfg
        return copy.deepcopy(_cache[task])


def save_task_config(task: str, cfg: dict[str, Any]) -> None:
    """写回任务配置（阈值在线调节后持久化用）。

    Windows 下 os.replace/Path.replace 偶尔会因目标文件被索引服务/杀软短暂占用
    而报 WinError 5，这里做有限重试，最后退化为直接覆盖写。
    """
    import time as _time

    path = config_path(task)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            tmp.replace(path)
            last_exc = None
            break
        except OSError as exc:  # noqa: PERF203
            last_exc = exc
            _time.sleep(0.15 * (attempt + 1))
    if last_exc is not None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    with _lock:
        _cache[task] = copy.deepcopy(cfg)


def resolve(raw: str | Path) -> Path:
    """把配置里的路径解析成绝对路径。

    绝对路径原样返回；相对路径按项目根目录展开，这样仓库可以放在任意位置运行，
    不需要把绝对路径写进 configs/*.json。
    """
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def paths(task: str) -> dict[str, Path]:
    cfg = load_task_config(task)
    return {
        "dataset_root": resolve(cfg["dataset_root"]),
        "train_good": resolve(cfg["train_good_dir"]),
        "test": resolve(cfg["test_dir"]),
        "gt": resolve(cfg["gt_dir"]),
        "model_dir": resolve(cfg["model_dir"]),
        "runs_dir": resolve(cfg["runs_dir"]),
        "anomalies_dir": resolve(cfg["runs_dir"]) / "anomalies",
    }


def ensure_task_dirs(task: str) -> None:
    p = paths(task)
    for key in ("model_dir", "runs_dir"):
        p[key].mkdir(parents=True, exist_ok=True)
    (p["runs_dir"] / "anomalies").mkdir(parents=True, exist_ok=True)
    (p["runs_dir"] / "history").mkdir(parents=True, exist_ok=True)
    (p["runs_dir"] / "logs").mkdir(parents=True, exist_ok=True)


def task_summary(task: str) -> dict[str, Any]:
    """返回前端任务切换区需要的最小信息。"""
    cfg = load_task_config(task)
    p = paths(task)
    model_file = p["model_dir"] / "memory_bank.npz"
    return {
        "task": cfg["task"],
        "display_name": cfg["display_name"],
        "accent": cfg.get("accent", "#2563eb"),
        "defect_classes": cfg.get("defect_classes", []),
        "model_ready": model_file.is_file(),
        "model_path": str(model_file),
        "input_size": cfg["model"]["input_size"],
        "threshold": cfg.get("threshold", {}),
        "threshold_presets": cfg.get("threshold_presets", {}),
    }
