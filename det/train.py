"""建模入口：只用 train/good 的正常样本建立特征库并标定阈值。

用法:
    python -m det.train --task example
    python -m det.train --task example
"""
from __future__ import annotations

import argparse
import json
import sys

from det.config import available_tasks, ensure_task_dirs
from det.engine import Detector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="无监督建模（仅使用正常样本）")
    ap.add_argument("--task", required=True, help=f"任务名，可选: {available_tasks()}")
    ap.add_argument("--quiet", action="store_true", help="只输出最终 JSON")
    args = ap.parse_args(argv)

    task = args.task
    if task not in available_tasks():
        print(f"未知任务: {task}，可用: {available_tasks()}", file=sys.stderr)
        return 2
    ensure_task_dirs(task)

    det = Detector(task)
    total = 0

    def progress(done: int, total_n: int) -> None:
        nonlocal total
        total = total_n

    try:
        info = det.build(progress=progress, verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001
        print(f"[{task}] 建模失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
