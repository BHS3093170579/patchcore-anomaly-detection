"""推理入口：加载已建模的任务，对单张图片或目录批量推理。

用法:
    python -m det.infer --task example --image data/example/test/defect/000.png
    python -m det.infer --task example --dir data/example/test --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from det.config import available_tasks
from det.engine import Detector

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="无监督缺陷检测推理")
    ap.add_argument("--task", required=True, help=f"任务名，可选: {available_tasks()}")
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--dir", help="目录批量推理")
    ap.add_argument("--limit", type=int, default=0, help="批量时最多处理张数(0=全部)")
    ap.add_argument("--no-overlay", action="store_true", help="不生成可视化叠加图")
    ap.add_argument("--save-anomaly", action="store_true", help="保存异常样本")
    ap.add_argument("--out", help="结果 JSON 输出路径")
    args = ap.parse_args(argv)

    task = args.task
    if task not in available_tasks():
        print(f"未知任务: {task}，可用: {available_tasks()}", file=sys.stderr)
        return 2
    det = Detector(task)
    try:
        det.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[{task}] 模型加载失败: {exc}", file=sys.stderr)
        return 1

    results: list[dict] = []
    if args.image:
        results.append(det.predict_path(
            args.image,
            with_overlay=not args.no_overlay,
            save_anomaly=args.save_anomaly,
        ))
    elif args.dir:
        d = Path(args.dir)
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if args.limit > 0:
            files = files[: args.limit]
        if not files:
            print(f"目录为空: {d}", file=sys.stderr)
            return 1
        for p in files:
            r = det.predict_path(p, with_overlay=False, save_anomaly=args.save_anomaly)
            r["image"] = str(p)
            results.append(r)
    else:
        print("需要 --image 或 --dir", file=sys.stderr)
        return 2

    payload = results[0] if len(results) == 1 else results
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        abnormals = [r for r in results if r.get("is_anomaly")]
        print(f"共 {len(results)} 张，异常 {len(abnormals)} 张，结果写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
