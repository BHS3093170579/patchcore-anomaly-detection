"""离线评估：在 test 集上计算图像级与像素级指标。

指标口径：
  - image AUROC / AP：以“每张图的图像异常分”为分数，label 为是否缺陷图
  - pixel AUROC：把每张图异常图双线性放大到 GT 尺寸后的逐像素比较
  - AUPRO：按 GT 连通区域计算重叠率随假正率变化的积分（FPR 上限 0.3）
  - F1max：在图像级最优阈值下的 F1

用法:
    python -m det.eval --task example
    python -m det.eval --task example --out runs/example/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from det.config import available_tasks, paths
from det.engine import Detector

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _load_mask(p: Path, size: tuple[int, int]) -> np.ndarray:
    m = Image.open(p).convert("L").resize(size, Image.NEAREST)
    return (np.asarray(m) > 127)


def compute_pro(region_scores: list[np.ndarray], fp_scores: np.ndarray,
                num_th: int = 200, fpr_limit: float = 0.3) -> float:
    """基于逐区域的 AUPRO。region_scores 为每个 GT 连通区域内像素的异常分。"""
    if not region_scores or fp_scores.size == 0:
        return float("nan")
    # 阈值网格：取全局分数分位数，保证覆盖高低两端
    all_s = np.concatenate([fp_scores] + region_scores)
    if all_s.size > 400_000:
        all_s = np.random.default_rng(0).choice(all_s, 400_000, replace=False)
    ths = np.quantile(all_s, np.linspace(0.01, 1.0, num_th))

    # 假正率曲线（用非区域像素）
    fp_sorted = np.sort(fp_scores)
    n_fp = fp_sorted.size
    fpr = 1.0 - np.searchsorted(fp_sorted, ths, side="left") / n_fp

    curves = []
    for s in region_scores:
        s_sorted = np.sort(s)
        overlap = 1.0 - np.searchsorted(s_sorted, ths, side="left") / s_sorted.size
        curves.append(overlap)
    overlap_mean = np.mean(np.vstack(curves), axis=0)

    order = np.argsort(fpr)
    fpr_s, ov_s = fpr[order], overlap_mean[order]
    keep = fpr_s <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    fpr_s, ov_s = fpr_s[keep], ov_s[keep]
    # 在统一 FPR 网格上插值，避免采样不均导致偏置
    grid = np.linspace(0.0, fpr_limit, 200)
    ov_i = np.interp(grid, fpr_s, ov_s)
    # 归一化到 [0,1]（PRO 常用口径：面积 / (fpr_limit * 1.0)）
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trap(ov_i, grid) / fpr_limit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线评估（图像级 + 像素级）")
    ap.add_argument("--task", required=True, help=f"任务名，可选: {available_tasks()}")
    ap.add_argument("--out", help="输出 JSON 路径（默认 runs/<task>/eval_<task>.json）")
    ap.add_argument("--no-pro", action="store_true", help="跳过 AUPRO（较慢）")
    args = ap.parse_args(argv)

    task = args.task
    if task not in available_tasks():
        print(f"未知任务: {task}，可用: {available_tasks()}", file=sys.stderr)
        return 2
    p = paths(task)
    test_dir, gt_dir = p["test"], p["gt"]
    if not test_dir.is_dir():
        print(f"测试目录不存在: {test_dir}", file=sys.stderr)
        return 1

    det = Detector(task)
    try:
        det.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[{task}] 模型加载失败: {exc}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for cls_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        cls = cls_dir.name
        is_good = cls == "good"
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        for f in files:
            img = Image.open(f)
            img.load()
            amap, score, size = det.score_map(img)
            mask = None
            if not is_good:
                rel = f.relative_to(test_dir)
                mp = gt_dir / rel.parent.name / rel.name.replace(".png", "_mask.png")
                if mp.is_file():
                    mask = _load_mask(mp, img.size)
            rows.append({
                "image": str(f.relative_to(p["dataset_root"])),
                "class": cls,
                "label": 0 if is_good else 1,
                "score": float(score),
                "map": amap,
                "mask": mask,
                "size": size,
            })

    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    img_auroc = float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan")
    img_ap = float(average_precision_score(labels, scores)) if labels.min() != labels.max() else float("nan")
    fpr, tpr, ths = roc_curve(labels, scores)
    prec = tpr * labels.mean() / np.maximum(tpr * labels.mean() + fpr * (1 - labels.mean()), 1e-9)
    f1 = 2 * prec * tpr / np.maximum(prec + tpr, 1e-9)
    f1max = float(np.nanmax(f1))

    # ---- 像素级 ----
    pred_px: list[np.ndarray] = []
    gt_px: list[np.ndarray] = []
    region_scores: list[np.ndarray] = []
    fp_scores: list[np.ndarray] = []
    for r in rows:
        if r["mask"] is None:
            continue
        h, w = r["mask"].shape
        up = cv2.resize(r["map"].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        pred_px.append(up.ravel())
        gt_px.append(r["mask"].ravel().astype(np.uint8))
        n, lab = cv2.connectedComponents(r["mask"].astype(np.uint8), connectivity=8)
        for i in range(1, n):
            reg = lab == i
            if reg.sum() < 4:
                continue
            region_scores.append(up[reg])
        fp_scores.append(up[~r["mask"]])

    pix_auroc = float("nan")
    if pred_px:
        pred_all = np.concatenate(pred_px)
        gt_all = np.concatenate(gt_px)
        pix_auroc = float(roc_auc_score(gt_all, pred_all)) if gt_all.min() != gt_all.max() else float("nan")
    pro = float("nan")
    if not args.no_pro and region_scores:
        fp_all = np.concatenate(fp_scores)
        if fp_all.size > 2_000_000:
            fp_all = np.random.default_rng(0).choice(fp_all, 2_000_000, replace=False)
        pro = compute_pro(region_scores, fp_all)

    # ---- 按缺陷类型 ----
    per_class: dict[str, dict] = {}
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r["score"])
    good_scores = np.array(by_cls.get("good", []))
    for cls, ss in sorted(by_cls.items()):
        if cls == "good":
            per_class[cls] = {"n": len(ss), "score_mean": round(float(np.mean(ss)), 4),
                              "score_max": round(float(np.max(ss)), 4)}
            continue
        ss = np.array(ss)
        auc = float("nan")
        if good_scores.size and "good" in by_cls:
            y = np.r_[np.zeros(good_scores.size), np.ones(ss.size)]
            s = np.r_[good_scores, ss]
            auc = float(roc_auc_score(y, s))
        per_class[cls] = {"n": int(ss.size), "score_mean": round(float(ss.mean()), 4),
                          "detected": int((ss > det.threshold_values()[0]).sum()),
                          "auroc_vs_good": round(auc, 4)}

    report = {
        "task": task,
        "model": str(det.bank_path),
        "bank_size": det.bank.size if det.bank else 0,
        "n_images": len(rows),
        "threshold": det.threshold_values(),
        "image_auroc": round(img_auroc, 4),
        "image_ap": round(img_ap, 4),
        "image_f1max": round(f1max, 4),
        "pixel_auroc": round(pix_auroc, 4),
        "pixel_pro": round(pro, 4) if pro == pro else None,
        "per_class": per_class,
        "samples": [{k: r[k] for k in ("image", "class", "label", "score")} for r in rows],
    }

    out = Path(args.out) if args.out else (p["runs_dir"] / f"eval_{task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        f"# {task} 评估结果", "",
        f"- 图像级 AUROC: **{img_auroc:.4f}**   AP: {img_ap:.4f}   F1max: {f1max:.4f}",
        f"- 像素级 AUROC: {pix_auroc:.4f}   AUPRO: {pro:.4f}" if pro == pro else f"- 像素级 AUROC: {pix_auroc:.4f}",
        f"- 阈值 image={det.threshold_values()[0]:.4f} pixel={det.threshold_values()[1]:.4f}",
        f"- 样本数: {len(rows)}", "",
        "| 缺陷类型 | 张数 | 平均分 | 超阈值检出 | 对 good 的 AUROC |",
        "|---|---|---|---|---|",
    ]
    for cls, v in per_class.items():
        if "detected" in v:
            md.append(f"| {cls} | {v['n']} | {v['score_mean']} | {v['detected']} | {v['auroc_vs_good']} |")
        else:
            md.append(f"| {cls}(正常) | {v['n']} | {v['score_mean']} | - | - |")
    (out.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\n结果已写入 {out} 和 {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
