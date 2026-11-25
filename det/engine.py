"""任务无关的检测器装配层。

Detector 只认一个 task 名字，通过该任务自己的配置读取数据/模型/阈值路径，
不接触其他任务的任何文件。训练（建模）与推理共用同一套特征与打分实现，
保证“训练时怎么算、推理时怎么算”一致。
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from det.config import ensure_task_dirs, load_task_config, paths, save_task_config
from det.patchcore import (
    FeatureExtractor,
    MemoryBank,
    build_memory_bank,
    gaussian_smooth,
    make_projection,
    nearest_distances,
    normalize_image,
    pick_device,
    preprocess,
)


class ModelNotReady(RuntimeError):
    pass


MAX_ANOMALY_KEEP = 200
# 叠加图合成分辨率上限：前端检测画面最宽 760px，超过这个尺寸的合成本就是浪费
OVERLAY_MAX_SIDE = 768
# 连通域峰值至少要高出像素阈值这么多，才算一个缺陷区域，否则视为纹理噪声
PEAK_MIN_RATIO = 1.08


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Detector:
    def __init__(self, task: str) -> None:
        self.task = task
        self.cfg = load_task_config(task)
        self.paths = paths(task)
        self.model_cfg = self.cfg["model"]
        self.device = pick_device()
        self.extractor: FeatureExtractor | None = None
        self.bank: MemoryBank | None = None
        self._proj_t: torch.Tensor | None = None
        self.threshold: dict[str, Any] = dict(self.cfg.get("threshold", {}))
        self.model_dir = self.paths["model_dir"]
        self.bank_path = self.model_dir / "memory_bank.npz"
        self.threshold_path = self.model_dir / "threshold.json"

    # ---- 模型 -------------------------------------------------------------
    def load(self) -> "Detector":
        if not self.bank_path.is_file():
            raise ModelNotReady(
                f"任务 {self.task} 尚未建模，缺少文件: {self.bank_path}"
            )
        self.bank = MemoryBank.load(self.bank_path)
        self._proj_t = None
        if self.threshold_path.is_file():
            with self.threshold_path.open("r", encoding="utf-8") as f:
                disk = json.load(f)
            self.threshold.update(disk)
        size = int(self.bank.meta.get("input_size", self.model_cfg["input_size"]))
        self.model_cfg = dict(self.model_cfg)
        self.model_cfg["input_size"] = size
        self.extractor = FeatureExtractor(
            layers=tuple(self.bank.meta.get("layers", self.model_cfg["layers"])),
            weights=self.bank.meta.get("weights", self.model_cfg["weights"]),
            device=self.device,
        )
        return self

    # ---- 特征与打分 -------------------------------------------------------
    def embed_image(self, img: Image.Image) -> tuple[np.ndarray, int]:
        if self.extractor is None or self.bank is None:
            raise ModelNotReady("模型尚未加载")
        size = int(self.model_cfg["input_size"])
        x = preprocess(img, size, self.device)
        feats = self.extractor.embed(x)  # (1,C,h,w)
        c = feats.shape[1]
        flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
        # 投影在 GPU 上做：省掉 4096×1536 的显存→内存搬运，再只把降维后的结果取回
        if self.bank.projection is not None and self.device.type == "cuda":
            if self._proj_t is None or self._proj_t.device != feats.device:
                self._proj_t = torch.from_numpy(self.bank.projection).to(self.device)
            flat = flat @ self._proj_t
            return flat.contiguous().cpu().numpy(), size
        flat_np = flat.cpu().numpy()
        return flat_np @ self.bank.projection, size

    def score_map(self, img: Image.Image) -> tuple[np.ndarray, float, int]:
        flat, size = self.embed_image(img)
        dist = nearest_distances(flat, self.bank.bank, device=self.device)
        side = int(round(np.sqrt(dist.shape[0])))
        amap = dist.reshape(side, side)
        sigma = float(self.model_cfg.get("smoothing_sigma", 4.0))
        amap = gaussian_smooth(amap, sigma)
        if amap.shape[0] > 6:
            # patch 网格外圈对应原图外框，特征感受野被截断，稳定出现伪影：按到边界的距离衰减
            gh, gw = amap.shape
            gy, gx = np.indices(amap.shape)
            dist_edge = np.minimum(np.minimum(gx, gy), np.minimum(gw - 1 - gx, gh - 1 - gy))
            amap = amap * np.clip(dist_edge / 2.5, 0.25, 1.0).astype(np.float32)
        topk = int(self.model_cfg.get("score_topk", 100))
        topk = max(1, min(topk, amap.size))
        image_score = float(np.mean(np.sort(amap.ravel())[-topk:]))
        return amap, image_score, size

    # ---- 阈值 -------------------------------------------------------------
    def threshold_values(self) -> tuple[float, float]:
        t_img = float(self.threshold.get("image_score") or 0.0)
        t_pix = float(self.threshold.get("pixel_score") or 0.0)
        return t_img, t_pix

    def set_threshold(self, image_score: float | None = None, pixel_score: float | None = None,
                      persist: bool = True, origin: str | None = None) -> dict[str, Any]:
        """改阈值。origin 用于说明来源：默认 manual（界面手动调），
        自动化脚本恢复原值时传回原标定来源，避免把元数据刷成 manual。"""
        if image_score is not None:
            self.threshold["image_score"] = float(image_score)
        if pixel_score is not None:
            self.threshold["pixel_score"] = float(pixel_score)
        if origin is not None:
            self.threshold["calibrated_from"] = origin
        else:
            self.threshold["calibrated_from"] = "manual"
        self.threshold["calibrated_at"] = _now()
        if persist:
            self.threshold_path.parent.mkdir(parents=True, exist_ok=True)
            with self.threshold_path.open("w", encoding="utf-8") as f:
                json.dump(self.threshold, f, ensure_ascii=False, indent=2)
            cfg = load_task_config(self.task)
            cfg["threshold"] = dict(self.threshold)
            save_task_config(self.task, cfg)
        return dict(self.threshold)

    # ---- 推理 -------------------------------------------------------------
    def predict(self, img: Image.Image, with_overlay: bool = True,
                save_anomaly: bool = False, overlay_quality: int = 82) -> dict[str, Any]:
        if self.bank is None:
            self.load()
        assert self.bank is not None and self.extractor is not None
        img = normalize_image(img)
        t0 = time.perf_counter()
        orig_w, orig_h = img.size
        amap, image_score, size = self.score_map(img)
        t_img, t_pix = self.threshold_values()
        is_anomaly = bool(t_img <= 0 or image_score > t_img)
        thr = t_pix if t_pix > 0 else float(amap.max()) * 0.5
        regions = self._extract_regions(amap, size, orig_w, orig_h, thr)
        latency = (time.perf_counter() - t0) * 1000.0
        result: dict[str, Any] = {
            "task": self.task,
            "score": round(image_score, 4),
            "threshold": round(t_img, 4),
            "pixel_threshold": round(thr, 4),
            "is_anomaly": is_anomaly,
            "latency_ms": round(latency, 1),
            "image_size": [orig_w, orig_h],
            "input_size": size,
            "regions": regions,
            "defect_count": len(regions),
            "timestamp": _now(),
        }
        if with_overlay:
            overlay = self._render_overlay(img, amap, regions, thr, is_anomaly)
            result["overlay"] = _to_b64_jpeg(overlay, quality=int(overlay_quality))
        if save_anomaly and (is_anomaly or regions):
            path = self._save_anomaly(img, amap, regions,
                                      overlay if with_overlay else None)
            result["saved_to"] = str(path)
        return result

    def predict_path(self, image_path: str | Path, **kwargs) -> dict[str, Any]:
        with Image.open(image_path) as im:
            im.load()
            return self.predict(im, **kwargs)

    # ---- 区域 -------------------------------------------------------------
    def _extract_regions(self, amap: np.ndarray, size: int, orig_w: int, orig_h: int,
                         thr: float) -> list[dict[str, Any]]:
        mask = (amap >= thr).astype(np.uint8)
        if mask.sum() == 0:
            # 无超阈值像素时，退回取最热点作为候选，供界面定位
            peak = float(amap.max())
            if peak <= 0:
                return []
            y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
            r = max(1, amap.shape[0] // 32)
            mask[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1] = 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: list[dict[str, Any]] = []
        edge_regions: list[dict[str, Any]] = []
        sx, sy = orig_w / amap.shape[1], orig_h / amap.shape[0]
        gh, gw = amap.shape[0], amap.shape[1]
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 2:
                continue
            component = labels == i
            peak = float(amap[component].max())
            mean = float(amap[component].mean())
            item = {
                "bbox": [int(round(x * sx)), int(round(y * sy)),
                         int(round((x + w) * sx)), int(round((y + h) * sy))],
                "area_px": int(round(area * sx * sy)),
                "area_ratio": round(float(area) / amap.size, 5),
                "peak": round(peak, 4),
                "mean": round(mean, 4),
                "class": "anomaly",
            }
            # 贴 amap 边界（图像外框）的连通域几乎都是 resize padding 伪影
            margin = 3
            if (x <= margin or y <= margin
                    or x + w >= gw - margin or y + h >= gh - margin):
                edge_regions.append(item)
            else:
                regions.append(item)
        if not regions and edge_regions:
            regions = [max(edge_regions, key=lambda r: r["peak"])]
            regions[0]["at_border"] = True
        # 阈值附近的连通域面积小、峰值只比阈值高一点点，属于纹理噪声，
        # 会让"区域数"在 1~5 之间跳动；要求峰值至少高出阈值 8% 才算一个缺陷区域。
        if thr > 0:
            strong = [r for r in regions if r["peak"] >= thr * PEAK_MIN_RATIO]
            if strong:
                regions = strong
        regions.sort(key=lambda r: r["peak"], reverse=True)
        return regions[:20]

    # ---- 可视化 -----------------------------------------------------------
    def _render_overlay(self, img: Image.Image, amap: np.ndarray,
                        regions: list[dict[str, Any]], thr: float | None = None,
                        is_anomaly: bool = True, max_side: int = OVERLAY_MAX_SIDE) -> np.ndarray:
        """只在超阈值区域叠加热力，正常区域保留原图；图像级判正常时直接返回原图。

        叠加图按 max_side 缩放后再合成：前端显示区最宽 760px，1024 全尺寸合成纯属浪费，
        缩放 + 稀疏混合后这一段的耗时从 ~86ms 降到 ~12ms。
        """
        base_full = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
        if not is_anomaly:
            return base_full
        h, w = base_full.shape[:2]
        scale = min(1.0, float(max_side) / max(h, w))
        if scale < 1.0:
            ow, oh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            base = cv2.resize(base_full, (ow, oh), interpolation=cv2.INTER_LINEAR)
        else:
            ow, oh = w, h
            base = base_full
        heat = cv2.resize(amap.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
        t = float(thr) if thr is not None else float(heat.max()) * 0.5
        span = max(float(heat.max()) - t, 1e-6)
        norm = np.clip((heat - t) / span, 0.0, 1.0)
        alpha = (norm ** 0.75) * 0.82
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        # 稀疏混合：只处理真正被染色的像素，正常区域保持原样
        mask = alpha > 0.004
        if mask.any():
            a = alpha[mask][:, None]
            b = base[mask].astype(np.float32)
            c = cmap[mask].astype(np.float32)
            base[mask] = (b * (1.0 - a) + c * a).astype(np.uint8)
        box_color = (255, 60, 60)
        thick = 2 if scale > 0.75 else 1
        font = 0.45 if scale > 0.75 else 0.35
        for r in regions:
            x1, y1, x2, y2 = [int(round(v * scale)) for v in r["bbox"]]
            cv2.rectangle(base, (x1, y1), (x2, y2), box_color, thick)
            cv2.putText(base, f'{r["peak"]:.2f}', (x1 + 3, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 0), 1, cv2.LINE_AA)
        return base

    def _save_anomaly(self, img: Image.Image, amap: np.ndarray,
                      regions: list[dict[str, Any]],
                      overlay: np.ndarray | None = None) -> Path:
        ensure_task_dirs(self.task)
        out_dir = self.paths["runs_dir"] / "anomalies"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 长期运行会持续堆积，只保留最近 MAX_ANOMALY_KEEP 组，避免占满磁盘
        stale = sorted(out_dir.glob("*_image.png"))
        if len(stale) >= MAX_ANOMALY_KEEP:
            for f in stale[: len(stale) - MAX_ANOMALY_KEEP + 1]:
                f.unlink(missing_ok=True)
                f.with_name(f.name.replace("_image.png", "_meta.json")).unlink(missing_ok=True)
                f.with_name(f.name.replace("_image.png", "_overlay.jpg")).unlink(missing_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = out_dir / f"{stamp}_image.png"
        img.save(img_path)
        if overlay is not None:
            cv2.imwrite(str(out_dir / f"{stamp}_overlay.jpg"), overlay,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        meta = {
            "task": self.task,
            "regions": regions,
            "image_size": list(img.size),
            "map_max": float(amap.max()),
            "saved_at": _now(),
        }
        (out_dir / f"{stamp}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return img_path

    # ---- 建模 -------------------------------------------------------------
    def build(self, progress=None, verbose: bool = True) -> dict[str, Any]:
        """用 train/good 建特征库并标定阈值。只用正常样本。"""
        train_dir = self.paths["train_good"]
        if not train_dir.is_dir():
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        files = sorted(
            p for p in train_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not files:
            raise FileNotFoundError(f"训练目录为空: {train_dir}")
        seed = int(self.model_cfg.get("seed", 0))
        val_ratio = float(self.cfg.get("train", {}).get("val_ratio", 0.1))
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        n_val = max(1, int(round(len(files) * val_ratio))) if len(files) > 4 else 0
        val_idx = set(int(i) for i in order[:n_val])
        fit_files = [f for i, f in enumerate(files) if i not in val_idx]
        val_files = [files[i] for i in sorted(val_idx)]

        if verbose:
            print(f"[{self.task}] 正常样本 {len(files)} 张：建模 {len(fit_files)}，阈值标定 {len(val_files)}")

        size = int(self.model_cfg["input_size"])
        self.extractor = FeatureExtractor(
            layers=tuple(self.model_cfg["layers"]),
            weights=self.model_cfg["weights"],
            device=self.device,
        )
        # 用第一张图确定特征维度，避免把 512+1024 硬编码
        with Image.open(fit_files[0]) as _im:
            _im.load()
            probe = self.extractor.embed(preprocess(_im, size, self.device))
        dim = int(probe.shape[1])
        del probe
        projection = make_projection(dim, int(self.model_cfg["projection_dim"]), seed)

        feats_all: list[np.ndarray] = []
        batch_size = 4
        buf: list[torch.Tensor] = []
        for k, f in enumerate(fit_files, 1):
            img = Image.open(f)
            img.load()
            buf.append(preprocess(img, size, self.device))
            if len(buf) == batch_size or k == len(fit_files):
                x = torch.cat(buf, 0)
                feats = self.extractor.embed(x)
                b, c, h, w = feats.shape
                flat = feats.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
                feats_all.append(flat @ projection)
                buf = []
                if progress:
                    progress(k, len(fit_files))
                if verbose and k % 32 == 0:
                    print(f"[{self.task}] 特征提取 {k}/{len(fit_files)}")
        feats_np = np.concatenate(feats_all, 0)
        if verbose:
            print(f"[{self.task}] patch 特征总数 {feats_np.shape[0]}")

        bank = build_memory_bank(
            feats_np,
            ratio=float(self.model_cfg["coreset_ratio"]),
            seed=seed,
        )
        meta = {
            "task": self.task,
            "backbone": self.model_cfg["backbone"],
            "weights": self.model_cfg["weights"],
            "layers": list(self.model_cfg["layers"]),
            "input_size": size,
            "projection_dim": int(self.model_cfg["projection_dim"]),
            "coreset_ratio": float(self.model_cfg["coreset_ratio"]),
            "bank_size": int(bank.shape[0]),
            "feature_dim": int(feats_np.shape[1]),
            "n_train_images": len(fit_files),
            "n_calib_images": len(val_files),
            "built_at": _now(),
            "device": str(self.device),
            "torch": torch.__version__,
        }
        self.bank = MemoryBank(bank=bank, projection=projection, meta=meta)
        self._proj_t = None
        self.bank.save(self.bank_path)

        # ---- 阈值标定（留出的正常图）----
        t0 = time.perf_counter()
        image_scores: list[float] = []
        pixel_scores: list[np.ndarray] = []
        for f in val_files:
            img = Image.open(f)
            img.load()
            amap, s, _ = self.score_map(img)
            image_scores.append(s)
            pixel_scores.append(amap.ravel())
        q = float(self.cfg["threshold"].get("normal_quantile", 0.995))
        # 留出集通常只有二三十张，0.995 分位接近最大值、统计不稳，
        # 再乘一个安全系数留出余量，避免线上正常样本略超就误报
        margin = float(self.cfg["threshold"].get("safety_margin", 1.0))
        if image_scores:
            t_img = float(np.quantile(image_scores, q)) * margin
            pix = np.concatenate(pixel_scores)
            t_pix = float(np.quantile(pix, q)) * margin
        else:
            t_img, t_pix = 0.0, 0.0
        calib_ms = (time.perf_counter() - t0) * 1000.0
        thr = {
            "image_score": round(t_img, 6),
            "pixel_score": round(t_pix, 6),
            "calibrated_from": f"train_good_holdout(n={len(val_files)})",
            "calibrated_at": _now(),
            "normal_quantile": q,
            "safety_margin": margin,
        }
        self.threshold = thr
        with self.threshold_path.open("w", encoding="utf-8") as fh:
            json.dump(thr, fh, ensure_ascii=False, indent=2)
        cfg = load_task_config(self.task)
        cfg["threshold"] = thr
        save_task_config(self.task, cfg)

        if verbose:
            print(f"[{self.task}] 特征库 {bank.shape[0]} 条，阈值 image={t_img:.4f} pixel={t_pix:.4f}")
            print(f"[{self.task}] 标定耗时 {calib_ms / 1000:.1f}s，模型 -> {self.bank_path}")
        return {
            "task": self.task,
            "bank_size": int(bank.shape[0]),
            "feature_patches": int(feats_np.shape[0]),
            "n_train_images": len(fit_files),
            "n_calib_images": len(val_files),
            "threshold": thr,
            "calib_seconds": round(calib_ms / 1000.0, 2),
            "model_path": str(self.bank_path),
        }


def _to_b64_jpeg(arr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    # 返回可直接塞进 <img src> 的 dataURL
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
