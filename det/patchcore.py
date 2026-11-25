"""PatchCore 风格的无监督异常检测核心（免训练）。

流程：
  正常图 --backbone--> layer2/layer3 多尺度 patch 特征
         --局部邻域聚合--> pad    --随机投影--> 低维特征
  全部正常 patch 特征 --k-center coreset--> memory bank
  测试图 patch 特征 --最近邻距离--> 异常图 --上采样+平滑--> 像素异常分
  异常图 top-k 均值 --> 图像异常分

整个过程不涉及任何缺陷图，也不需要反向传播。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def normalize_image(img: Image.Image) -> Image.Image:
    """统一输入图像的朝向与位深，保证检测框坐标与原图一致。

    - 按 EXIF Orientation 摆正（手机竖拍图否则会被当成横图，框会错位）
    - 16 位 / 浮点灰度按最大值线性拉伸到 8 位（直接 convert('RGB') 会全部截成 255）
    """
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img, dtype=np.float32)
        hi = float(arr.max()) if arr.size else 0.0
        if hi > 255.0:
            arr = arr * (255.0 / hi)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """WideResNet50-2 多尺度中间层特征提取（冻结，不训练）。"""

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        weights: str = "IMAGENET1K_V2",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or pick_device()
        w = getattr(Wide_ResNet50_2_Weights, weights)
        net = wide_resnet50_2(weights=w)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(self.device)
        self.layers = tuple(layers)
        self._out: dict[str, torch.Tensor] = {}
        for name in self.layers:
            getattr(self.net, name).register_forward_hook(self._hook(name))

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            self._out[name] = out

        return fn

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self._out = {}
        self.net(batch)
        return {k: v.detach() for k, v in self._out.items()}

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """返回 (N, C, H, W) 的融合 patch 特征，H/W 为 layer2 的尺度。"""
        feats = self.forward(batch)
        base = feats[self.layers[0]]
        parts = [base]
        for name in self.layers[1:]:
            f = feats[name]
            f = F.interpolate(f, size=base.shape[-2:], mode="bilinear", align_corners=False)
            parts.append(f)
        f = torch.cat(parts, dim=1)
        # 局部邻域聚合：3x3 平均池化后再 1x1，抑制单像素噪声
        f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
        return f


def preprocess(img: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    """缩放到 size 后只把 uint8 传到 GPU，归一化在 GPU 上做（省掉 CPU 浮点运算与 4 倍传输量）。"""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    t = t.to(dtype=torch.float32).div_(255.0)
    mean = _MEAN_T.to(device, non_blocking=True)
    std = _STD_T.to(device, non_blocking=True)
    return t.sub_(mean).div_(std)


def make_projection(dim_in: int, dim_out: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 1.0 / math.sqrt(dim_in), size=(dim_in, dim_out)).astype(np.float32)
    return m


def kcenter_greedy(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-center 贪心采样，返回选中下标。"""
    n = x.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    sel = np.empty(k, dtype=np.int64)
    first = int(rng.integers(n))
    sel[0] = first
    d = ((x - x[first]) ** 2).sum(axis=1)
    for i in range(1, k):
        idx = int(np.argmax(d))
        sel[i] = idx
        d = np.minimum(d, ((x - x[idx]) ** 2).sum(axis=1))
    return sel


def build_memory_bank(
    features: np.ndarray,
    ratio: float,
    seed: int = 0,
    max_candidates: int = 30000,
    min_points: int = 128,
) -> np.ndarray:
    """从全部正常 patch 特征里抽样出一个紧凑特征库。"""
    n = features.shape[0]
    target = max(min_points, int(round(n * ratio)))
    rng = np.random.default_rng(seed)
    if n > max_candidates:
        idx = rng.choice(n, max_candidates, replace=False)
        idx.sort()
        cand = features[idx]
    else:
        cand = features
    target = min(target, cand.shape[0])
    sel = kcenter_greedy(cand, target, seed)
    return cand[sel]


def nearest_distances(q: np.ndarray, bank: np.ndarray, chunk: int = 4096,
                      device: torch.device | None = None) -> np.ndarray:
    """每个 query patch 到特征库的最近距离（L2）。给了 CUDA 设备就走 GPU 计算。"""
    use_cuda = device is not None and device.type == "cuda"
    if use_cuda:
        q_t = torch.from_numpy(q).to(device, non_blocking=True)
        b_t = torch.from_numpy(bank).to(device, non_blocking=True)
    else:
        q_t = torch.from_numpy(q)
        b_t = torch.from_numpy(bank)
    out = torch.empty(q_t.shape[0], dtype=torch.float32, device=q_t.device)
    for i in range(0, q_t.shape[0], chunk):
        d = torch.cdist(q_t[i: i + chunk], b_t)
        out[i: i + chunk] = d.min(dim=1).values
    return out.cpu().numpy()


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    import cv2

    return cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


class MemoryBank:
    """一个任务的正常样本特征库 + 投影矩阵 + 标定信息。"""

    def __init__(
        self,
        bank: np.ndarray,
        projection: np.ndarray,
        meta: dict,
    ) -> None:
        self.bank = bank.astype(np.float32)
        self.projection = projection.astype(np.float32)
        self.meta = meta

    # ---- 持久化 -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            bank=self.bank.astype(np.float16),
            projection=self.projection,
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        with np.load(path, allow_pickle=False) as z:
            bank = z["bank"].astype(np.float32)
            proj = z["projection"].astype(np.float32)
            meta = json.loads(str(z["meta"]))
        return cls(bank=bank, projection=proj, meta=meta)

    @property
    def size(self) -> int:
        return int(self.bank.shape[0])
