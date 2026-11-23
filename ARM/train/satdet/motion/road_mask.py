"""道路掩膜：用"车只会在路上出现"这个先验剔除 90% 的非道路动态噪点。

为什么不在图上涂黑
------------------
把非道路区域涂黑会在掩膜边界引入**强梯度**，正好是小目标检测最怕的干扰
（网络会把黑边当纹理）。本项目只在 **cell 打分阶段**判断"这个格子要不要看"，
图像本身一个像素都不改。

先验从哪来（三条路，按可用性排序）
--------------------------------
1. **运动累积热图（默认，无监督）**：把若干帧的运动掩膜累加、模糊、取分位阈值。
   车辆只在道路上运动，累积起来的带状区域天然就是路网。
   优点：不需要标注、不需要下载数据集、每个序列自适应。
2. **人工多边形**：用 ``polygon`` 方式加载 JSON（[x,y] 列表），适合现场精修。
3. 语义分割预训练模型（DeepGlobe + UNet）：依赖下载与额外模型，本项目暂不引入。

保存格式
--------
``artifacts/road/<seq>_roadmask.npz``：``mask`` (H,W) uint8（255=道路）、
``meta`` 里记录帧数/阈值/来源，便于复现。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from satdet.core.logging import get_logger

__all__ = ["build_road_prior", "save_road_mask", "load_road_mask",
           "apply_road_to_cells", "cell_centers"]

log = get_logger("road_mask")


def cell_centers(width: int, height: int, tile: int) -> np.ndarray:
    """返回 (ny*nx, 2) 的 cell 中心像素坐标，顺序与 ``cell_scores.ravel()`` 一致。"""
    ny = max(1, int(np.ceil(height / tile)))
    nx = max(1, int(np.ceil(width / tile)))
    ys = np.arange(ny) * tile + tile / 2.0
    xs = np.arange(nx) * tile + tile / 2.0
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def build_road_prior(mask_frames: list[np.ndarray], width: int, height: int,
                     blur: int = 31, quantile: float = 0.80,
                     dilate: int = 5) -> np.ndarray:
    """由若干帧运动掩膜累积成道路先验。

    Parameters
    ----------
    mask_frames : list of (H,W) uint8
        运动掩膜（0/255）。
    blur : 高斯核，把散点连成带状。
    quantile : 对"每像素出现过运动的帧占比"取该分位作为阈值。
        0.80 意味着"至少 80% 的分位水平以上"才当道路，避免偶发噪点被连成片。
    """
    if not mask_frames:
        raise ValueError("没有可用的运动掩膜帧")
    acc = np.zeros((height, width), np.float32)
    for m in mask_frames:
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        acc += (m > 0).astype(np.float32)
    acc /= float(len(mask_frames))          # 每像素"出现运动的帧占比" 0~1

    if blur >= 3:
        acc = cv2.GaussianBlur(acc, (blur, blur), 0)

    thr = float(np.quantile(acc, quantile))
    thr = max(thr, 1e-4)
    road = (acc >= thr).astype(np.uint8) * 255

    if dilate >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        road = cv2.dilate(road, k)
    log.info("道路先验: 覆盖 %.1f%% 像素 (阈值=%.4f, 帧数=%d)",
             100.0 * (road > 0).mean(), thr, len(mask_frames))
    return road


def save_road_mask(road: np.ndarray, path: str | Path, meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, mask=road, meta=np.array([str(meta or {})]))
    return p


def load_road_mask(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    with np.load(p) as z:
        return z["mask"]


def apply_road_to_cells(scores: np.ndarray, road: np.ndarray, tile: int,
                        width: int, height: int, min_road_frac: float = 0.02,
                        soft: bool = False) -> np.ndarray:
    """按道路先验给 cell 打分加权/清零。

    ``min_road_frac``：格内道路像素占比达到该值才保留。
    比"只判断格中心是否在路上"宽容得多——cell 是 256×256，中心点很容易
    落在两车道之间的绿地上，但格内其实有路。
    ``soft=True`` 时不丢弃，而是按道路占比缩放分数。
    """
    if road is None or scores.size == 0:
        return scores
    ny, nx = scores.shape
    ratio = cv2.resize((road > 0).astype(np.float32), (nx, ny),
                       interpolation=cv2.INTER_AREA)
    out = scores.astype(np.float64).copy()
    if soft:
        out *= np.clip(ratio, 0.0, 1.0)
    else:
        out[ratio < float(min_road_frac)] = 0.0
    return out
