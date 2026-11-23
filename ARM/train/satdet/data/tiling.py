"""Tiling：把"整帧检测"换成"选择性图块精检"的纯几何实现。

为什么需要 tiling（本项目最核心的一步）
--------------------------------------
VISO 实测目标边长中位数只有 5~9 px（1024x1024 帧）。整帧送进 YOLOv8 时，
6 px 目标在 stride=32 的 P5 特征图上只剩 0.19 px，结构上不可能学到；
即便加 P2(stride 4)，6 px 也只有 1.5 格。把 256x256 图块放大 2 倍到 512
后，同一目标变成 12 px，在 P2 上有 3 格，才真正"可学"。

关键实测结论（2025-11，见 experiments/results/tiling_curve.csv）
---------------------------------------------------------------
逐目标各自裁一块（snap=32）在密集车流下**比整帧更贵**：
005 段每帧 34~65 个目标，会裁出 45 块 256 → 输入像素达到整帧的 283%。
把图块原点对齐到图块网格（``snap=tile``）后，同格目标共享图块，
只需 8~9 块 → 像素量降到 42~56%，而覆盖率仍 >=99.9%。
**因此本项目默认 snap = tile，即"自适应网格"。**

对比口径（报告里必须写清是哪一个）
----------------------------------
* ``pixel_ratio``             相对"整帧 1024 输入"
* ``pixel_ratio_vs_upscaled`` 相对"要达到同等分辨率的整帧 2048 输入"（约 12%）

所有函数输入 ``boxes`` 为 ``[N, 4]`` 的 ``x1 y1 x2 y2``（与 ann_parsers 一致）。
本模块不读图片、不依赖 torch/cv2，因此可单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Tile",
    "grid_origins",
    "roi_origins",
    "clip_origins",
    "boxes_in_tile",
    "covered_mask",
    "covered_fraction",
    "pixel_ratio",
    "evaluate_plan",
    "tiling_curve",
]


@dataclass(frozen=True)
class Tile:
    """一个图块：左上角 + 边长。"""

    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


# =====================================================================
# 原点方案
# =====================================================================
def grid_origins(width: int, height: int, tile: int, stride: int | None = None) -> np.ndarray:
    """均匀网格覆盖，返回 ``(M, 2)`` 的 ``[x0, y0]``。

    最后一列/行贴边对齐（``W - tile``），保证不漏右/下边缘，代价是边缘重叠。
    """
    if tile <= 0 or tile > width or tile > height:
        raise ValueError(f"tile={tile} 必须满足 0 < tile <= min(W,H)={min(width, height)}")
    stride = tile if stride is None else max(1, stride)

    xs = list(range(0, max(1, width - tile + 1), stride))
    if xs[-1] != width - tile:
        xs.append(width - tile)
    ys = list(range(0, max(1, height - tile + 1), stride))
    if ys[-1] != height - tile:
        ys.append(height - tile)

    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.int32)


def roi_origins(
    boxes: np.ndarray,
    tile: int,
    width: int,
    height: int,
    snap: int | None = None,
) -> np.ndarray:
    """按 RoI 生成图块原点：让每个目标尽量落在图块中心。

    Parameters
    ----------
    boxes : (N, 4)
        感兴趣框。GT 框、帧差连通域、卡尔曼预测框都可以。
    snap : int | None
        原点对齐到该像素倍数。**默认 ``tile``（网格对齐）**：
        同一格里的多个目标共享一个图块，这是像素量能低于整帧的原因；
        设成 32/64 就退化回"逐目标裁剪"，密集场景会比整帧更贵。
    dilate : 见 :func:`_dilate`（未启用，保留说明）

    Returns
    -------
    (M, 2) int32，已去重、已裁进图像范围。
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    snap = int(tile if snap is None else max(1, snap))
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5

    # 把目标中心换成"以中心为原点、边长 tile 的窗口左上角"，再按 snap 取整
    x0 = np.round((cx - tile * 0.5) / snap) * snap
    y0 = np.round((cy - tile * 0.5) / snap) * snap
    x0 = np.clip(x0, 0, max(0, width - tile))
    y0 = np.clip(y0, 0, max(0, height - tile))

    origins = np.stack([x0, y0], axis=1).astype(np.int32)
    return np.unique(origins, axis=0)


def clip_origins(origins: np.ndarray, width: int, height: int, tile: int) -> np.ndarray:
    """把原点裁进 ``[0, W-tile] x [0, H-tile]`` 并去重。"""
    if origins is None or len(origins) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    o = np.asarray(origins, dtype=np.int32).reshape(-1, 2).copy()
    o[:, 0] = np.clip(o[:, 0], 0, max(0, width - tile))
    o[:, 1] = np.clip(o[:, 1], 0, max(0, height - tile))
    return np.unique(o, axis=0)


# =====================================================================
# 框与图块的几何关系
# =====================================================================
def boxes_in_tile(
    boxes: np.ndarray,
    x0: int,
    y0: int,
    tile: int,
    min_visible: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """挑出落在图块内的框，并换算成图块局部坐标。

    Returns
    -------
    keep : (N,) bool
    local : (K, 4) float32，已裁到 ``[0, tile]``
    """
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if b.size == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 4), dtype=np.float32)

    local = b - np.asarray([x0, y0, x0, y0], dtype=np.float32)
    lx1 = np.clip(local[:, 0], 0, tile)
    ly1 = np.clip(local[:, 1], 0, tile)
    lx2 = np.clip(local[:, 2], 0, tile)
    ly2 = np.clip(local[:, 3], 0, tile)

    inter = np.clip(lx2 - lx1, 0, None) * np.clip(ly2 - ly1, 0, None)
    area = (np.clip(local[:, 2] - local[:, 0], 1e-6, None)
            * np.clip(local[:, 3] - local[:, 1], 1e-6, None))
    visible = inter / area

    keep = visible >= float(min_visible)
    return keep, np.stack([lx1[keep], ly1[keep], lx2[keep], ly2[keep]], axis=1).astype(np.float32)


def covered_mask(
    boxes: np.ndarray,
    origins: np.ndarray,
    tile: int,
    min_visible: float = 0.5,
) -> np.ndarray:
    """逐框判断是否至少被一个图块以足够可见度覆盖。"""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    covered = np.zeros(len(b), dtype=bool)
    if len(b) == 0 or origins is None or len(origins) == 0:
        return covered
    for x0, y0 in np.asarray(origins, dtype=np.int32).reshape(-1, 2):
        keep, _ = boxes_in_tile(b, int(x0), int(y0), tile, min_visible)
        covered |= keep
        if covered.all():
            break
    return covered


def covered_fraction(
    boxes: np.ndarray,
    origins: np.ndarray,
    tile: int,
    min_visible: float = 0.5,
) -> float:
    """覆盖率 = 被覆盖框 / 总框；空帧返回 1.0（无目标即无漏检）。"""
    b = np.asarray(boxes).reshape(-1, 4)
    if len(b) == 0:
        return 1.0
    return float(covered_mask(b, origins, tile, min_visible).mean())


def pixel_ratio(origins: np.ndarray, tile: int, width: int, height: int) -> float:
    """输入像素占比 = 图块总面积 / 全图像素。"""
    n = 0 if origins is None else int(np.asarray(origins).reshape(-1, 2).shape[0])
    return float(n * tile * tile) / float(max(1, width * height))


# =====================================================================
# 方案评估
# =====================================================================
def evaluate_plan(
    boxes_per_frame: Sequence[np.ndarray],
    width: int,
    height: int,
    origin_fn: Callable[[np.ndarray], np.ndarray],
    tile: int,
    min_visible: float = 0.5,
    max_boxes: int = 300,
) -> dict[str, float]:
    """跑遍全部帧，统计平均图块数 / 像素占比 / 覆盖率。

    Parameters
    ----------
    origin_fn : 给定一帧的框，返回图块原点
    max_boxes : 每帧最多参与计算的目标数（密集场景控制耗时）
    """
    tiles_total = 0
    covered_num = 0.0
    covered_den = 0
    frames_used = 0

    for boxes in boxes_per_frame:
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(b) > max_boxes:
            idx = np.linspace(0, len(b) - 1, max_boxes).astype(int)
            b = b[idx]
        origins = origin_fn(b)
        tiles_total += int(len(origins)) if origins is not None and len(origins) else 0
        if len(b):
            covered_num += float(covered_mask(b, origins, tile, min_visible).sum())
            covered_den += len(b)
        frames_used += 1

    return {
        "frames": float(frames_used),
        "tiles_per_frame": tiles_total / max(1, frames_used),
        "pixel_ratio": (tiles_total * tile * tile) / float(max(1, frames_used * width * height)),
        "coverage": covered_num / covered_den if covered_den else 1.0,
    }


def tiling_curve(
    boxes_per_frame: Sequence[np.ndarray],
    width: int,
    height: int,
    tile_sizes: Sequence[int] = (128, 192, 256, 320, 384, 512),
    min_visible: float = 0.5,
    max_boxes: int = 300,
    upscale: float = 2.0,
) -> list[dict[str, float]]:
    """扫图块边长，输出 RoI(网格对齐) 与均匀网格两种方案的像素/覆盖率。

    这条曲线直接回答"为什么选 256 而不是 128/512"，写进报告与 PPT。
    """
    rows: list[dict[str, float]] = []
    for tile in tile_sizes:
        if tile > min(width, height):
            continue
        roi = evaluate_plan(boxes_per_frame, width, height,
                            lambda b, t=int(tile): roi_origins(b, t, width, height, snap=t),
                            int(tile), min_visible, max_boxes)
        grid = evaluate_plan(boxes_per_frame, width, height,
                             lambda b, t=int(tile): grid_origins(width, height, t),
                             int(tile), min_visible, max_boxes)
        rows.append({
            "tile": float(tile),
            "roi_tiles_per_frame": roi["tiles_per_frame"],
            "roi_pixel_ratio": roi["pixel_ratio"],
            "roi_pixel_ratio_vs_upscaled": roi["pixel_ratio"] / (upscale ** 2),
            "roi_coverage": roi["coverage"],
            "grid_tiles_per_frame": grid["tiles_per_frame"],
            "grid_pixel_ratio": grid["pixel_ratio"],
            "grid_coverage": grid["coverage"],
        })
    return rows
