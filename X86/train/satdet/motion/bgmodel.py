"""时域背景模型 + 前景掩膜（第二轮重写：解决"掩膜几乎空"）。

第一轮的实现有两个致命问题（由 check-effect 诊断确认）
----------------------------------------------------
1. **1/4 降采样用面积平均**：6×6 px 的车被摊进 4×4=16 个原始像素，
   对比度稀释约 16 倍（~40 灰阶 → ~5），直接被阈值杀掉。
   实测单帧只剩 **10 个运动像素**，GT 覆盖 37%。
2. **3×3 开运算吃掉孤立小目标**：面积平均后一辆车只剩 1~2 个像素，
   开运算（先腐蚀）把它们整体抹除，掩膜几乎全黑。

本轮的修正（对应计划2 条目 4/8）
--------------------------------
* ``diff_scale``：差分可选在全分辨率（1）或 1/2 上算；降采样一律用 **MaxPool**
  保极值，绝不用面积平均。
* ``morph="close"``：默认只做闭运算（先膨胀后腐蚀），半径可调；``"none"`` 完全不做。
* ``thr_mode``：``global``（mean+kσ）/ ``otsu`` / ``local``（局部均值+bias，积分图加速）。
* ``abs_protect``：绝对差超过该值的像素**强制保留**（在形态学之后叠加），
  保证"亮车 vs 暗路"这类强信号不会被形态学误杀。

设计原则：**掩膜宁可多给一些像素，也不要漏掉目标**——多给的像素由道路掩膜
和 cell 打分去抑制，漏掉的目标则完全无法挽回。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

__all__ = [
    "RunningAverageBackground",
    "MedianBackground",
    "to_gray",
    "maxpool_down",
    "foreground_mask",
    "mask_to_boxes",
]


def to_gray(frame: np.ndarray, downscale: int = 1, pool: str = "max") -> np.ndarray:
    """转灰度并降采样。

    ``pool="max"`` 用最大值池化（保留小目标的峰值），``"area"`` 才是面积平均。
    第一轮默认 area，是掩膜变空的直接原因，这里把默认改成 max。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if downscale > 1:
        if pool == "max":
            gray = maxpool_down(gray, downscale)
        else:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (max(1, w // downscale), max(1, h // downscale)),
                              interpolation=cv2.INTER_AREA)
    return gray


def maxpool_down(img: np.ndarray, k: int) -> np.ndarray:
    """整数倍最大值池化（保留极值）。尺寸除不尽时先裁到整数倍。"""
    if k <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = (h // k) * k, (w // k) * k
    if h2 == 0 or w2 == 0:
        return img
    v = img[:h2, :w2]
    v = v.reshape(h2 // k, k, w2 // k, k)
    return v.max(axis=(1, 3))


class RunningAverageBackground:
    """指数滑动平均背景。

    Parameters
    ----------
    alpha : float
        学习率。越小背景越"迟钝"，慢速目标越不容易被吸收。
        实测 0.02~0.05 在 10 fps 卫星视频上合适。
    downscale : int
        背景维护分辨率（1 = 全分辨率）。
    pool : str
        降采样方式，见 :func:`to_gray`。
    """

    def __init__(self, alpha: float = 0.02, downscale: int = 1, pool: str = "max") -> None:
        self.alpha = float(alpha)
        self.downscale = int(downscale)
        self.pool = pool
        self._bg: np.ndarray | None = None
        self.n_updates = 0

    def update(self, frame: np.ndarray) -> None:
        gray = to_gray(frame, self.downscale, self.pool).astype(np.float32)
        if self._bg is None:
            self._bg = gray.copy()
        else:
            cv2.accumulateWeighted(gray, self._bg, self.alpha)
        self.n_updates += 1

    @property
    def background(self) -> np.ndarray | None:
        return None if self._bg is None else self._bg.astype(np.uint8)

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        """帧与背景的绝对差（uint8，0~255）。

        用 ``cv2.absdiff`` 而不是 float 减法：1024² 上从 2.4 ms 降到约 0.4 ms，
        精度对阈值判定足够（差分本来就只有灰阶量级）。
        """
        gray = to_gray(frame, self.downscale, self.pool)
        if self._bg is None:
            return np.zeros_like(gray, dtype=np.uint8)
        return cv2.absdiff(gray, self._bg.astype(np.uint8))


class MedianBackground:
    """最近 N 帧中值背景（抗慢速目标吸收，但更贵）。"""

    def __init__(self, n: int = 15, downscale: int = 1, pool: str = "max") -> None:
        self.n = int(n)
        self.downscale = int(downscale)
        self.pool = pool
        self._buf: deque[np.ndarray] = deque(maxlen=self.n)

    def update(self, frame: np.ndarray) -> None:
        self._buf.append(to_gray(frame, self.downscale, self.pool))

    @property
    def background(self) -> np.ndarray | None:
        if len(self._buf) < max(3, self.n // 3):
            return None
        return np.median(np.stack(self._buf, axis=0), axis=0).astype(np.uint8)

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        bg = self.background
        gray = to_gray(frame, self.downscale, self.pool).astype(np.float32)
        if bg is None:
            return np.zeros_like(gray)
        return np.abs(gray - bg.astype(np.float32))


def _local_threshold(diff: np.ndarray, win: int = 31, bias: float = 6.0,
                     floor: float = 3.0) -> np.ndarray:
    """局部均值 + 偏置的逐像素阈值（boxFilter 近似积分图，O(n)）。"""
    mean = cv2.boxFilter(diff, ddepth=cv2.CV_32F, ksize=(win, win), normalize=True)
    return np.maximum(floor, mean + bias)


def foreground_mask(diff: np.ndarray, thr: float | None = None, k_sigma: float = 4.0,
                    min_thr: float = 8.0, kernel: int = 3, morph: str = "close",
                    thr_mode: str = "global", abs_protect: float = 0.0,
                    local_win: int = 31, local_bias: float = 6.0,
                    pct: float = 99.5) -> np.ndarray:
    """差异图 -> 二值前景掩膜。

    Parameters
    ----------
    morph : {"close", "none", "open_close", "dilate"}
        形态学策略。默认 ``close``（补洞、不杀孤立点）。
        ``none`` 完全不处理；``open_close`` 是**旧行为**（会抹掉 1~2 px 的车）。
    thr_mode : {"global", "otsu", "local"}
        阈值策略。``global`` = max(min_thr, mean+kσ)；``otsu`` 双峰；
        ``local`` = 局部均值+bias（云影/明暗变化场景更稳）。
    abs_protect : float
        绝对差保护：``diff > abs_protect`` 的像素在形态学之后强制置 255。
        0 表示关闭。目的：强信号（亮车）不被形态学或阈值误杀。
    """
    if diff.size == 0:
        return np.zeros((0, 0), np.uint8)

    # ---- 阈值 ----
    if thr is not None:
        thr_map: float | np.ndarray = float(thr)
    elif thr_mode == "local":
        thr_map = _local_threshold(diff, local_win, local_bias, min_thr * 0.5)
    elif thr_mode == "otsu":
        u8 = diff.astype(np.uint8, copy=False)
        t_otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_map = max(float(t_otsu), min_thr * 0.5)
    elif thr_mode == "percentile":
        # 数据驱动：取差分分布的高分位，自动适应不同场景的噪声底
        thr_map = max(float(min_thr), float(np.percentile(diff, pct)))
    else:
        if k_sigma <= 0:
            # 固定阈值：跳过 mean/std（1024² 上这两个统计要 4 ms，白花）
            thr_map = float(min_thr)
        else:
            mu, sd = float(diff.mean()), float(diff.std())
            thr_map = max(min_thr, mu + k_sigma * sd)

    mask = (diff > thr_map).astype(np.uint8) * 255

    # ---- 形态学：默认只闭运算 ----
    if morph != "none" and kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        if morph == "open_close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        elif morph == "dilate":
            mask = cv2.dilate(mask, k)
        else:  # close
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # ---- 绝对差保护：强信号强制保留 ----
    if abs_protect > 0:
        mask = np.where(diff > abs_protect, 255, mask).astype(np.uint8)

    return mask


def mask_to_boxes(mask: np.ndarray, min_area: int = 4, max_area: int = 20000,
                  max_boxes: int = 400) -> np.ndarray:
    """连通域 -> 框 (N,4) xyxy（掩膜坐标系）。

    面积下限从旧的 8 降到 4：全分辨率下一辆 6×6 的车约 20~36 px²，
    阈值+闭运算后可能只剩 6~10 px²，旧的 8 会漏掉一部分。
    """
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        boxes.append((x, y, x + w, y + h))
        if len(boxes) >= max_boxes:
            break
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
