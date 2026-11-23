"""车速换算：像素位移 -> 物理速度（长时序差分 + EWMA）。

为什么不用逐帧差分（第二轮修正，计划2 条目 2）
--------------------------------------------
逐帧位移只有 2~3 px，叠上检测抖动（±1 px）后相对误差可达 ±30%~50%，
表现为车速尖峰（实测出现过 141 km/h）。改用 **当前帧与 t−K 帧的位移**
（默认 K=12）后，位移涨到 24~36 px，同样的 1 px 抖动只占 ±3%，
再做 EWMA(α=0.7) 平滑，序列基本单调。

数学
----
``v[km/h] = 3.6 × (位移(t) − 位移(t−K)) / K × k(米/像素) × fps``

``k`` 是唯一需要标定的量（见 docs/12）。**未标定就不出物理速度**：
输出 px/s 并标记 ``unit="px/s"``，避免在报告里给出错误结论。

降级策略
--------
轨迹没有 ``box`` 字段（部分单测/轻量调用）或历史不足 ``min_gap`` 帧时，
退回卡尔曼速度 ``hypot(vx, vy)`` + 中值平滑，保证接口向后兼容。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SpeedConfig", "SpeedEstimator", "estimate_speed"]


@dataclass
class SpeedConfig:
    """车速参数（与 configs/measure.yaml 对应）。"""

    fps: float = 10.0
    meters_per_pixel: float | None = None      # None = 未标定

    # ---- 长时序差分 ----
    interval_frames: int = 12                  # 与 t-K 帧比较；太小则抖动大，太大则跟不上变速
    min_gap: int = 3                           # 至少跨 3 帧才算有效差分
    ewma_alpha: float = 0.7                    # 新值权重；0.7 = 响应快且能压住尖峰
    max_gap_miss: int = 6                      # 找不到接近 t-K 的样本时容忍的帧差
    multi_lags: tuple[int, ...] = (6, 12, 18)  # 多个滞后取中位，降低单点异常的影响
    max_px_per_frame: float = 6.0              # 物理上限（6 px/帧 ≈ 200 km/h @0.92/10fps）
    uncalibrated_unit: str = "px/s"            # 未标定时显示单位：px/s | px/帧（帧率未知时用后者）
    max_slew_kmh: float = 35.0                 # 单帧最大变化量（限幅，抑制尖峰）

    # ---- 兼容字段（卡尔曼速度降级路径）----
    window_frames: int = 5
    min_track_age: int = 5
    min_hit_ratio: float = 0.6

    @property
    def calibrated(self) -> bool:
        return self.meters_per_pixel is not None and float(self.meters_per_pixel) > 0


@dataclass
class SpeedEstimator:
    """把轨迹位移转成速度 + 置信度。

    对外只暴露 ``update(tracks, frame=None) -> list[dict]``，
    每个 dict 在 ``TrackState.to_dict()`` 的基础上追加：
    ``speed`` / ``speed_unit`` / ``speed_conf`` / ``calibrated`` /
    ``meters_per_pixel`` / ``px_per_frame`` / ``interval`` / ``speed_src``。
    """

    cfg: SpeedConfig = field(default_factory=SpeedConfig)
    _buf: dict = field(default_factory=dict, repr=False)      # 中值平滑（降级路径）
    _hist: dict = field(default_factory=dict, repr=False)     # id -> deque[(frame, cx, cy)]
    _ewma: dict = field(default_factory=dict, repr=False)     # id -> 平滑后的 px/帧
    _frame: int = 0

    # ------------------------------------------------------------------
    def _push(self, tid: int, frame: int, cx: float, cy: float) -> None:
        h = self._hist.get(tid)
        if h is None:
            h = deque(maxlen=max(8, self.cfg.interval_frames * 3))
            self._hist[tid] = h
        h.append((frame, float(cx), float(cy)))

    def _long_interval_px(self, tid: int, frame: int) -> tuple[float, int] | None:
        """用多个滞后（6/12/18 帧）的位移取**中位**，得到 px/帧。

        为什么用中位而不是单一滞后：一次 ID 跳变或漏检会让某个滞后端的坐标
        变到别的车上，单滞后会直接把速度拉到 200 km/h 以上；多滞后取中位后，
        只要多数滞后端是同一辆车，结果就是干净的。找不到合格样本返回 None。
        """
        h = self._hist.get(tid)
        if not h or len(h) < 2:
            return None
        f_now, x_now, y_now = h[-1]
        if f_now != frame:
            return None

        samples: list[tuple[float, int]] = []
        for lag in self.cfg.multi_lags:
            target = frame - lag
            best = None
            for f, x, y in h:
                gap = f_now - f
                if gap < self.cfg.min_gap:
                    continue
                if abs(f - target) <= self.cfg.max_gap_miss:
                    if best is None or abs(f - target) < abs(best[0] - target):
                        d = float(np.hypot(x_now - x, y_now - y))
                        best = (f, d, gap)
            if best is not None:
                samples.append((best[1] / float(best[2]), best[2]))

        if not samples:
            return None
        vals = np.asarray([s[0] for s in samples], dtype=np.float64)
        return float(np.median(vals)), int(samples[len(samples) // 2][1])

    def _smooth_median(self, tid: int, raw: float) -> float:
        buf = self._buf.setdefault(tid, [])
        buf.append(raw)
        if len(buf) > self.cfg.window_frames:
            buf.pop(0)
        return float(np.median(buf))

    # ------------------------------------------------------------------
    def update(self, tracks: list, frame: int | None = None) -> list[dict]:
        """``tracks`` 为 TrackState 列表（需有 ``track_id`` 与可选 ``box``）。"""
        self._frame = (self._frame + 1) if frame is None else int(frame)
        f = self._frame
        out: list[dict] = []

        for t in tracks:
            tid = int(t.track_id)
            box = getattr(t, "box", None)
            px_per_frame = None
            gap = 0
            src = "kalman"

            if box is not None:
                cx = float(box[0] + box[2]) * 0.5
                cy = float(box[1] + box[3]) * 0.5
                self._push(tid, f, cx, cy)
                got = self._long_interval_px(tid, f)
                if got is not None:
                    px_per_frame, gap = got
                    src = "interval"
                    prev = self._ewma.get(tid)
                    if prev is None:
                        # 首个区间样本用卡尔曼速度做初值，从第一帧起就有限幅保护，
                        # 否则"首帧无 prev"会漏过 slew 限制，造成 100+ km/h 的突变
                        prev = float(np.hypot(getattr(t, "vx", 0.0) or 0.0,
                                              getattr(t, "vy", 0.0) or 0.0))
                    a = float(np.clip(self.cfg.ewma_alpha, 0.0, 1.0))
                    px_per_frame = (px_per_frame if prev is None
                                    else a * px_per_frame + (1.0 - a) * prev)
                    # 物理上限：超过 max_px_per_frame 视为跟踪跳变，保持上一值
                    # 先给 prev 也上物理上限，否则"保持上一值"会把历史的异常值一直传下去
                    if prev is not None:
                        prev = float(min(prev, self.cfg.max_px_per_frame))
                    if px_per_frame > self.cfg.max_px_per_frame:
                        if prev is not None:
                            px_per_frame = float(prev)
                            src = "held"
                        else:
                            px_per_frame = float(self.cfg.max_px_per_frame)
                            src = "capped"
                    # 限幅：单帧变化不超过 max_slew_kmh 对应的像素速度
                    if prev is not None:
                        slew = self.cfg.max_slew_kmh / max(1e-6, 3.6 * float(self.cfg.meters_per_pixel or 1.0) * self.cfg.fps)
                        lo, hi = float(prev) - slew, float(prev) + slew
                        px_per_frame = float(np.clip(px_per_frame, max(0.0, lo), hi))
                    self._ewma[tid] = px_per_frame

            if px_per_frame is None:
                vx = float(getattr(t, "vx", 0.0) or 0.0)
                vy = float(getattr(t, "vy", 0.0) or 0.0)
                px_per_frame = self._smooth_median(tid, float(np.hypot(vx, vy)))

            if self.cfg.calibrated:
                v = 3.6 * px_per_frame * float(self.cfg.meters_per_pixel) * self.cfg.fps
                unit = "km/h"
            elif self.cfg.uncalibrated_unit == "px/帧":
                v = px_per_frame                       # 帧率未知的异源序列：只给 px/帧
                unit = "px/帧"
            else:
                v = px_per_frame * self.cfg.fps        # px/s
                unit = "px/s"

            age = float(getattr(t, "age", 0) or 0)
            conf = 1.0 if age >= self.cfg.min_track_age else age / max(1, self.cfg.min_track_age)

            row = t.to_dict() if hasattr(t, "to_dict") else {}
            row.update({
                "speed": v,
                "speed_unit": unit,
                "speed_conf": float(min(1.0, max(0.0, conf))),
                "calibrated": self.cfg.calibrated,
                "meters_per_pixel": self.cfg.meters_per_pixel,
                "px_per_frame": float(px_per_frame),
                "interval": int(gap),
                "speed_src": src,
            })
            out.append(row)
        return out

    def reset(self) -> None:
        self._buf.clear()
        self._hist.clear()
        self._ewma.clear()
        self._frame = 0


def estimate_speed(displacement_px_per_frame: float, fps: float,
                   meters_per_pixel: float) -> float:
    """单点换算，便于单测与文档示例。"""
    return 3.6 * float(displacement_px_per_frame) * float(meters_per_pixel) * float(fps)
