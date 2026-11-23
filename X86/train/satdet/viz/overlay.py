"""叠加渲染：框 + ID + 车速。

小目标（6 px）的显示要点
-----------------------
* 标签不能压在目标上，放到框右上角外侧；
* 目标太密时按速度降序只显示前 ``max_labels`` 个，避免糊成一片；
* 框线宽固定 1 px，颜色按速度分级（见 configs/measure.yaml）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["OverlayConfig", "draw_tracks", "make_writer"]


@dataclass
class OverlayConfig:
    """显示参数。"""

    font_scale: float = 0.35
    thickness: int = 1
    text_offset: tuple[int, int] = (2, -4)
    max_labels: int = 20
    box_color: tuple[int, int, int] = (0, 200, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)
    unit: str = "km/h"
    color_scale: list = field(default_factory=lambda: [
        (40.0, (160, 160, 160)),
        (80.0, (80, 200, 80)),
        (120.0, (60, 200, 240)),
        (1e9, (60, 60, 240)),
    ])


def _color_for(speed: float, cfg: OverlayConfig):
    for thr, color in cfg.color_scale:
        if speed <= thr:
            return color
    return cfg.box_color


def draw_tracks(img: np.ndarray, tracks: list[dict], cfg: OverlayConfig | None = None,
                show_speed: bool = True) -> np.ndarray:
    """在图像上画框与标签（就地修改并返回）。"""
    cfg = cfg or OverlayConfig()
    if not tracks:
        return img
    ranked = sorted(tracks, key=lambda t: -float(t.get("speed", 0.0)))[:cfg.max_labels]
    for t in ranked:
        b = t.get("box")
        if b is None:
            b = (t["x1"], t["y1"], t["x2"], t["y2"])
        x1, y1, x2, y2 = (int(round(float(v))) for v in b)
        speed = float(t.get("speed", 0.0))
        color = _color_for(speed, cfg)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cfg.thickness)
        if show_speed:
            tid = int(t.get("track_id", -1))
            label = f"#{tid} {speed:.0f}"
            if cfg.unit:
                label += f" {cfg.unit}"
            tx = x2 + cfg.text_offset[0]
            ty = max(10, y1 + cfg.text_offset[1])
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, (0, 0, 0), cfg.thickness + 1, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, cfg.text_color, cfg.thickness, cv2.LINE_AA)
    return img


def make_writer(path: str, width: int, height: int, fps: float = 10.0):
    """创建 VideoWriter（mp4v；文件不存在时返回 None 而不是抛异常）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
