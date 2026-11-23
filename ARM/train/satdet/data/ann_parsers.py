"""VISO 标注解析：把 4 种异构标注统一成 (frame, id, xyxy, score)。

实测格式（见 02_数据说明.md，数据核查于 2025-XX）
--------------------------------------------------------------------
=============  =====================================================
文件            格式
=============  =====================================================
label.txt      空格分隔: ``frame id x1 y1 x2 y2 -1 -1 -1 -1``
               注意 x2/y2 是右/下边界，宽度 ``w = x2 - x1``（非 +1）
gt2.txt        逗号分隔 MOT 风格: ``frame id x y w h conf ...``
               id 恒为 -1，conf = 1；是 label.txt 的清洗版
mot/gt.txt     逗号分隔 MOT 风格，**带真实轨迹 ID**，conf = 1
det.txt        逗号分隔 MOT 风格，id = -1，conf 是第三方检测器置信度(0~100)
xml_det*/      每帧一个 VOC XML，``<name>`` 即轨迹 ID
=============  =====================================================

坐标系约定
----------
所有解析结果统一为 ``boxes = [x1, y1, x2, y2]``（float32），
``w = x2 - x1``、``h = y2 - y1``，与 MOT 的 (x, y, w, h) 完全自洽。
经核对：label.txt 第 1 帧 ``377 970 386 977`` 与 gt2.txt ``377 970 9 7``
数值一致，因此 **不存在 +1 的闭区间修正**，任何地方都不要擅自 +1。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from satdet.core.logging import get_logger

__all__ = [
    "Frames",
    "parse_label_txt",
    "parse_mot_txt",
    "parse_voc_dir",
    "load_annotations",
    "sniff_format",
]

log = get_logger("ann_parsers")

_NUM_RE = re.compile(r"(\d+)")


# =====================================================================
# 统一容器
# =====================================================================
@dataclass
class Frames:
    """一个序列的全部标注，按行平铺存储（列式，便于向量化）。

    Attributes
    ----------
    frame : (N,) int32
        帧号，从 1 开始。
    ids : (N,) int32
        轨迹 ID；-1 表示该标注没有轨迹号（纯检测数据）。
    boxes : (N, 4) float32
        ``[x1, y1, x2, y2]``，像素坐标。
    scores : (N,) float32
        置信度，归一化到 [0, 1]；GT 文件恒为 1。
    """

    frame: np.ndarray
    ids: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray

    # ---------- 构造 ----------
    @classmethod
    def empty(cls) -> "Frames":
        return cls(
            frame=np.zeros(0, np.int32),
            ids=np.zeros(0, np.int32),
            boxes=np.zeros((0, 4), np.float32),
            scores=np.zeros(0, np.float32),
        )

    @classmethod
    def from_rows(cls, rows: list[tuple[int, int, float, float, float, float, float]]) -> "Frames":
        """从 ``(frame, id, x1, y1, x2, y2, score)`` 列表构造。"""
        if not rows:
            return cls.empty()
        arr = np.asarray(rows, dtype=np.float64)
        return cls(
            frame=arr[:, 0].astype(np.int32),
            ids=arr[:, 1].astype(np.int32),
            boxes=arr[:, 2:6].astype(np.float32),
            scores=arr[:, 6].astype(np.float32),
        )

    # ---------- 基本属性 ----------
    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def num_frames(self) -> int:
        """标注里出现过的最大帧号（不是图片数量）。"""
        return int(self.frame.max()) if self.frame.size else 0

    @property
    def num_tracks(self) -> int:
        """有效轨迹数（忽略 id = -1）。"""
        if not self.has_ids:
            return 0
        return int(np.unique(self.ids[self.ids >= 0]).size)

    @property
    def has_ids(self) -> bool:
        return bool(self.ids.size and (self.ids >= 0).any())

    @property
    def wh(self) -> np.ndarray:
        """(N, 2) 宽高。"""
        return np.stack([self.boxes[:, 2] - self.boxes[:, 0],
                         self.boxes[:, 3] - self.boxes[:, 1]], axis=1)

    @property
    def centers(self) -> np.ndarray:
        """(N, 2) 中心点。"""
        return np.stack([(self.boxes[:, 0] + self.boxes[:, 2]) * 0.5,
                         (self.boxes[:, 1] + self.boxes[:, 3]) * 0.5], axis=1)

    # ---------- 增删查 ----------
    def subset(self, mask: np.ndarray) -> "Frames":
        return Frames(self.frame[mask], self.ids[mask], self.boxes[mask], self.scores[mask])

    def frame_mask(self, frame_id: int) -> np.ndarray:
        return self.frame == frame_id

    def frame_rows(self, frame_id: int) -> "Frames":
        return self.subset(self.frame_mask(frame_id))

    def sort(self) -> "Frames":
        """按 (frame, id) 排序，保证不同来源的读取顺序一致。"""
        order = np.lexsort((self.ids, self.frame))
        return Frames(self.frame[order], self.ids[order],
                      self.boxes[order], self.scores[order])

    def clip(self, width: int, height: int) -> "Frames":
        """把框裁进图像范围，丢弃裁剪后宽或高 <= 0 的框。"""
        b = self.boxes.copy()
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
        keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        return Frames(self.frame[keep], self.ids[keep], b[keep], self.scores[keep])

    @staticmethod
    def concat(parts: list["Frames"]) -> "Frames":
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return Frames.empty()
        return Frames(
            frame=np.concatenate([p.frame for p in parts]),
            ids=np.concatenate([p.ids for p in parts]),
            boxes=np.concatenate([p.boxes for p in parts]),
            scores=np.concatenate([p.scores for p in parts]),
        )

    def to_mot(self) -> np.ndarray:
        """导出 MOT 风格数组 ``(N, 7) = frame,id,x,y,w,h,score``。"""
        wh = self.wh
        return np.column_stack([self.frame, self.ids, self.boxes[:, 0], self.boxes[:, 1],
                                wh[:, 0], wh[:, 1], self.scores])


# =====================================================================
# 解析器
# =====================================================================
def sniff_format(path: str | Path) -> str:
    """嗅探标注格式，返回 ``"mot"`` / ``"label"`` / ``"voc"``。"""
    p = Path(path)
    if p.is_dir():
        return "voc"
    if p.suffix.lower() == ".xml":
        return "voc"
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return "mot" if "," in s else "label"
    return "label"


def _detect_mot_box_format(vals: list[list[float]]) -> str:
    """判断 MOT 文本的框列是 ``x, y, w, h`` 还是 ``x1, y1, x2, y2``。

    数据集里两种约定都真实存在（同一目录下不同序列混杂）：

    * ``001~027``：``1,1,582,24,6,9`` —— 逗号分隔，**x,y,w,h**；
    * ``028~038``：``1 1 247 47 251 56`` —— 空格分隔，**x1,y1,x2,y2**
      （与 ``sot/<seq>/gt/*.txt`` 的 ``247,47,4,9`` 对应：251-247=4、56-47=9）。

    判据：xyxy 里第 3 列恒大于第 1 列（x2 > x1）、第 4 列恒大于第 2 列（y2 > y1）；
    xywh 里只有紧贴左/上边界的目标才会出现这种关系。因此当 ≥90% 的行满足
    两轴都"后列 > 前列"时判为 xyxy。样本过少（<10 行）时保守按 xywh 处理。
    """
    if len(vals) < 10:
        return "xywh"
    # vals = [第3列, 第4列, 第5列, 第6列]
    # xyxy：x2(第5列) > x1(第3列)、y2(第6列) > y1(第4列)
    gt_x = sum(1 for v in vals if v[2] > v[0]) / len(vals)
    gt_y = sum(1 for v in vals if v[3] > v[1]) / len(vals)
    return "xyxy" if (gt_x >= 0.9 and gt_y >= 0.9) else "xywh"


def _parse_txt(path: Path, mode: str) -> Frames:
    """逐行解析文本标注，报错时给出行号与原始内容。"""
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    raw_rows: list[tuple[int, int, list[float], float]] = []   # (fid, tid, 4 值, score)
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [t for t in re.split(r"[,\s]+", s) if t]
            try:
                if len(parts) < 6:
                    raise ValueError("列数不足 6")
                fid = int(float(parts[0]))
                tid = int(float(parts[1]))
                vals = [float(v) for v in parts[2:6]]
                score = float(parts[6]) if len(parts) >= 7 else 1.0
                # det.txt 的置信度按 0~100 存，统一归一化到 0~1
                score = score / 100.0 if score > 1.0 else score
                if mode == "label":
                    x1, y1, x2, y2 = vals
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(f"非法框 w={x2 - x1} h={y2 - y1}")
                    rows.append((fid, tid, x1, y1, x2, y2, score))
                else:  # mot：格式稍后统一判定
                    raw_rows.append((fid, tid, vals, score))
            except (ValueError, IndexError) as exc:
                bad += 1
                if bad <= 5:
                    log.warning("%s:%d 跳过无法解析的行 (%s): %r", path.name, lineno, exc, s[:80])

    if mode == "mot" and raw_rows:
        fmt = _detect_mot_box_format([r[2] for r in raw_rows])
        for fid, tid, (a, b, c, d), score in raw_rows:
            if fmt == "xyxy":
                x1, y1, x2, y2 = a, b, c, d
            else:
                x1, y1, x2, y2 = a, b, a + c, b + d
            if x2 <= x1 or y2 <= y1:
                bad += 1
                continue
            rows.append((fid, tid, x1, y1, x2, y2, score))
        if fmt != "xywh":
            log.info("%s 判定为 x1,y1,x2,y2 格式（%d 行）", path.name, len(raw_rows))

    if bad:
        log.warning("%s 共跳过 %d 行非法标注", path.name, bad)
    return Frames.from_rows(rows)


def parse_label_txt(path: str | Path) -> Frames:
    """解析 ``label.txt``：空格分隔，``frame id x1 y1 x2 y2 ...``。"""
    return _parse_txt(Path(path), "label")


def parse_mot_txt(path: str | Path) -> Frames:
    """解析 MOT 风格 ``*.txt``：逗号分隔，``frame id x y w h conf ...``。

    适用于 ``gt2.txt`` / ``det.txt`` / ``mot/gt.txt``。
    空文件（例如 003/gt.txt 为 0 字节）返回空 Frames，不报错。
    """
    p = Path(path)
    if p.stat().st_size == 0:
        log.debug("%s 为空文件，返回空标注", p.name)
        return Frames.empty()
    return _parse_txt(p, "mot")


def parse_voc_dir(path: str | Path) -> tuple[Frames, tuple[int, int] | None]:
    """解析 VOC XML 目录（每帧一个 xml），返回 (Frames, (width, height))。

    帧号取文件名末尾的数字；``<name>`` 是数字时当作轨迹 ID，否则记 -1。
    """
    p = Path(path)
    files = sorted(p.glob("*.xml"))
    if not files:
        log.warning("目录中没有 xml: %s", p)
        return Frames.empty(), None

    rows: list[tuple[int, int, float, float, float, float, float]] = []
    size: tuple[int, int] | None = None
    for order, xf in enumerate(files, 1):
        try:
            root = ET.parse(xf).getroot()
        except ET.ParseError as exc:
            log.warning("XML 解析失败 %s: %s", xf.name, exc)
            continue

        if size is None:
            sz = root.find("size")
            if sz is not None and sz.findtext("width") and sz.findtext("height"):
                size = (int(float(sz.findtext("width"))), int(float(sz.findtext("height"))))

        m = _NUM_RE.search(xf.stem)
        fid = int(m.group(1)) if m else order

        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            name = (obj.findtext("name") or "").strip()
            tid = int(name) if name.isdigit() else -1
            rows.append((fid, tid, x1, y1, x2, y2, 1.0))

    return Frames.from_rows(rows), size


def load_annotations(path: str | Path) -> Frames:
    """自动嗅探格式并解析（不关心目录结构，只管单个标注源）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"标注不存在: {p}")
    fmt = sniff_format(p)
    if fmt == "voc":
        frames, _ = parse_voc_dir(p)
        return frames
    if fmt == "mot":
        return parse_mot_txt(p)
    return parse_label_txt(p)
