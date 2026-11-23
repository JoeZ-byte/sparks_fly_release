"""VISO 数据集索引：扫描目录、定位图像/标注/视频，暴露统一接口。

目录结构（实测，9 个序列略有差异）
--------------------------------------------------------------------
<dataset_root>/<seq>/
    img1/ 或 img/       帧图片，命名 000001.jpg（6 位，从 1 开始）
    mot/gt.txt          带轨迹 ID 的 MOT 标注（最优先）
    gt2.txt             清洗后的 MOT 标注（无 ID）
    label.txt           空格分隔标注（有 ID，但含少量未清洗框）
    det.txt             第三方检测器输出（可作"别人家基线"对照）
    xml_det*/           少数序列只有 VOC XML
    *.avi               少量序列附带原始视频，用于流式 demo
    fd/                 数据集自带的帧差图（预处理过，仅参考）
    crop/  img_gt/  sot/  Marked/  ...  其它用途目录，本项目不用

标注优先级
----------
``mot/gt.txt`` > ``gt2.txt`` > ``label.txt`` > VOC XML 目录。
理由：MOT 版带真实轨迹号，且已被数据集作者清洗（实测 003 少 24 个、
008 少 42 个脏框），对训练更可靠。详见 02_数据说明.md。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from satdet.core.logging import get_logger
from satdet.core.paths import dataset_root
from satdet.data.ann_parsers import Frames, parse_label_txt, parse_mot_txt, parse_voc_dir

__all__ = ["SequenceInfo", "VisoIndex", "discover_sequences", "GT_CANDIDATES"]

log = get_logger("viso_index")

#: 标注候选（按优先级从高到低）
GT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("mot/gt.txt", "mot"),
    ("gt/gt.txt", "mot"),      # mot/car/<seq>/gt/gt.txt（全量 38 段语料布局）
    ("gt2.txt", "mot"),
    ("label.txt", "label"),
)

#: 只含 VOC XML 的目录候选
XML_DIR_CANDIDATES: tuple[str, ...] = ("xml_det00", "xml_det01", "xml_det", "xml_mot")

#: 图像目录候选（按优先级）
IMG_DIR_CANDIDATES: tuple[str, ...] = ("img1", "img", "img_dn", "img0", "img1_01")

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =====================================================================
# 单个序列
# =====================================================================
@dataclass
class SequenceInfo:
    """一个序列的元信息。标注用 :meth:`load_gt` 惰性加载并缓存。"""

    seq_id: str
    root: Path
    image_dir: Path | None
    frames: list[Path] = field(default_factory=list)
    width: int = 0
    height: int = 0
    gt_path: Path | None = None
    gt_format: str = ""          # "mot" / "label" / "voc" / ""
    video: Path | None = None
    extra: dict = field(default_factory=dict)
    _gt_cache: Frames | None = field(default=None, repr=False, compare=False)

    # ---------- 惰性标注 ----------
    def load_gt(self) -> Frames:
        """加载并缓存标注；无标注时返回空 Frames。"""
        if self._gt_cache is not None:
            return self._gt_cache
        if self.gt_path is None:
            self._gt_cache = Frames.empty()
            return self._gt_cache
        if self.gt_format == "voc":
            frames, _ = parse_voc_dir(self.gt_path)
        elif self.gt_format == "label":
            frames = parse_label_txt(self.gt_path)
        else:
            frames = parse_mot_txt(self.gt_path)
        self._gt_cache = frames
        return frames

    # ---------- 派生属性 ----------
    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def has_gt(self) -> bool:
        return self.gt_path is not None and len(self.load_gt()) > 0

    @property
    def has_ids(self) -> bool:
        return self.load_gt().has_ids

    @property
    def eval_frame_limit(self) -> int:
        """可评测的最大帧号 = min(图片帧数, 标注最大帧号)。

        早期外部视频 有 700 张图但只标了前 299 帧（见 docs/02）。评测/训练时必须用
        这个上限，否则未标注的帧会被当成"负样本"，召回率被稀释。
        """
        gt = self.load_gt()
        if len(gt) == 0:
            return 0
        return int(min(self.num_frames, gt.num_frames))

    def image_path(self, frame_id: int) -> Path:
        """取第 ``frame_id`` 帧图片（1-based）。"""
        if not 1 <= frame_id <= self.num_frames:
            raise IndexError(f"{self.seq_id}: 帧号 {frame_id} 超出范围 1..{self.num_frames}")
        return self.frames[frame_id - 1]

    def summary(self) -> dict:
        gt = self.load_gt()
        return {
            "seq": self.seq_id,
            "frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "instances": len(gt),
            "tracks": gt.num_tracks,
            "gt_source": self.gt_path.name if self.gt_path else "-",
            "gt_format": self.gt_format or "-",
            "has_ids": bool(gt.has_ids),
            "video": self.video.name if self.video else "-",
        }

    def to_dict(self) -> dict:
        """序列化（不含标注内容，标注由 gt_path 重建）。"""
        return {
            "seq_id": self.seq_id,
            "root": str(self.root),
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "num_frames": len(self.frames),
            "width": self.width,
            "height": self.height,
            "gt_path": str(self.gt_path) if self.gt_path else None,
            "gt_format": self.gt_format,
            "video": str(self.video) if self.video else None,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceInfo":
        image_dir = Path(d["image_dir"]) if d.get("image_dir") else None
        frames: list[Path] = []
        if image_dir and image_dir.is_dir():
            frames = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)
        return cls(
            seq_id=d["seq_id"],
            root=Path(d["root"]),
            image_dir=image_dir,
            frames=frames,
            width=int(d.get("width") or 0),
            height=int(d.get("height") or 0),
            gt_path=Path(d["gt_path"]) if d.get("gt_path") else None,
            gt_format=d.get("gt_format") or "",
            video=Path(d["video"]) if d.get("video") else None,
            extra=d.get("extra") or {},
        )


# =====================================================================
# 数据集索引
# =====================================================================
@dataclass
class VisoIndex:
    """全部序列的索引。``order`` 保持字母序，避免不同机器遍历顺序不一致。"""

    sequences: dict[str, SequenceInfo] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: dataset_root())

    # ---------- 访问 ----------
    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterable[SequenceInfo]:
        return (self.sequences[k] for k in self.order)

    def __getitem__(self, seq_id: str) -> SequenceInfo:
        return self.sequences[seq_id]

    @property
    def order(self) -> list[str]:
        return sorted(self.sequences)

    def pick(self, seq_ids: Iterable[str]) -> list[SequenceInfo]:
        """按给定顺序取序列，缺一个就报错（避免静默少数据）。"""
        out = []
        for s in seq_ids:
            if s not in self.sequences:
                raise KeyError(f"序列 {s} 不在索引中，可选: {self.order}")
            out.append(self.sequences[s])
        return out

    def filter(self, seq_ids: Iterable[str]) -> "VisoIndex":
        keep = [s for s in seq_ids if s in self.sequences]
        return VisoIndex({s: self.sequences[s] for s in keep}, self.root)

    # ---------- 统计 ----------
    def summary_rows(self) -> list[dict]:
        return [self.sequences[k].summary() for k in self.order]

    def total_frames(self) -> int:
        return sum(s.num_frames for s in self)

    def total_instances(self) -> int:
        return sum(len(s.load_gt()) for s in self)

    # ---------- 诊断 ----------
    def diagnostics(self) -> list[str]:
        """返回潜在问题清单，供 ``satdet doctor`` / stats 打印。"""
        issues: list[str] = []
        for s in self:
            if not s.has_gt:
                issues.append(f"[{s.seq_id}] 无标注文件，只能用于推理/无监督实验")
                continue
            gt = s.load_gt()
            if gt.num_frames > s.num_frames:
                issues.append(f"[{s.seq_id}] 标注帧数({gt.num_frames}) > 图片帧数({s.num_frames})")
            if gt.num_frames < s.num_frames:
                issues.append(f"[{s.seq_id}] 标注只覆盖 {gt.num_frames}/{s.num_frames} 帧")
            if not gt.has_ids:
                issues.append(f"[{s.seq_id}] 标注无轨迹 ID，仅可用于检测任务")
            if not (s.width, s.height) == (1024, 1024):
                issues.append(f"[{s.seq_id}] 分辨率 {s.width}x{s.height} 与主流序列 1024x1024 不同")
        return issues

    # ---------- 持久化 ----------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "sequences": {k: self.sequences[k].to_dict() for k in self.order},
        }
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        log.info("索引已写入 %s (%d 个序列)", p, len(self))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "VisoIndex":
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        seqs = {k: SequenceInfo.from_dict(v) for k, v in payload.get("sequences", {}).items()}
        return cls(seqs, Path(payload.get("root", dataset_root())))


# =====================================================================
# 扫描
# =====================================================================
def _pick_image_dir(seq_dir: Path) -> Path | None:
    """挑出可用的图像目录（优先 img1）。"""
    for name in IMG_DIR_CANDIDATES:
        d = seq_dir / name
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                    return d
    # 兜底：序列根目录下直接放图片
    for p in seq_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
            return seq_dir
    return None


def _pick_gt(seq_dir: Path) -> tuple[Path | None, str]:
    """按优先级挑出标注源。"""
    for rel, fmt in GT_CANDIDATES:
        p = seq_dir / rel
        if p.is_file() and p.stat().st_size > 0:
            return p, fmt
    for name in XML_DIR_CANDIDATES:
        d = seq_dir / name
        if d.is_dir() and any(d.glob("*.xml")):
            return d, "voc"
    return None, ""


def _pick_video(seq_dir: Path) -> Path | None:
    vids = sorted([p for p in seq_dir.glob("*.avi")] + [p for p in seq_dir.glob("*.mp4")])
    return vids[0] if vids else None


def _read_resolution(seq_dir: Path, img_dir: Path | None, frames: list[Path]) -> tuple[int, int]:
    """读首帧分辨率；PIL 不可用时退回读取旧的元信息文件。"""
    if frames:
        try:
            from PIL import Image
            with Image.open(frames[0]) as im:
                return int(im.width), int(im.height)
        except Exception as exc:  # noqa: BLE001 - 只在诊断里提示
            log.warning("读取分辨率失败 %s: %s", frames[0], exc)
    return 0, 0


def discover_sequences(
    root: str | Path | None = None,
    seq_ids: Iterable[str] | None = None,
) -> VisoIndex:
    """扫描数据集根目录，返回 :class:`VisoIndex`。

    Parameters
    ----------
    root : 默认取 ``SATDET_DATA_ROOT`` 或 ``<项目>/xinhuo_data/data``。
    seq_ids : 只扫描指定序列；None 表示全部。
    """
    base = Path(root) if root is not None else dataset_root()
    if not base.is_dir():
        raise FileNotFoundError(
            f"数据集根目录不存在: {base}\n"
            f"请设置环境变量 SATDET_DATA_ROOT 指向 VISO 的 data 目录。"
        )

    wanted = set(seq_ids) if seq_ids is not None else None
    out: dict[str, SequenceInfo] = {}

    for seq_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        sid = seq_dir.name
        if wanted is not None and sid not in wanted:
            continue
        img_dir = _pick_image_dir(seq_dir)
        if img_dir is None:
            log.debug("跳过 %s：没有图像目录", sid)
            continue

        frames = sorted(p for p in img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES)
        if not frames:
            log.debug("跳过 %s：图像目录为空", sid)
            continue

        gt_path, gt_fmt = _pick_gt(seq_dir)
        w, h = _read_resolution(seq_dir, img_dir, frames)
        info = SequenceInfo(
            seq_id=sid,
            root=seq_dir,
            image_dir=img_dir,
            frames=frames,
            width=w,
            height=h,
            gt_path=gt_path,
            gt_format=gt_fmt,
            video=_pick_video(seq_dir),
        )
        out[sid] = info
        log.info("%-4s 帧=%-4d 分辨率=%dx%d 标注=%-12s", sid, len(frames), w, h,
                 gt_path.name if gt_path else "无")

    if not out:
        raise RuntimeError(f"在 {base} 下没有发现任何可用序列")

    return VisoIndex(out, base)
