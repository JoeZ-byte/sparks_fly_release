"""数据集划分：**按序列划分**，杜绝同场景泄漏。

为什么按序列而不是按帧随机划分
------------------------------
同一序列的相邻帧几乎相同（帧间位移仅 2~3 px），若随机抽 10% 当验证集，
模型等于"见过"验证帧的邻居，mAP 会虚高。卫星视频每一段对应不同地理
区域/光照/路网，只有按序列划分才能反映真实泛化能力。

角色约定
--------
train    训练
val      选模型/调阈值（早停、NWD 权重选择都用它）
test     只跑一次，用于报告最终数字
holdout  不参与任何调优：009 目标尺度差异大，
         用于"跨场景/跨分辨率泛化"章节
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from satdet.core.logging import get_logger
from satdet.data.viso_index import VisoIndex

__all__ = [
    "SplitConfig",
    "load_split",
    "validate_split",
    "split_report",
    "save_split",
    "scale_diversity",
]

log = get_logger("split")

ROLES = ("train", "val", "test", "holdout")


@dataclass
class SplitConfig:
    """四个角色的序列清单。"""

    name: str = "viso"
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    holdout: list[str] = field(default_factory=list)

    def role_of(self, seq_id: str) -> str | None:
        for role in ROLES:
            if seq_id in getattr(self, role):
                return role
        return None

    def all_seq_ids(self) -> list[str]:
        return [s for role in ROLES for s in getattr(self, role)]

    def as_dict(self) -> dict:
        return {"name": self.name, **{role: list(getattr(self, role)) for role in ROLES}}


def _seq_id(value: object) -> str:
    """把 YAML 里的序列号统一成 3 位字符串。

    YAML 会把未加引号的 ``002`` 解析成整数 2（历史踩坑，见 docs/09 的"易错点"），
    这里兜底补零，避免配置里少写一对引号就整段数据消失。
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:03d}"
    return str(value).strip()


def load_split(cfg: dict) -> SplitConfig:
    """从 dataset 配置（已含 ``split`` 段）构造。"""
    raw = cfg.get("split") or {}
    missing = [r for r in ("train", "val") if not raw.get(r)]
    if missing:
        raise ValueError(f"配置缺少 split.{'/'.join(missing)}，无法训练/验证")
    return SplitConfig(
        name=str(cfg.get("name", "viso")),
        train=[_seq_id(s) for s in raw.get("train", [])],
        val=[_seq_id(s) for s in raw.get("val", [])],
        test=[_seq_id(s) for s in raw.get("test", [])],
        holdout=[_seq_id(s) for s in raw.get("holdout", [])],
    )


def validate_split(index: VisoIndex, split: SplitConfig) -> list[str]:
    """返回问题清单；空列表表示划分合法。"""
    problems: list[str] = []

    seen: dict[str, str] = {}
    for role in ROLES:
        for seq_id in getattr(split, role):
            if seq_id in seen:
                problems.append(f"序列 {seq_id} 同时出现在 {seen[seq_id]} 与 {role}")
            seen[seq_id] = role
            if seq_id not in index.sequences:
                problems.append(f"序列 {seq_id}（{role}）在数据集中不存在，可选: {index.order}")

    unassigned = [s for s in index.order if s not in seen]
    if unassigned:
        problems.append(f"以下序列未分配角色（不会被使用）: {', '.join(unassigned)}")

    for role in ("train", "val"):
        if not getattr(split, role):
            problems.append(f"{role} 为空")

    return problems


def scale_diversity(index: VisoIndex, split: SplitConfig) -> list[dict]:
    """各序列目标尺度表，用于判断训练集是否尺度单一。

    单列出来是因为它需要真实标注（``Frames.wh``），而 :func:`validate_split`
    只做结构校验、不碰数据，方便单测与在无数据环境下跑通。
    """
    rows: list[dict] = []
    for role in ROLES:
        for seq_id in getattr(split, role):
            if seq_id not in index.sequences:
                continue
            gt = index[seq_id].load_gt()
            if len(gt) == 0:
                continue
            wh = gt.wh
            area = wh[:, 0] * wh[:, 1]
            rows.append({
                "role": role,
                "seq": seq_id,
                "side_med": round(float(np.median(np.sqrt(area))), 2),
                "side_mean": round(float(np.sqrt(area).mean()), 2),
                "side_p95": round(float(np.percentile(np.sqrt(area), 95)), 2),
            })
    return rows


def split_report(index: VisoIndex, split: SplitConfig) -> list[dict]:
    """生成每个角色/序列的规模表，用于文档与日志。"""
    rows: list[dict] = []
    for role in ROLES:
        for seq_id in getattr(split, role):
            if seq_id not in index.sequences:
                rows.append({"role": role, "seq": seq_id, "frames": 0, "instances": 0, "tracks": 0})
                continue
            info = index[seq_id]
            gt = info.load_gt()
            rows.append({
                "role": role,
                "seq": seq_id,
                "frames": info.num_frames,
                "instances": len(gt),
                "tracks": gt.num_tracks,
                "resolution": f"{info.width}x{info.height}",
                "gt": info.gt_path.name if info.gt_path else "-",
            })
    return rows


def role_totals(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        agg = out.setdefault(str(r["role"]), {"seqs": 0, "frames": 0, "instances": 0})
        agg["seqs"] += 1
        agg["frames"] += int(r.get("frames") or 0)
        agg["instances"] += int(r.get("instances") or 0)
    return out


def save_split(split: SplitConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(split.as_dict(), fh, ensure_ascii=False, indent=2)
    log.info("划分已写入 %s", p)
    return p


def load_split_file(path: str | Path) -> SplitConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    return SplitConfig(
        name=d.get("name", "viso"),
        train=list(d.get("train", [])),
        val=list(d.get("val", [])),
        test=list(d.get("test", [])),
        holdout=list(d.get("holdout", [])),
    )
