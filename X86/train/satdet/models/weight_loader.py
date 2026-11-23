"""加载官方 COCO 权重到自研模型，并给出**可审计的加载报告**。

设计要点
--------
* 权重文件是纯 state_dict（由 ``scripts/01_convert_weights.py`` 生成），
  运行时不依赖 ultralytics。
* 官方的 ``model.22.*``（Detect）在本项目模型里叫 ``detect.*``，
  由 ``YOLOv8.load_state_dict`` 统一重映射。
* **方框分支（detect.cv2.*）原样复用**（框回归与类别无关，可迁移），
  但**分类分支（detect.cv3.*）因类别数从 80 变 1 而形状不匹配，必须重新初始化**。
  这一点必须在日志里明确写出来，避免误以为"全都加载成功了"。
* P2 分支（``p2_branch.*`` / ``detect.cv2p2 / cv3p2``）全新随机初始化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from satdet.core.logging import get_logger

__all__ = ["LoadReport", "load_coco_weights", "freeze_backbone", "count_parameters"]

log = get_logger("weight_loader")


@dataclass
class LoadReport:
    """一次权重加载的可审计结果。"""

    path: str = ""
    loaded: int = 0                                   # 成功灌入的张量数
    reinit_cls_head: list[str] = field(default_factory=list)   # 因类别数不同而重建
    random_init: list[str] = field(default_factory=list)       # 全新模块（如 P2 分支）
    missing: list[str] = field(default_factory=list)  # 模型有、权重没有
    unexpected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """除"预期内的重建/新模块"外，不应有缺失。"""
        return not self.missing and not self.unexpected

    def summary(self) -> str:
        lines = [
            f"权重: {self.path}",
            f"  已加载张量     : {self.loaded}",
            f"  分类头重建     : {len(self.reinit_cls_head)} (nc 80 -> 本项目类别数)",
            f"  随机初始化     : {len(self.random_init)} (P2 分支 / 新增层)",
            f"  模型有而权重缺 : {len(self.missing)}",
            f"  权重有而模型无 : {len(self.unexpected)}",
        ]
        return "\n".join(lines)


def _is_cls_head(key: str) -> bool:
    return key.startswith("detect.cv3.") or key.startswith("detect.cv3p2.")


def _is_p2_new(key: str) -> bool:
    return key.startswith("p2_branch.") or key.startswith("detect.cv2p2.") \
        or key.startswith("detect.cv3p2.")


def load_coco_weights(model: nn.Module, path: str | Path, strict: bool = True) -> LoadReport:
    """把纯 state_dict 权重加载进模型。

    Parameters
    ----------
    strict : bool
        为 True 时要求"除预期内的重建/新模块外"全部命中，否则抛异常。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"权重不存在: {p}\n先运行: python scripts/01_convert_weights.py"
        )

    state = torch.load(p, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"权重不是 state_dict: {type(state)}")
    # 训练检查点里包了一层 {"model": state_dict, "epoch":..., "mAP30":...}
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model_keys = set(model.state_dict().keys())
    src_keys = set(state.keys())

    # 官方 Detect 前缀 model.22. -> detect.
    norm = {(("detect." + k[len("model.22."):]) if k.startswith("model.22.") else k): v
            for k, v in state.items()}

    report = LoadReport(path=str(p))
    to_load: dict[str, torch.Tensor] = {}

    for key, tensor in norm.items():
        if key not in model_keys:
            report.unexpected.append(key)
            continue
        target_shape = model.state_dict()[key].shape
        if tuple(target_shape) != tuple(tensor.shape):
            if _is_cls_head(key):
                report.reinit_cls_head.append(key)
            else:
                raise RuntimeError(
                    f"张量形状不匹配且不属于预期的分类头: {key} "
                    f"权重 {tuple(tensor.shape)} vs 模型 {tuple(target_shape)}"
                )
            continue
        to_load[key] = tensor

    missing = sorted(model_keys - set(to_load.keys()))
    for key in missing:
        if _is_cls_head(key) or _is_p2_new(key):
            report.random_init.append(key)
        else:
            report.missing.append(key)

    model.load_state_dict(to_load, strict=False)
    report.loaded = len(to_load)

    log.info("\n%s", report.summary())
    if strict and not report.ok:
        raise RuntimeError(
            "权重加载存在非预期缺失/多余键，请检查模型结构与转换脚本是否匹配"
        )
    return report


def freeze_backbone(model: nn.Module, upto: int = 9) -> int:
    """冻结 ``model.model[0..upto]``（默认 0..9，即主干 + SPPF）。

    只训 Neck/Head 的理由：
    1. COCO 主干特征对"亮斑/边缘"已足够，数据集只有 2255 帧，全量微调易过拟合；
    2. 冻结后主干可以做 INT8 量化而不掉点（权重不再变动，量化误差可控）；
    3. 训练时间缩短，单卡数小时可收敛。

    Returns
    -------
    int : 被冻结的参数张量个数。
    """
    seq = getattr(model, "model", None)
    if seq is None:
        raise AttributeError("模型没有 .model 顺序容器，无法按层冻结")

    frozen = 0
    for i, layer in enumerate(seq):
        if i > upto:
            break
        for param in layer.parameters():
            param.requires_grad = False
            frozen += 1
    log.info("冻结 backbone: model[0..%d]，共 %d 个参数张量", upto, frozen)
    return frozen


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """参数量统计（写报告用）。"""
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


def load_meta(path: str | Path) -> dict:
    """读取转换脚本产出的 meta（类别名、stride 等）。"""
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
