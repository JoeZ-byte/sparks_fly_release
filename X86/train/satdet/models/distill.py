"""知识蒸馏：用 512 大模型的特征教 320 小模型（FitNets 式特征模仿）。

为什么这么做
------------
板端为了速度把输入从 512 降到 320（算力 0.39×），代价是 mAP30 掉 14+ 个点。
Hinton 2015 的 KD 用在检测上的常见做法是"特征模仿"（Fine-grained Feature
Imitation, Wang CVPR2019）：让学生网络的中间特征去逼近教师网络的同层特征。

本实现只做特征层，不碰检测头：学生和教师是**同一套结构**（YOLOv8n+P2），
唯一的差别是输入分辨率，所以特征图通道数一致、只有空间尺寸不同——
学生特征直接插值到教师尺寸后做归一化 MSE 即可，不需要额外的投影层。

用法（configs/train/*.yaml）::

    distill:
      enabled: true
      teacher: weights/a_nwd_p2_v3_best.pt
      teacher_imgsz: 512
      weight: 0.5          # 蒸馏损失总权重
      proj: true           # 是否加 1x1 投影（默认 true）
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureDistiller"]


class FeatureDistiller(nn.Module):
    """教师特征 -> 学生特征 的归一化 MSE 蒸馏。

    Parameters
    ----------
    teacher : nn.Module
        已训练好的教师模型（同结构），内部会冻结并置为 eval。
    teacher_imgsz : int
        教师输入边长（学生见 tuple 的第一个维度）。
    weight : float
        蒸馏损失权重（相对检测损失）。
    proj : bool
        是否给每层加 1x1 投影（通道不变，只做特征重标定）。
    """

    def __init__(self, teacher: nn.Module, teacher_imgsz: int = 512,
                 weight: float = 0.5, proj: bool = True) -> None:
        super().__init__()
        self.teacher = teacher
        self.teacher_imgsz = int(teacher_imgsz)
        self.weight = float(weight)
        self.proj = bool(proj)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self._adapters: nn.ModuleList | None = None

    # ------------------------------------------------------------------
    def _ensure_adapters(self, feats: list[torch.Tensor]) -> None:
        if self._adapters is not None or not self.proj:
            return
        self._adapters = nn.ModuleList(
            [nn.Conv2d(f.shape[1], f.shape[1], 1) for f in feats]
        ).to(feats[0].device)
        # 初始化为近似恒等，避免一上来把学生的特征拉坏
        with torch.no_grad():
            for conv in self._adapters:
                conv.weight.copy_(torch.eye(conv.in_channels).view(
                    conv.out_channels, conv.in_channels, 1, 1))
                conv.bias.zero_()

    @torch.no_grad()
    def teacher_feats(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        """教师看"同一块图的高分辨率版本"，返回特征金字塔。"""
        big = F.interpolate(imgs, size=(self.teacher_imgsz, self.teacher_imgsz),
                            mode="bilinear", align_corners=False) if \
            imgs.shape[-1] != self.teacher_imgsz else imgs
        return self.teacher.forward_neck(big)

    # ------------------------------------------------------------------
    def forward(self, student_feats: list[torch.Tensor], imgs: torch.Tensor) -> torch.Tensor:
        """返回加权后的蒸馏损失（标量）。"""
        t_feats = self.teacher_feats(imgs)
        self._ensure_adapters(student_feats)
        total = 0.0
        n = 0
        for i, (sf, tf) in enumerate(zip(student_feats, t_feats)):
            if self.proj and self._adapters is not None:
                sf = self._adapters[i](sf)
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear",
                                   align_corners=False)
            # 归一化后比"方向"，避免只学幅度
            a = F.normalize(sf.flatten(2), dim=1)
            b = F.normalize(tf.detach().flatten(2), dim=1)
            total = total + ((a - b) ** 2).mean()
            n += 1
        loss = total / max(1, n)
        return self.weight * loss