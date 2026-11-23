"""motion —— L1 运动先验层 + L2 时序跟踪层。

已完成（M4 前半）
-----------------
* ``bgmodel.py``   滑动平均 / 中值背景、自适应阈值前景掩膜、连通域过滤
* ``proposal.py``  运动提案器（运动框 ∪ 跟踪预测框 -> 网格对齐图块）

待实现（M4 后半）
-----------------
* ``road_mask.py`` 道路掩膜（低频更新）
* ``tracker.py``   卡尔曼 + 匈牙利关联，输出 ID / 速度状态
"""

from satdet.motion.bgmodel import (
    MedianBackground,
    RunningAverageBackground,
    foreground_mask,
    mask_to_boxes,
)
from satdet.motion.proposal import MotionConfig, MotionProposer

__all__ = [
    "MedianBackground",
    "RunningAverageBackground",
    "foreground_mask",
    "mask_to_boxes",
    "MotionConfig",
    "MotionProposer",
]
