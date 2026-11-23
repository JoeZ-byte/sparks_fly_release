"""models

L4 网络精检层的自研实现（不依赖 ultralytics）。

**当前状态：尚未实现（M2/M3 阶段实现）**

规划中的模块：

- `blocks.py`       Conv/Bottleneck/C2f/SPPF，命名与官方权重 1:1 对齐
- `yolov8.py`       YOLOv8n 主干 + 检测头（可选 P2）
- `p2_head.py`      P2(stride 4) 浅层检测头，专治 5~9 px 目标
- `nwd_loss.py`     归一化高斯 Wasserstein 距离 Loss
- `weight_loader.py` 把官方 yolov8n.pt 的 state_dict 映射进自研模型并校验

详见 01_工作目录架构.md 的"模块职责与状态"表。
"""

__all__: list[str] = []
