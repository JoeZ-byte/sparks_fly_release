"""measure —— 车速标定与换算（M6）。

* ``speed.py``  像素速度 -> km/h 或 px/s，含滑动中值与置信度
* ``scale.py``  米/像素标定（车道宽/GSD/地物），**待 M6 实现**

设计见 04_开发轮次.md。
"""

from satdet.measure.speed import SpeedConfig, SpeedEstimator, estimate_speed

__all__ = ["SpeedConfig", "SpeedEstimator", "estimate_speed"]
