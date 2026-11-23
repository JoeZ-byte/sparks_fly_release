"""data —— 数据层：标注解析、序列索引、划分、tiling、统计。

对外主要入口
------------
viso_index.discover_sequences()   扫描数据集，得到 VisoIndex
viso_index.SequenceInfo.load_gt() 惰性加载某个序列的标注
split.load_split()                按配置划分 train/val/test
tiling.*                          纯几何函数（无 IO，可单测）
stats.compute_stats()             数据集统计
"""

from satdet.data.ann_parsers import Frames, load_annotations, parse_label_txt, parse_mot_txt
from satdet.data.viso_index import SequenceInfo, VisoIndex, discover_sequences

__all__ = [
    "Frames",
    "load_annotations",
    "parse_label_txt",
    "parse_mot_txt",
    "SequenceInfo",
    "VisoIndex",
    "discover_sequences",
]
