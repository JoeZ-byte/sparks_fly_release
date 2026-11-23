"""路径解析：项目中所有路径的唯一来源。

约定
----
1. 任何模块都不得硬编码绝对路径，一律通过本模块取得。
2. 环境变量可覆盖默认位置，便于队友把数据放在别的盘：
     SATDET_DATA_ROOT  数据集根目录（默认 <项目>/xinhuo_data/data）
     SATDET_ARTIFACTS  中间产物目录（默认 <项目>/artifacts）
     SATDET_WEIGHTS    权重目录（默认 <项目>/weights）
3. 注意：G: 盘是 exFAT，不支持硬链接与 POSIX 权限。所有写文件逻辑
   都用普通文件写入，不要依赖 os.link / os.symlink。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "dataset_root",
    "weights_dir",
    "artifacts_dir",
    "experiments_dir",
    "results_dir",
    "figures_dir",
    "docs_dir",
    "configs_dir",
    "ensure_dir",
    "relpath",
]


def project_root() -> Path:
    """项目根目录（satdet 包的上一级）。"""
    return Path(__file__).resolve().parents[2]


def _override(env_name: str, default: Path) -> Path:
    """环境变量优先，其次默认值。"""
    raw = os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else default


def dataset_root() -> Path:
    """VISO 数据集根目录，其下每个子目录是一个序列（002/003/...）。"""
    return _override("SATDET_DATA_ROOT", project_root() / "xinhuo_data" / "data")


def weights_dir() -> Path:
    """模型权重目录（*.pt / *.onnx / *.onnx.int8）。"""
    return _override("SATDET_WEIGHTS", project_root() / "weights")


def artifacts_dir() -> Path:
    """可重建的中间产物：序列索引、划分、tile 清单、缓存。"""
    return _override("SATDET_ARTIFACTS", project_root() / "artifacts")


def experiments_dir() -> Path:
    """实验目录：runs/ 下放训练日志，results/ 放结论级 CSV，figures/ 放图。"""
    return project_root() / "experiments"


def results_dir() -> Path:
    """结论级产物（入版本库，与报告一一对应）。"""
    return experiments_dir() / "results"


def figures_dir() -> Path:
    """报告用图（入版本库）。"""
    return experiments_dir() / "figures"


def docs_dir() -> Path:
    return project_root() / "docs"


def configs_dir() -> Path:
    return project_root() / "configs"


def ensure_dir(path: str | Path) -> Path:
    """创建目录（含父目录）并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relpath(path: str | Path) -> str:
    """转成相对项目根目录的短路径，便于日志与报告里显示。"""
    try:
        return str(Path(path).resolve().relative_to(project_root()))
    except ValueError:
        return str(path)
