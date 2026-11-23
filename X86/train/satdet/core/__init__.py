"""core —— 基础设施层。

对上层暴露最小接口，避免各处重复实现路径/日志/配置逻辑。
"""

from satdet.core.logging import banner, get_logger, setup_logging
from satdet.core.paths import (
    artifacts_dir,
    dataset_root,
    ensure_dir,
    experiments_dir,
    figures_dir,
    project_root,
    results_dir,
    weights_dir,
)

__all__ = [
    "banner",
    "get_logger",
    "setup_logging",
    "artifacts_dir",
    "dataset_root",
    "ensure_dir",
    "experiments_dir",
    "figures_dir",
    "project_root",
    "results_dir",
    "weights_dir",
]
