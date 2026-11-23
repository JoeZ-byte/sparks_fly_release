"""统一日志：所有脚本输出同一格式，方便留档对比。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_logging", "get_logger", "banner"]

_FORMAT = "[%(asctime)s][%(levelname)-5s][%(name)-18s] %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(level: str = "INFO", logfile: str | Path | None = None) -> None:
    """初始化根 logger。

    Parameters
    ----------
    level : str
        日志级别，如 "INFO" / "DEBUG"。
    logfile : str | Path | None
        若给出，则同时写入文件（UTF-8 无 BOM）。
    """
    global _CONFIGURED
    root = logging.getLogger("satdet")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    if _CONFIGURED:
        return

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """取得子 logger，名称会自动加 ``satdet.`` 前缀。"""
    short = name.split(".")[-1] if name.startswith("satdet.") else name
    return logging.getLogger(f"satdet.{short}")


def banner(title: str, char: str = "=", width: int = 78, logger: logging.Logger | None = None) -> None:
    """打印分节标题，让长日志可读。"""
    log = logger or get_logger("banner")
    log.info(char * width)
    log.info(title)
    log.info(char * width)
