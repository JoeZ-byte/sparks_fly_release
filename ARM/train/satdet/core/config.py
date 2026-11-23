"""配置读取：所有可调参数放 configs/*.yaml，代码里不写魔法数。

用法
----
>>> from satdet.core.config import load_config
>>> cfg = load_config("dataset/viso.yaml")          # 按相对 configs/ 的路径
>>> cfg = load_config("train/baseline.yaml")        # 支持继承
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from satdet.core.paths import configs_dir

__all__ = ["load_yaml", "deep_merge", "load_config", "dump_yaml"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件；文件不存在直接报错（不要静默用默认值）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置文件根节点必须是映射(dict): {p}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个 dict，override 优先；不修改入参。"""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(rel_path: str | Path, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """读取配置并处理 ``_base_`` 继承链。

    ``_base: train/baseline.yaml`` 表示先加载该文件，再用本文件覆盖。
    """
    p = Path(rel_path)
    if not p.is_file():
        p = configs_dir() / Path(rel_path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到配置: {rel_path} (configs 目录: {configs_dir()})")

    key = str(p.resolve())
    if key in _seen:
        raise RecursionError(f"配置继承出现环: {' -> '.join((*_seen, key))}")

    cfg = load_yaml(p)
    parent = cfg.pop("_base_", None)
    if parent is None:
        cfg["_source_"] = str(p)
        return cfg

    base = load_config(parent, (*_seen, key))
    merged = deep_merge(base, cfg)
    merged["_source_"] = str(p)
    return merged


def dump_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """把实验最终配置落盘，供复现（allow_unicode=True，中文注释不转义）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return p
