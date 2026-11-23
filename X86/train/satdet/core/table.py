"""控制台表格：按**显示宽度**对齐（中文占两格），避免日志里表格错位。"""

from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence

__all__ = ["display_width", "pad", "format_table"]

_WIDE = {"W", "F"}


def display_width(text: str) -> int:
    """字符串在等宽终端里占的列数。"""
    w = 0
    for ch in str(text):
        w += 2 if unicodedata.east_asian_width(ch) in _WIDE else 1
    return w


def pad(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格。"""
    s = str(text)
    gap = max(0, width - display_width(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def format_table(
    rows: Sequence[dict],
    columns: Sequence[tuple[str, str]] | None = None,
    right_align: Iterable[str] = (),
) -> str:
    """把 dict 列表渲染成对齐表格。

    Parameters
    ----------
    columns : [(key, 表头)]；None 表示用第一行(或全部行的并集)的键。
    right_align : 需要右对齐的列（数字列）。
    """
    if not rows:
        return "(无数据)"
    if columns is None:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        columns = [(k, k) for k in keys]

    right = set(right_align)
    headers = [label for _, label in columns]
    body = [[str(r.get(key, "")) for key, _ in columns] for r in rows]

    widths = [display_width(h) for h in headers]
    for line in body:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], display_width(cell))

    def render(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            align = "right" if (columns[i][0] in right) else "left"
            parts.append(pad(cell, widths[i], align))
        return "  ".join(parts).rstrip()

    sep = "-" * min(120, sum(widths) + 2 * (len(widths) - 1))
    out = [render(headers), sep]
    out.extend(render(line) for line in body)
    return "\n".join(out)
