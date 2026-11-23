"""Copy-Paste 增强：把训练集里的小目标贴到别的图块上。

动机
----
VISO 可用训练实例只有 8.7 万（4 段），模型第 2 个 epoch 后就开始过拟合。
Simple Copy-Paste（Ghiasi et al. CVPR 2021）对小目标检测是公认有效的增广：
把真实车辆图块随机贴到其他图块的路面/背景上，等效扩充实例数、
并让模型学会"在任意背景下认出车"。图块本身来自真实数据，
所以没有合成伪影。

实现要点
--------
* 图块库在训练集上**离线提取一次**并缓存到 ``artifacts/aug/patch_bank.npz``；
* 增强发生在 512×512 训练图上（图块 256 放大 2 倍之后），
  贴片按同一 2 倍比例放大后随机摆放；
* 与已有框 IoU > 0.3 的位置跳过（避免在同一处叠出"分身"）；
* 贴片亮度做 ±10% 抖动，防止整张图出现重复亮度指纹。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from satdet.core.logging import get_logger
from satdet.core.paths import artifacts_dir

__all__ = ["PatchBank"]

log = get_logger("copy_paste")


class PatchBank:
    """训练目标图块库（原图分辨率坐标）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._patches: list[np.ndarray] = []
        self._boxes: list[np.ndarray] = []        # 贴片内目标框 (x1,y1,x2,y2)，贴片局部坐标
        if path is not None and Path(path).is_file():
            self.load(path)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._patches)

    def build(self, index, seqs, max_patches: int = 3000, pad: int = 2,
              min_side: int = 3, rng=None) -> "PatchBank":
        """从训练序列的 GT 框切图块。"""
        rng = rng or np.random.default_rng(0)
        cand = []
        for seq in seqs:
            info = index[seq]
            gt = info.load_gt()
            for (f, x1, y1, x2, y2) in zip(gt.frame, gt.boxes[:, 0], gt.boxes[:, 1],
                                           gt.boxes[:, 2], gt.boxes[:, 3]):
                if x2 - x1 < min_side or y2 - y1 < min_side:
                    continue
                cand.append((seq, int(f), int(x1), int(y1), int(x2), int(y2)))
        if len(cand) > max_patches:
            idx_sel = rng.choice(len(cand), max_patches, replace=False)
            cand = [cand[i] for i in idx_sel]

        for seq, f, x1, y1, x2, y2 in cand:
            info = index[seq]
            img = cv2.imread(str(info.image_path(f)), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
            if px2 - px1 < 2 or py2 - py1 < 2:
                continue
            patch = img[py1:py2, px1:px2]
            self._patches.append(np.ascontiguousarray(patch))
            self._boxes.append(np.asarray([[x1 - px1, y1 - py1, x2 - px1, y2 - py1]],
                                          np.float32))
        log.info("Copy-Paste 图块库: %d 个（来自 %s）", len(self._patches), ",".join(seqs))
        return self

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, patches=np.asarray(self._patches, dtype=object),
                            boxes=np.asarray(self._boxes, dtype=object))
        log.info("图块库已缓存: %s", p)

    def load(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=True) as z:
            self._patches = list(z["patches"])
            self._boxes = list(z["boxes"])
        log.info("图块库已加载: %d 个", len(self._patches))

    # ------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
        """随机取一个 (贴片, 局部框)。"""
        if not self._patches:
            return None
        k = int(rng.integers(len(self._patches)))
        return self._patches[k], self._boxes[k].copy()

    @staticmethod
    def default_path() -> Path:
        return artifacts_dir() / "aug" / "patch_bank.npz"
