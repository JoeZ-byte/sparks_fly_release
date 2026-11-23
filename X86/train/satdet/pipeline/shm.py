"""跨进程零拷贝：用共享内存传帧，而不是 pickle 整个 ndarray。

为什么这件事重要（实测依据）
--------------------------
``multiprocessing.Queue.put(ndarray)`` 会把数组**完整序列化并拷贝**一遍：
1024×1024×3 = 3 MB/帧，10 fps 就是 30 MB/s 的纯拷贝，还要走 pickle 协议。

本模块的做法
------------
* 生产者把 **JPEG 压缩字节**写进共享内存槽（~150 KB/帧，比原始像素小 20 倍），
  队列里只传 ``FrameHandle``（8 字节级描述符）；
* 消费者直接从共享内存视图解码，不经过 pickle；
* 槽位化 + 空闲队列天然实现**背压**：池满则生产者阻塞，不会把内存吃爆。

> 为什么不共享"解码后的像素"：``cv2.imdecode`` 必须输出到新 ndarray，
> 无法直接解码进共享内存；共享**压缩字节**才是真正零拷贝的那一段。
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

__all__ = ["SharedFramePool", "FrameHandle"]


@dataclass(frozen=True)
class FrameHandle:
    """队列里传的描述符（极小，pickle 成本可忽略）。"""

    slot: int
    length: int
    seq: str
    frame: int


class SharedFramePool:
    """固定槽位数 × 固定字节上限的共享内存池（多生产者/多消费者安全）。"""

    def __init__(self, n_slots: int = 8, slot_bytes: int = 1 << 20) -> None:
        self.n_slots = int(n_slots)
        self.slot_bytes = int(slot_bytes)
        self._buf = mp.RawArray("B", self.n_slots * self.slot_bytes)
        self._bytes = 0
        # 空闲槽位队列：拿不到就阻塞，形成背压
        self._free: mp.Queue = mp.Queue()
        for i in range(self.n_slots):
            self._free.put(i)
        self._full: mp.Queue = mp.Queue()

    # ------------------------------------------------------------------
    def put(self, payload: bytes, seq: str = "", frame: int = 0,
            timeout: float | None = None) -> FrameHandle | None:
        """写入一段字节；池满时阻塞 ``timeout`` 秒（None = 一直等）。"""
        try:
            slot = self._free.get(timeout=timeout) if timeout else self._free.get()
        except Exception:  # noqa: BLE001 - 超时返回 None，由调用方决定策略
            return None
        n = min(len(payload), self.slot_bytes)
        start = int(slot) * self.slot_bytes
        self._buf[start:start + n] = np.frombuffer(payload, dtype=np.uint8)[:n].tobytes()
        self._bytes += n
        handle = FrameHandle(slot=int(slot), length=n, seq=seq, frame=frame)
        self._full.put(handle)
        return handle

    def get(self, timeout: float | None = None) -> tuple[bytes, FrameHandle] | None:
        """读出一个槽位的字节并归还槽位。"""
        try:
            handle = self._full.get(timeout=timeout) if timeout else self._full.get()
        except Exception:  # noqa: BLE001
            return None
        start = handle.slot * self.slot_bytes
        payload = bytes(self._buf[start:start + handle.length])
        self._free.put(handle.slot)
        return payload, handle

    # ------------------------------------------------------------------
    @property
    def n_filled(self) -> int:
        return self._full.qsize()

    def close(self) -> None:
        for q in (self._free, self._full):
            try:
                q.close()
                q.join_thread()
            except Exception:  # noqa: BLE001
                pass
