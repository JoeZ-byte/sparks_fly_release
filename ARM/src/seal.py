"""密封（加密）通道：AES-256-GCM 分块流式加解密 + 平台判定。

仅飞腾派（异构成 2 大核 + 2 小核的 Phytium 板）触发本模块；
其它平台除非显式 `--force-seal`，否则拒绝执行，避免误用。

文件格式（与工程内其它实现一致，可互通）：
    MAGIC(8) "SFLYSEAL" | chunk_size(4, LE) | { nonce(12) | ciphertext(chunk+16) } *
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

MAGIC = b"SFLYSEAL"
ENC_SUFFIX = ".enc"
DEFAULT_KEY_ENV = "SATSTREAM_SEAL_KEY"


# ---------------------------------------------------------------- 平台判定
def board_model() -> str:
    for p in ("/sys/firmware/devicetree/base/model", "/proc/device-tree/model"):
        try:
            return Path(p).read_bytes().decode("utf-8", "ignore").strip("\x00")
        except OSError:
            continue
    return ""


def cpu_capacities() -> list[int]:
    caps = []
    base = Path("/sys/devices/system/cpu")
    for d in sorted(base.glob("cpu[0-9]*")):
        f = d / "cpu_capacity"
        if f.exists():
            try:
                caps.append(int(f.read_text().strip()))
            except (OSError, ValueError):
                pass
    return caps


def is_phytium() -> bool:
    """飞腾派的判定：设备树型号含 Phytium/E2000，或 aarch64 且大小核容量不同。"""
    try:
        model = board_model().lower()
    except Exception:
        model = ""
    if "phytium" in model or "e2000" in model:
        return True
    try:
        import platform
        if platform.machine() != "aarch64":
            return False
    except Exception:
        return False
    caps = cpu_capacities()
    if len(caps) >= 2 and max(caps) != min(caps):
        return True
    return False


def big_cores() -> list[int]:
    """返回大核编号列表（容量等于最大值的那些核）；无法判定时返回空列表。"""
    caps = cpu_capacities()
    if not caps or max(caps) == min(caps):
        return []
    top = max(caps)
    return [i for i, c in enumerate(caps) if c == top]


# ---------------------------------------------------------------- 加解密
def parse_key(text: str) -> bytes:
    raw = (text or "").strip()
    if not raw:
        raw = os.environ.get(DEFAULT_KEY_ENV, "")
    if len(raw) != 64:
        raise SystemExit("密钥需要 64 位十六进制字符（AES-256）：--seal-key 或环境变量 %s"
                         % DEFAULT_KEY_ENV)
    return bytes.fromhex(raw)


def _aead(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise SystemExit("需要 cryptography：pip install cryptography") from exc
    return AESGCM(key)


def seal_file(src: Path, dst: Path, key: bytes, chunk: int = 1 << 20) -> int:
    aead = _aead(key)
    total = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(MAGIC)
        fout.write(struct.pack("<I", chunk))
        while True:
            data = fin.read(chunk)
            if not data:
                break
            nonce = os.urandom(12)
            fout.write(nonce)
            fout.write(aead.encrypt(nonce, data, None))
            total += len(data)
    return total


def unseal_file(src: Path, dst: Path, key: bytes) -> int:
    aead = _aead(key)
    total = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        if fin.read(8) != MAGIC:
            raise SystemExit("不是密封文件（magic 不匹配）：%s" % src)
        (chunk,) = struct.unpack("<I", fin.read(4))
        while True:
            nonce = fin.read(12)
            if len(nonce) < 12:
                break
            blob = fin.read(chunk + 16)
            if not blob:
                break
            plain = aead.decrypt(nonce, blob, None)
            fout.write(plain)
            total += len(plain)
    return total


def wipe(path: Path, passes: int = 1) -> None:
    """把明文文件覆写后删除（内存文件系统下删除即回收）。"""
    try:
        n = path.stat().st_size
        with open(path, "r+b") as f:
            f.write(os.urandom(n))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def tmp_plain_path(tag: str, suffix: str = ".plain") -> Path:
    """解密落地路径：优先 /dev/shm（内存），否则用系统临时目录。"""
    shm = Path("/dev/shm")
    base = shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(os.environ.get("TEMP", "/tmp"))
    return base / ("satstream_%s_%d%s" % (tag, os.getpid(), suffix))


def seal_path_for(p: str) -> Path:
    q = Path(p)
    return q if q.suffix == ENC_SUFFIX else q.with_suffix(q.suffix + ENC_SUFFIX)