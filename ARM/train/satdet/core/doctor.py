"""环境自检：一条命令看清"能不能开工"。

用途
----
1. 新同学到岗：``satdet doctor`` 直接告诉他缺什么。
2. 换机器/换环境：把 JSON 报告贴进 issue 或流程单。
3. 出报告时：把版本号写进"实验环境"章节，保证可复现。

检查项
------
* Python / 操作系统 / CPU 核数 / 内存
* torch + CUDA + GPU 型号与算力（sm_120 这类新卡必须核验）
* 关键三方库版本（cv2 / onnx / onnxruntime / torchvision / ultralytics）
* 数据目录是否可读、序列是否齐全、标注是否有硬伤
* 产物目录是否可写（G: 盘是 exFAT，要单独验证）
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from satdet.core.logging import get_logger
from satdet.core.paths import artifacts_dir, dataset_root, ensure_dir, project_root

__all__ = ["collect_env", "format_report", "run_doctor"]

log = get_logger("doctor")

_PROBE_PACKAGES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python"),
    ("PIL", "pillow"),
    ("yaml", "PyYAML"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
)


def _pkg_version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - 缺包是正常状态，不该炸
        return None
    return str(getattr(mod, "__version__", "?"))


def _torch_info() -> dict:
    info: dict = {"installed": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
        return info

    info.update({
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    })
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "device_name": props.name,
            "device_count": torch.cuda.device_count(),
            "capability": f"sm_{props.major}{props.minor}",
            "vram_gb": round(props.total_memory / 2**30, 2),
        })
    return info


def _cpu_ram() -> dict:
    out = {"logical_cores": os.cpu_count()}
    try:
        import psutil
        out["physical_cores"] = psutil.cpu_count(logical=False)
        out["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def _disk(path: Path) -> dict:
    try:
        usage = shutil.disk_usage(path)
        return {"path": str(path), "free_gb": round(usage.free / 2**30, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": repr(exc)}


def _writable_probe(path: Path) -> bool:
    """真实写一个小文件：exFAT / 网络盘 / 只读挂载都会在这里暴露。"""
    try:
        d = ensure_dir(path)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".probe_", suffix=".tmp", delete=True) as fh:
            fh.write(b"ok")
            fh.flush()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("产物目录不可写 %s: %s", path, exc)
        return False


def collect_env(with_dataset: bool = True) -> dict:
    """采集全部环境信息，返回可 JSON 序列化的字典。"""
    report: dict = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu_ram": _cpu_ram(),
        "torch": _torch_info(),
        "packages": {label: _pkg_version(module) for module, label in _PROBE_PACKAGES},
        "paths": {
            "project_root": str(project_root()),
            "dataset_root": str(dataset_root()),
            "artifacts_dir": str(artifacts_dir()),
        },
        "disk": _disk(project_root()),
        "writable": {"artifacts": _writable_probe(artifacts_dir())},
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith("SATDET_")},
    }

    if with_dataset:
        ds = dataset_root()
        report["dataset"] = {"root": str(ds), "exists": ds.is_dir(), "sequences": [], "issues": []}
        if ds.is_dir():
            try:
                from satdet.data.viso_index import discover_sequences

                index = discover_sequences(ds)
                report["dataset"]["sequences"] = index.summary_rows()
                report["dataset"]["issues"] = index.diagnostics()
            except Exception as exc:  # noqa: BLE001
                report["dataset"]["issues"] = [f"索引失败: {exc!r}"]
    return report


def format_report(report: dict) -> str:
    """把字典渲染成对齐的纯文本表格。"""
    lines: list[str] = []
    py = report["python"]
    plat = report["platform"]
    lines.append(f"Python      : {py['version']} ({py['implementation']})")
    lines.append(f"解释器路径  : {py['executable']}")
    lines.append(f"系统        : {plat['system']} {plat['release']} {plat['machine']}")

    cpu = report["cpu_ram"]
    ram = cpu.get("ram_gb", "?")
    lines.append(f"CPU / 内存  : {cpu.get('physical_cores', '?')} 物理核 / "
                 f"{cpu.get('logical_cores', '?')} 逻辑核 / {ram} GB")

    torch = report["torch"]
    if torch.get("installed"):
        cuda = "可用" if torch["cuda_available"] else "不可用（将退回 CPU）"
        lines.append(f"PyTorch     : {torch['version']}  CUDA {torch.get('cuda_version')}  cuDNN {torch.get('cudnn_version')}")
        lines.append(f"CUDA        : {cuda}")
        if torch["cuda_available"]:
            lines.append(f"GPU         : {torch['device_name']} ({torch['capability']}, "
                         f"{torch['vram_gb']} GB)")
    else:
        lines.append(f"PyTorch     : 未安装 ({torch.get('error')})")

    lines.append("")
    lines.append(f"{'依赖包':<18}{'版本':<20}状态")
    lines.append("-" * 52)
    for label, ver in report["packages"].items():
        lines.append(f"{label:<18}{(ver or '-'):<20}{'OK' if ver else '缺失'}")

    disk = report["disk"]
    lines.append("")
    lines.append(f"项目根目录  : {report['paths']['project_root']}")
    lines.append(f"数据集目录  : {report['paths']['dataset_root']}")
    lines.append(f"磁盘剩余    : {disk.get('free_gb', '?')} GB")
    lines.append(f"产物可写    : {report['writable']['artifacts']}")

    ds = report.get("dataset")
    if ds:
        lines.append("")
        if not ds["exists"]:
            lines.append(f"!! 数据集目录不存在: {ds['root']}")
        else:
            lines.append(f"{'序列':<8}{'帧数':>7}{'分辨率':>12}{'实例':>9}{'轨迹':>7}  标注源")
            lines.append("-" * 60)
            for r in ds["sequences"]:
                res = f"{r['width']}x{r['height']}"
                lines.append(f"{r['seq']:<8}{r['frames']:>7}{res:>12}"
                             f"{r['instances']:>9}{r['tracks']:>7}  {r['gt_source']}")
            if ds["issues"]:
                lines.append("")
                lines.append("数据隐患:")
                for issue in ds["issues"]:
                    lines.append(f"  - {issue}")
    return "\n".join(lines)


def run_doctor(save: bool = True, with_dataset: bool = True) -> dict:
    """执行自检，打印报告，可选落盘 JSON。"""
    report = collect_env(with_dataset=with_dataset)
    text = format_report(report)
    print(text)
    if save:
        path = ensure_dir(artifacts_dir()) / "doctor_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        log.info("自检报告已写入 %s", path)
    return report
