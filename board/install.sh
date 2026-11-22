#!/bin/bash
# 飞腾派板端依赖安装（优先离线 wheel，板子无外网也能装）
#   bash install.sh                  # 用同目录 wheels/ 离线安装
#   bash install.sh --online         # 有外网时走镜像（不推荐，板子通常无网）
set -e
cd "$(dirname "$0")"
PKGS="numpy onnxruntime opencv-python-headless cryptography"

if [ "$1" = "--online" ]; then
    echo "== 在线安装（需要外网）=="
    python3 -m pip install --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple $PKGS
else
    if [ ! -d wheels ]; then
        echo "缺少 wheels/ 目录。请先在 PC 上下载 aarch64 wheel（见 docs/12 第 8 节），"
        echo "或用 bash install.sh --online（板子能上外网时）。"
        exit 1
    fi
    echo "== 1) 引导 pip（解包 wheel，板子没有 pip/ensurepip）=="
    PKG=$(ls wheels/pip-*.whl | head -1)
    rm -rf /tmp/pipboot && mkdir -p /tmp/pipboot
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$PKG" /tmp/pipboot
    PYTHONPATH=/tmp/pipboot python3 -m pip --version

    echo "== 2) 离线安装依赖 =="
    PYTHONPATH=/tmp/pipboot python3 -m pip install --no-index --find-links wheels \
        --break-system-packages $PKGS
fi

echo "== 3) 验证 =="
python3 - <<'PY'
import numpy, cv2, onnxruntime as ort
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("ort", ort.__version__, ort.get_available_providers())
try:
    import cryptography
    print("cryptography", cryptography.__version__, "(AES-GCM 硬件加速可用)")
except ImportError:
    print("cryptography 未安装：密封功能不可用（其余功能正常）")
PY
echo "== 完成 =="