# 飞腾派版 · 卫星视频车辆流式识别（板端专用）

> 赛题：**基于飞腾 CPU 的创新应用系统设计**（软件赛道）
> 本目录是**板端独立的一套**：x86 版在 `release/`，两者核心算法一致、默认参数不同。


## 0. 快速开始（含连接）

```powershell
# PC 侧：一键打通串口 -> 板端 IP -> SSH 免密（USB-TTL 已接好时）
powershell -ExecutionPolicy Bypass -File scripts\96_board_setup.ps1

# PC 侧：把本目录推到板子并安装依赖（板子无外网，用离线 wheel）
scp -r board board:~/
ssh board "cd ~/board && bash install.sh"

# 板端：跑演示
ssh board "cd ~/board && python3 stream_board.py --input samples/sample_01_highway_002.mp4"
```

连接细节（接线、串口、IP、排障、重启恢复）见 **`docs/12_飞腾派部署与优化.md`**。

## 1. 应用场景与创新点（对应赛题要求 1、4）

| 赛题要求 | 本作品的落地 |
|---|---|
| 真实场景 + 行业痛点 | 卫星视频（吉林一号类）车辆监控：目标只有 **5~9 像素**、视频是**压缩流**、许多场景**没有外网**。传统"整帧缩放到 640 再送网络"的做法在结构上无效（6px 目标缩成 3.75px） |
| 功能闭环 | 读视频 → 运动先验选图块 → 检测 → 跟踪 + 测速 → 叠加渲染，全链路在板端离线完成 |
| 更低功耗 | **整机 4 核 1.5 GHz / 3.85 GB / 无独显**，功耗数量级低于带 GPU 的工作站 |
| 更高安全性 | **全程离线**（板子无外网），视频与结果不出本地；依赖离线 wheel 安装 |
| 更智能交互 | 画面实时叠加 `#ID + 车速`，静止目标自动不画（只关注交通流） |

## 2. 挖掘 CPU 性能（对应赛题要求 2）

| 手段 | 说明 | 数据 |
|---|---|---|
| **多核并行** | ORT 4 线程（= 物理核数）；线程数扫描确定甜点 | 1/2/3/4 线程：728 / 391 / 359 / **310** ms/块 |
| **异步流水** | 检测放后台线程，显示与检测解耦；中间帧卡尔曼外推 | 显示 **0.9 → 10.0 FPS** |
| **算力预算** | 每帧只推理 4 个 256² 图块（运动先验 + 轨迹先验 + 陈旧度轮转） | 覆盖 25%→轮转补全 |
| **缓存友好** | 图块 256 → 网络 512（2× 放大），小目标落在 P2(stride 4) 上 | 6px 目标 ≈ 3 格 |
| **NEON** | aarch64 wheel 自带 NEON（`asimd` 已确认） | 无需自编译 |

## 3. 代码级优化与前后对比（对应赛题要求 3）

### 3.1 模型：输入降维 + INT8 静态量化

| 版本 | 精度 | 单块（4 线程） | 相对 |
|---|---|---|---|
| v3 @512 | FP32 | 876 ms | 1.0× |
| v3 @512 | INT8 | 310 ms | 2.8× |
| **arm320 @320（板端专用）** | **INT8** | **130 ms** | **6.7×** |

模型对比（验证集 010 段，oracle 提案、无 TTA）：

| 模型 | 输入 | mAP30 | 板端单块 |
|---|---|---|---|
| v3 | 512 | **0.8474** | 306 ms |
| arm320 | 320 | 0.8003 | **130 ms** |

> 用 4.7 个点的 mAP 换 2.4× 速度，板上 4 块一帧从 1.22 s 降到 0.52 s。

> 板子 CPU 只有 `asimd/aes/sha2`，**没有 dotprod/i8mm**，所以 INT8 拿不到 4× 的理论收益，
> 但 2.8× 仍然是最大的一笔；且量化后 mAP30 不降（笔记本上实测 0.8297 → 0.8927）。

### 3.2 端到端：异步检测 + 外推

| 指标 | 最初（512 模型 + 同步） | 512 模型 + 异步 | **320 模型 + 异步（当前默认）** |
|---|---|---|---|
| 显示帧率（1024²） | 0.9 FPS | 10.0 FPS | **10.1 FPS** |
| 检测频率 | 0.9 Hz | 0.70 Hz | **1.62 Hz（1.8×）** |
| 单块耗时 | 306 ms | 306 ms | **130 ms** |

**相对最初版本：显示 11×、检测 1.8×。**

### 3.3 复现

```bash
# 板端（首次）
bash install.sh                     # 离线装 numpy/onnxruntime/opencv
python3 bench.py                    # 打印线程/精度/端到端基准

# 板端（演示）
python3 stream_board.py --input samples/sample_01_highway_002.mp4
```

`stream_board.py` 的板端默认值：异步检测开、10 FPS 节流、外推 15 帧、4 线程、4 图块、不预热。


## 3.4 涉密数据保护（加密链路）

卫星影像属于涉密数据，本版提供**文件级 AES-256-GCM 密封**，全程走飞腾
ARMv8 Crypto Extensions（硬件指令）：

```bash
# 地面站侧：加密视频（模拟专网下发）
python scripts/97_seal_tools.py seal --in sample_01.mp4 --out sample_01.enc --key <64hex>

# 板端：解密到内存（/dev/shm）后解码，退出擦除明文；结果加密落盘
python3 stream_board.py --seal-input samples/sample_01.enc --seal-key <64hex> \
        --seal-csv /dev/shm/tracks.enc
```

| 指标 | 实测（飞腾派） |
|---|---|
| AES-256-GCM 吞吐（硬件） | **1.31 GB/s**（软件 47.9 MB/s，**27×**） |
| 解密 13 MB 视频 | **82 ms**（占播放时长 2%） |
| 明文残留 | 无（`/dev/shm` + 0 覆写删除） |

详见 `docs/14_加密与安全设计.md`。

## 4. 与 x86 版的关系

| | `release/`（x86 版） | `board/`（飞腾派版） |
|---|---|---|
| 检测后端 | PyTorch CUDA（可选）/ onnxruntime CPU | onnxruntime CPU |
| 默认输入 | 512 | 512（320 版模型训练中） |
| 检测调度 | 同步逐帧 | **异步 + 外推** |
| 帧率 | 26~42 FPS（GPU）/ 10 FPS（CPU） | 10 FPS 显示 / 0.7 Hz 检测 |
| 生成本目录 | `scripts/92_make_board_edition.py` 从 x86 版生成，逻辑同源 | — |

## 5. 板卡事实（现场可复现）

```
Phytium Pi Board | aarch64 | 4 核 1.5 GHz | 3.85 GB
Linux 6.6.12-phytium-embedded-v3.0 (Debian 12) | Python 3.11.2
CPU: asimd aes sha2   （无 dotprod / i8mm / fp16 / sve）
加速器: /dev/dri/card0（显示），无 NPU
外网: 无
```