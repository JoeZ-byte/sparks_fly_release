# 卫星视频车辆检测与测速（x86 / ARM 双平台）

面向卫星与无人机视频的**极小目标（2~26 像素）**车辆实时检测与测速流水线。
输入一路 1024×1024 视频，输出每帧的车辆框、轨迹 ID 与车速，可直接落到 CSV 或叠加视频。

- **只算该算的像素**：以运动提案选出每帧至多 6 个 256×256 图块，参与网络的像素量约为全图的 44%。
- **只跑跑得动的模型**：YOLOv8n + P2 高分辨率分支 + NWD 回归损失，INT8 量化后 3.47 MB。
- **板端不依赖 Python**：核心链路（图像算子、DFL 解码、NMS、卡尔曼、三级指派、测速、光流先验）用 C/C++ 实现，
  Python 主程序通过 ctypes 直接链接编译好的加速库；同时保留一份**纯 Python** 实现，便于对照与无编译环境使用。

## 目录

```
release/
├── README.md              总说明（本文件）
├── X86/                   x86 平台（512 输入，精度优先）
│   ├── README.md
│   ├── config.json        默认模型与流水线参数
│   ├── stream.py          主程序（--backend native|python）
│   ├── src/               native_bridge.py（ctypes 绑定）、pipeline_py.py（纯 Python 流水线）
│   ├── native/            加速库源码 + 预编译库（satstream.dll）
│   ├── models/            检测模型（ONNX，fp32 + int8）
│   ├── train/             训练代码与配置（satdet 包 + configs + 预训练权重）
│   ├── docs/              架构、指标、部署说明
│   └── samples/           示例视频
└── ARM/                   ARM 平台（320 输入，实时优先）
    └── …（同上，models 为 320 系列，native 预编译库为 libsatstream.so）
```

## 快速开始

```bash
pip install -r requirements.txt            # opencv-python-headless, numpy, onnxruntime

cd X86                                     # 或 cd ARM（板端）
python stream.py --input samples/sample_01_highway_002.mp4 --csv tracks.csv
python stream.py --input samples/sample_08_tiny_033.mp4 --backend python --frames 60
```

默认使用加速库（`--backend native`）；若所在平台没有预编译库，可先编译（见各平台 README），
或直接用 `--backend python` 跑纯 Python 版。

常用参数：

| 参数 | 说明 |
|---|---|
| `--backend native\|python` | native=链接 C/C++ 加速库（默认）；python=纯 Python |
| `--imgsz 512\|320` | 网络输入边长，需与模型匹配 |
| `--max-cells 6` | 每帧最多推理的图块数 |
| `--flow-w 1.0` / `--no-flow` | 光流速度先验权重（**默认开启**） |
| `--no-tophat` | 关闭提案前的白顶帽增强（默认开启） |
| `--min-hits 5` `--gate 0.7` | 轨迹确认门槛与关联门限 |
| `--pin-cores auto\|none\|2,3` | 绑核。飞腾派默认自动绑大核（cpu2,3） |
| `--csv` `--save-video` `--show` | 输出轨迹 CSV / 叠加视频 / 实时窗口 |
| `--seal-input/--seal-csv/--seal-video` + `--seal-key` | 密封（AES-256-GCM）通道，**仅飞腾派启用** |

## 模型一览

所有模型均在同一份训练集上训练（37 段序列，验证固定 010），指标为 010 序列的 mAP30。

**X86（512 输入）**

| 模型 | 文件 | mAP30 | 说明 |
|---|---|---|---|
| **v3（默认）** | `models/a_p2_nwd_v3_raw_int8.onnx` | **0.8474** | 主线模型，精度最高 |
| v3 fp32 | `models/a_p2_nwd_v3_raw_fp32.onnx` | 0.8474 | 不量化版本，便于二次量化 |
| full | `models/a_nwd_p2_full_raw_fp32.onnx` | 0.8379 | 同架构早期版本 |
| full2 | `models/a_nwd_p2_full2_raw_fp32.onnx` | 0.8281 | 同架构另一组采样参数 |

**ARM（320 输入）**

| 模型 | 文件 | mAP30 | 说明 |
|---|---|---|---|
| **kd320（默认）** | `models/a_p2_nwd_kd320_raw_int8.onnx` | **0.8104** | 320 输入 + 特征蒸馏 |
| arm320 | `models/a_p2_nwd_arm320_raw_{fp32,int8}.onnx` | 0.8003 | 320 输入基线 |
| rep320 | `models/a_p2_nwd_rep320_raw_{fp32,int8}.onnx` | 0.8083 | MobileOne 重参数化训练，推理结构与基线一致 |

> 飞腾派不是 4 个同构核：cpu2/3 为 1.8 GHz 大核（cpu_capacity 1024），cpu0/1 为 1.5 GHz 小核（429）。
> 程序会在飞腾派上自动 2 线程 + 绑大核（实测 1088 → 944 ms/帧，+15%；绑小核会慢到 2206 ms/帧）。
>
> 选型依据：x86 有余量，用 512 输入换精度；板端推理是瓶颈，用 320 输入换帧率。
> 同一份 320 模型在树莓派 4B 上单图块 98.6 ms、飞腾派 E2000Q 上 130 ms。

## 实测性能

| 平台 | 输入 | 单图块推理 | 同步端到端 | 备注 |
|---|---|---|---|---|
| x86（Ryzen 9 8945HX，4 线程） | 512 | 26.5 ms | 5.3 FPS | 6 图块/帧 |
| 树莓派 4B（4×A72 1.8 GHz） | 320 | 98.6 ms | 1.49 FPS | 默认配置（光流+白顶帽开启） |
| 飞腾派 E2000Q（2 大核 1.8 GHz + 2 小核 1.5 GHz） | 320 | 130 ms | **1.06 FPS** | 2 线程绑大核（默认） |

同步模式下瓶颈全部在推理；若显示端需要流畅，可用异步检测 + 卡尔曼外推（显示 10 fps、检测 1.2~1.6 Hz），
参数见各平台 docs。

## 文档

- `X86/docs/01_系统架构.md`、`ARM/docs/01_系统架构.md`：五级流水线与各层职责
- `X86/docs/02_性能与精度.md`、`ARM/docs/02_性能与精度.md`：模型与平台实测数据
- `X86/docs/03_部署说明.md`、`ARM/docs/03_部署说明.md`：构建、运行与参数

## 许可

本项目代码用于竞赛与学习用途；模型权重基于 VisDrone 数据训练，使用时请遵守原始数据集许可。