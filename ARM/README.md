# ARM 版（320 输入，实时优先）

默认模型 `models/a_p2_nwd_kd320_raw_int8.onnx`（320 输入，010 序列 mAP30 0.8104）。

## 环境（板端）

```bash
sudo apt install -y python3-opencv python3-numpy        # 或 pip install -r ../requirements.txt
pip install onnxruntime                                  # 树莓派/飞腾派均为 aarch64 轮子
```

## 两种后端

| 后端 | 命令 | 说明 |
|---|---|---|
| 加速库（默认） | `python3 stream.py --input samples/sample_01_highway_002.mp4` | 链接 `native/prebuilt/libsatstream.so` |
| 纯 Python | 加 `--backend python` | 不需要 g++ 与预编译库 |

## 编译加速库（板端原生编译）

```bash
cd native
make -j4          # 产物 prebuilt/libsatstream.so
```

已随包提供 aarch64 预编译库（树莓派 4B / 飞腾派通用）。若目标板 glibc 版本较低，
请在板端重新 `make` 一次。

## 运行示例

```bash
python3 stream.py --input samples/sample_01_highway_002.mp4 --csv out.csv
python3 stream.py --input samples/sample_08_tiny_033.mp4 --max-cells 4 --threads 4
```

## 实测

| 平台 | 单图块推理（320 INT8） | 同步端到端 | 异步（显示 / 检测） |
|---|---|---|---|
| 树莓派 4B | 98.6 ms | 671.6 ms/帧（1.49 FPS） | 9.4 fps / 1.17 Hz |
| 飞腾派 E2000Q | 130 ms | 1050 ms/帧（0.95 FPS） | 10.1 fps / 1.62 Hz |

板端瓶颈在推理：6 图块/帧时推理占端到端 89%。若需要流畅显示，可降 `--max-cells` 到 4，
或使用异步检测（见 `docs/03_部署说明.md`）。

## 训练

```bash
cd train
python satdet.py train --config configs/a_nwd_p2_arm320_kd.yaml     # 320 + 蒸馏
python satdet.py train --config configs/a_nwd_p2_arm320_rep4.yaml   # 320 + 重参数化
```

蒸馏配置需要一个 512 的教师权重（默认指向 `weights/a_nwd_p2_v3_best.pt`，可通过 `teacher` 字段改路径）。

## 文档

- `docs/01_系统架构.md`：五级流水线
- `docs/02_性能与精度.md`：模型与平台数据
- `docs/03_部署说明.md`：板端部署与调参