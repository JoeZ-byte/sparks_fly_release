# x86 版（512 输入，精度优先）

默认模型 `models/a_p2_nwd_v3_raw_int8.onnx`（512 输入，010 序列 mAP30 0.8474）。

## 环境

```bash
pip install -r ../requirements.txt
```

需要 `opencv-python`（含 imshow 的完整版可做实时预览）、`numpy`、`onnxruntime`。
加速库运行时会在 pip 安装目录里自动寻找 `onnxruntime.dll`，无需额外配置 PATH。

## 两种后端

| 后端 | 命令 | 说明 |
|---|---|---|
| 加速库（默认） | `python stream.py --input samples/sample_01_highway_002.mp4` | ctypes 链接 `native/prebuilt/satstream.dll` |
| 纯 Python | 加 `--backend python` | 无编译依赖，速度接近（见 docs/02），便于对照 |

## 编译加速库（可选）

预编译的 `native/prebuilt/satstream.dll` 已随包提供（Windows x64，静态链接，无外部依赖）。
如需自行编译：

```bat
cd native
mingw32-make          :: MinGW-w64；产物在 prebuilt\
```

Linux/macOS 下 `make` 会产出 `libsatstream.so`；`stream.py` 会自动按平台挑选库文件名，
也可用 `--native-lib <path>` 显式指定。

## 运行示例

```bash
python stream.py --input samples/sample_08_tiny_033.mp4 --csv out.csv
python stream.py --input samples/sample_01_highway_002.mp4 --max-cells 6 --threads 8
python stream.py --input samples/sample_01_highway_002.mp4 --save-video out.mp4 --show
```

## 2.2 输出模式与异步

```bash
python stream.py --input in.mp4 --mode csv                    # 只记 CSV（不开窗口/不写视频）
python stream.py --input in.mp4 --mode video --save-video r.mp4
python stream.py --input in.mp4 --mode show                   # 实时演示
python stream.py --input in.mp4 --mode show --async           # 异步：显示流畅、检测按能力跑
```

- `--mode` 取值 `csv|video|show|all`，`auto`（默认）按 `--show/--save-video` 推断。
- `--async` 的后台线程与 `--out-fps` 限速；**默认值由 `config.json` 的 `"async": true` 控制**，两平台一致。
- 无窗口无视频时异步自动改为逐帧阻塞，保证每帧都被处理。

## 实测（Ryzen 9 8945HX，4 线程，6 图块/帧）

| 配置 | 单图块推理 | 端到端 |
|---|---|---|
| 512 输入 INT8 | 26.5 ms | 189 ms/帧（5.3 FPS） |
| 320 输入 INT8 | 12.0 ms | 83 ms/帧（12.1 FPS） |

## 训练

```bash
cd train
python satdet.py train --config configs/a_nwd_p2_v3.yaml      # 512 主线
```

配置里的 `data.train_seqs` 为 37 段序列、`val_seqs` 为 010。
训练需要 PyTorch（见仓库根 `requirements-gpu.txt` 的说明），推理不需要。
`pretrained/yolov8n_coco.pt` 是 COCO 初始化权重，训练时用于加载骨干。

## 文档

- `docs/01_系统架构.md`：五级流水线
- `docs/02_性能与精度.md`：模型与平台数据
- `docs/03_部署说明.md`：参数与常见问题