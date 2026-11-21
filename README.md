# 卫星视频车辆流式识别 —— Release 包

把项目流水线打包成**单文件**，对外 0 依赖：不需要项目代码、不需要数据集、
不需要配置环境（exe 版连 Python 都不需要）。

## 目录

```
release/
├─ stream_demo.exe        # ★ 单文件可执行程序（PyInstaller，免 Python）
├─ stream_demo.py         #   同源源码（放 src/，需要 Python 时用）
├─ run_demo.bat           #   双击运行 / 把视频拖到它上面
├─ requirements.txt       #   源码版需要的三个基础库
├─ model/
│  └─ a_p2_nwd_v3_raw_fp32.onnx        # 我们的 YOLO（YOLOv8n+P2+NWD+ECA，38 段全量，12.2 MB）
└─ samples/               # 10 段样例视频（VISO 数据集片段，mp4v/10fps）
   ├─ sample_01_highway_002.mp4   1024² 高速公路立交（效果最好）
   ├─ sample_02_urban_003.mp4     1024² 城区
   ├─ sample_03~05_*.mp4          1024² 城区 ×3
   ├─ sample_06_wide_009.mp4      1024² 目标更大
   ├─ sample_07_test_010.mp4      1024² 测试段
   ├─ sample_08_tiny_033.mp4      280×230 单目标·框仅 6~15 px（最难的尺度）
   ├─ sample_09_tiny_036.mp4      543×285 单目标·框仅 2~7 px
   └─ sample_10_highway2_002.mp4  1024² 高速公路另一段
```

## 性能（CPU 为主，GPU 可选）

同一台笔记本（Ryzen 9 8945HX 16 核 + RTX 5060）实测**稳态 FPS**，
CPU 与 GPU 两条路径都开箱即用：

| 样例 | 分辨率 | **CPU（默认）** | GPU（可选） |
|---|---|---|---|
| 002 高速立交 | 1024² | **11.5** | 53.0 |
| 003 城区 | 1024² | 11.2 | 49.3 |
| 016 极密 | 1024² | 12.7 | 39.1 |
| 022 最密（165 目标/帧） | 1024² | 11.1 | 44.7 |
| 024 / 025 | 1024² | 11.4 / 10.2 | 48.0 / 40.7 |
| 010 | 1024² | 11.3 | 44.1 |
| 001 | 1025² | 10.2 | 41.8 |
| 033 单目标 | 280×230 | 28.8 | 66.8 |

**CPU 路径怎么做到的**（每一步都有实测依据）：

1. **INT8 静态量化**（QDQ、per-channel、240 个真实图块校准）：
   单块 512² 从 35 ms 降到 **17 ms**（2.1×），mAP30 0.8297 → **0.8927**
   （量化反而去掉了部分噪声，精度没降）。
2. **提案硬上限**：原来"每条轨迹加一个图块"，密集场景 42 块/帧；
   改成运动得分归一化 + 轨迹先验后取前 K 个格子，**K≤8 且自适应**。
3. **自适应图块预算**：每 10 帧按实测帧率调整 K（默认目标 12 FPS，
   下限 4）。机器快就自动多用图块提精度，机器慢就自动降到 4 保帧率。
4. **渲染/掩膜优化**：密集场景 150 个目标时，标题文字从每帧 300 次绘制
   减到 ≤60 个标签（18 ms → 1 ms）；运动掩膜改成半分辨率计算（8 ms → 3 ms）。
5. **批量推理**：一帧的 K 个图块打包成一次 `sess.run`（INT8 下快约 18%）。

**GPU 路径**（`--backend auto` 检测到 CUDA 才启用）额外做了：TorchScript
前向 + 固定输入缓冲（避免 JIT 反复重建执行计划导致秒级卡顿）+ 预处理/解码
全部放在显存里，只回传最终框。

> 没有显卡：什么都不用装，`pip install -r requirements.txt` 就是 CPU 路径。
> 有显卡想要更高帧率：`pip install torch --index-url https://download.pytorch.org/whl/cu128`。

### 极慢机器上的进一步提速

```powershell
python src\stream_demo.py --input ... --min-cells 3        # 图块下限降到 3（约 +25% 帧率）
python src\stream_demo.py --input ... --det-every 2        # 隔帧检测（帧率翻倍，MOTA 0.41 -> 0.21）
python src\stream_demo.py --input ... --imgsz 448          # 输入降到 448（快 25%，mAP30 -2 点）
```

## 使用

**方式一（exe，零依赖）**：双击 `run_demo.bat`（默认播放 sample_01），
或把任意视频文件拖到 `run_demo.bat` 上。

**方式二（源码）**：

```powershell
pip install -r requirements.txt        # numpy + opencv + onnxruntime（CPU 路径）
# 想要 GPU 加速（推荐，快 8~10 倍）：
pip install torch --index-url https://download.pytorch.org/whl/cu128
python src\stream_demo.py --input samples\sample_01_highway_002.mp4
```

`--backend auto`（默认）会自动选：有 CUDA 用 PyTorch GPU，否则用 onnxruntime CPU。

常用参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--input` | 输入视频 | 必填 |
| `--conf` | 检测置信度 | 0.35 |
| `--k` | 米/像素标定（0=不显示 km/h，只显示 px/帧） | 0.92 |
| `--fps` | 源视频帧率（测速换算用） | 10 |
| `--save out.mp4` | 把叠加结果存成视频 | 不存 |
| `--max-cells` | 每帧最多推理的图块数（算力预算，越小越快） | 6 |
| `--backend` | `auto` / `onnx` / `torch` | auto |
| `--fp32` | GPU 上用 fp32（默认 fp16） | 关 |
| `--imgsz` | 图块送入网络的边长 | 512 |

## 实现说明（与项目 satdet 同源）

* **运动先验**：背景差分（固定阈值 40、闭运算）→ 16 格投票 → 每帧只推理
  动静最大的 K 块；冷启动前 15 帧全图扫描。
* **跟踪**：8 态卡尔曼 + 椭圆搜索域 + 匈牙利匹配（内置实现，免 scipy）+
  暂定态转正（连续 3 帧才发 ID）+ 轨迹复活。
* **检测**：YOLOv8n+P2+NWD+ECA（**v3 模型**：38 段 / 10.1 万图块 / 84 万实例，
  含 JPEG 退化增强与 NWD C=8），图块 256→512 放大；只解码高分 anchor。
* **车速**：6/12/18 帧多滞后中位差分 + EWMA(0.7) + 物理上限与限幅。
* **静止门限**：连续低速（<0.2 px/帧 ≈ 7 km/h）的目标不显示——交通流
  只关心在动的车，顺带滤掉原地晃的噪点。

## 如实说明

* **速度标签**：k=0.92 米/像素是按吉林一号量级标定的（±10%）；换别的卫星/
  分辨率请用 `--k 0`（只显示 px/帧），否则 km/h 数字不可信。
* **最密场景**：022 这类 165 目标/帧的画面 mAP30 约 0.52，遮挡严重，是已知短板。
* **单目标极小框**：028~038 这类框仅 2~26 px 的序列 mAP30 约 0.28~0.44，
  召回尚可（0.81）但定位精度不足，是当前最难的边界。
* **实时性**：有显卡时 1024² 视频 **26~42 FPS**（远高于 10 FPS 的演示门槛）；
  纯 CPU 约 4 FPS（6 块/帧，已是 ORT 在 4 线程下的极限——单块 36 ms，
  加线程反而变慢，因为瓶颈是内存带宽）。
