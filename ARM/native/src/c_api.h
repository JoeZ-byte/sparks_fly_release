/* 纯 C 接口：C 胶水层（main_c.c）与 C++ 内核之间的唯一边界。
 *
 * 设计动机
 * --------
 * 板端（ARM）交付要给评委看"关键胶水代码是 C"，所以把工程切成两半：
 *   - 内核（C++）：解码/卡尔曼/指派/图像算子，已经在 parity 测试里逐值对齐；
 *   - 胶水（C99）：命令行、视频读取、CSV、统计打印，全部用 C 写。
 * 这个头文件是两者之间**唯一**的契约，只用 C 的类型，不出现 std::*。
 *
 * 用法
 * ----
 *   SflyCtx* ctx = sfly_create(&cfg, err, sizeof err);
 *   ...
 *   SflyFrame fr;
 *   if (sfly_step(ctx, bgr, w, h, &fr, err, sizeof err) == 0) { ... }
 *   sfly_destroy(ctx);
 *
 * 线程安全：一个 ctx 只允许单线程调用（内部有可变状态）。
 */
#ifndef SFLY_C_API_H
#define SFLY_C_API_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef SFLY_BUILD_DLL
#    define SFLY_API __declspec(dllexport)
#  else
#    define SFLY_API __declspec(dllimport)
#  endif
#else
#  define SFLY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SflyCtx SflyCtx;

/* 建路参数。字符串用 const char*，由调用方保证生命周期覆盖 sfly_create。 */
typedef struct {
    const char* model_path; /* 必填：板端 INT8 ONNX 路径 */
    const char* ort_lib;    /* 可空：手动指定 onnxruntime 动态库；空=自动搜索 */
    int imgsz;              /* 网络输入边长，板端 320 */
    int max_cells;          /* 每帧最多推理几个 256x256 图块 */
    int threads;            /* ORT 线程数 */
    int warmup;             /* 前面多少帧只更新背景、不检测 */
    int max_coast;          /* 轨迹最多外推显示多少帧 */
    float conf;             /* 置信度阈值（对 logit 生效，与 Python 版一致） */
    int pre_mode;           /* 小目标增强：0=关 1=白顶帽 2=DoG（docs/14 一档） */
    int min_hits;           /* 轨迹确认所需连续命中帧数（默认 3） */
    float gate;             /* 跟踪门限倍率（默认 1.0，越小越严） */
    float app_w;            /* 外观代价权重（0=关；docs/14 E8） */
    int diff_mode;          /* 帧差模式：0=背景帧差 1=三帧差分（docs/14 E5） */
    int bg_mode;            /* 背景模型：0=EMA 1=ViBe 2=并 3=交（docs/14 E6） */
    float flow_w;           /* 光流先验权重：0=关（docs/14 E10） */
} SflyConfig;

/* 一个提案图块的左上角（供消融实验统计粗筛召回） */
typedef struct {
    int x, y;
} SflyOrigin;

/* 一条输出轨迹（已确认 pid + 测速） */
typedef struct {
    int track_id;
    float x1, y1, x2, y2;
    float px_per_frame; /* 像素/帧 */
    int speed_ready;    /* 1 = 区间法已算出，0 = 卡尔曼兜底 */
    float kmh;          /* 已按 k=0.92 m/px、fps=10 换算好，供直接显示 */
} SflyTrack;

/* 单帧输出。tracks/origins 指向 ctx 内部缓冲，下一帧调用后失效。 */
typedef struct {
    int tiles;            /* 本帧实际推理的图块数 */
    int n_dets;           /* 本帧检测数（合并 + NMS 之后） */
    int n_tracks;         /* 本帧输出轨迹数 */
    const SflyTrack* tracks;
    const SflyOrigin* origins; /* 本帧提案的图块左上角，长度 = tiles */
} SflyFrame;

/* 运行统计（累计值） */
typedef struct {
    double motion_ms;
    double infer_ms;
    double track_ms;
    long long tiles_total;
    long long dets_total;
    int frames;
} SflyStats;

/* 创建 / 销毁。失败返回 NULL，并把原因写进 err（可为 NULL）。 */
SFLY_API SflyCtx* sfly_create(const SflyConfig* cfg, char* err, int err_len);
SFLY_API void sfly_destroy(SflyCtx* ctx);

/* 处理一帧 BGR24（行优先、紧密排列）。成功返回 0，失败返回 -1。 */
SFLY_API int sfly_step(SflyCtx* ctx, const uint8_t* bgr, int w, int h, SflyFrame* out, char* err,
              int err_len);

/* 取累计统计 */
SFLY_API void sfly_get_stats(const SflyCtx* ctx, SflyStats* out);

/* 库版本（打印用） */
SFLY_API const char* sfly_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SFLY_C_API_H */
