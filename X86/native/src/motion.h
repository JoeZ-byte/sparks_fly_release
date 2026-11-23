// 运动提案：半分辨率帧差 + 格子投票 + 轨迹先验 + 陈旧度轮转
// 与 Python stream_demo.MotionProposer 逐值对齐
#pragma once
#include <vector>

#include "core.h"
#include "img.h"

namespace sfly {

struct MotionProposer {
    int w = 0, h = 0;
    float thr = 40.f;
    int max_cells = 6;
    float alpha = 0.02f;
    std::vector<float> bg;        // (h/2) x (w/2) 背景模型
    std::vector<float> last_seen; // ny x nx，每格上次被检查的帧号
    int nx = 0, ny = 0;
    int frame = 0;

    MotionProposer(int w_, int h_, float thr_ = 40.f, int max_cells_ = 6,
                   float alpha_ = 0.02f);

    // 半分辨率帧差掩膜（bg 为空时初始化并返回全 0）
    GrayImage mask(const GrayImage& gray);
    // 每格的"运动像素数"（ny x nx，行优先）
    void cells(const GrayImage& mask, std::vector<int64_t>& out) const;
    // 选本帧要推理的图块（返回 (x0,y0) 列表，长度 = max_cells）
    void propose(const GrayImage& gray, const std::vector<Box>& track_boxes,
                 std::vector<std::pair<int, int>>& origins, float track_bonus = 0.6f);
    void update_bg(const GrayImage& gray);
    // 半分辨率入口：直接喂 gray_area_half 的结果，省掉内部 resize
    void update_bg_half(const GrayImage& half);
    GrayImage mask_half(const GrayImage& half);
    void propose_half(const GrayImage& half, const std::vector<Box>& track_boxes,
                      std::vector<std::pair<int, int>>& origins, float track_bonus = 0.6f);

    /* 小目标增强（docs/14 一档）：0=关闭 1=白顶帽 2=DoG。
     * 打开后掩膜与背景模型都在"增强域"里累积，等价于在帧差之前做增强。 */
    int pre_mode = 0;
    /* 帧差模式（docs/14 E5）：0=与背景帧差（当前）；1=三帧差分（|It-It1| 与 |It1-It2| 取交） */
    int diff_mode = 0;
    GrayImage prev1, prev2;   // 增强域的前两帧
    void update_bg_prepared(const GrayImage& enhanced);
    GrayImage mask_prepared(const GrayImage& enhanced);

    /* 背景模型（docs/14 E6）：
     *   0 = EMA（当前实现，默认）
     *   1 = ViBe（逐像素 N 个历史样本，命中不到就判前景）
     *   2 = 两者取并（EMA 或 ViBe 任一判前景即前景）
     *   3 = 两者取交（都判前景才算，最保守）
     * ViBe 用 N=8 个样本、距离阈值 R、随机替换概率 1/phi。 */
    int bg_mode = 0;
    int vibe_n = 8;
    int vibe_r = 20;
    int vibe_update_div = 16;      // 1/16 概率替换样本
    std::vector<uint8_t> vibe;     // (h/2)*(w/2)*vibe_n
    unsigned vibe_seed = 12345u;
    void vibe_fill(const GrayImage& half);
    GrayImage vibe_mask_prepared(const GrayImage& half);

    // 冷启动稀疏网格
    void warmup_grid(int stride, std::vector<std::pair<int, int>>& origins) const;
};

}  // namespace sfly