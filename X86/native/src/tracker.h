// 跟踪：8 状态卡尔曼 + 门限匹配（与 Python stream_demo 逐值对齐）
#pragma once
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core.h"

namespace sfly {

// ---------------------------------------------------------------------
// 小矩阵工具（double，行优先；只做 8 维以内的定长运算）
// ---------------------------------------------------------------------
using Mat8 = double[64];

void mat_mul(const double* a, const double* b, double* out, int n, int k, int m);
void mat_add(const double* a, const double* b, double* out, int n);
void mat_inv4(const double* a, double* out);   // 4x4 高斯-约当（部分主元）

// ---------------------------------------------------------------------
// 8 状态卡尔曼：x = [cx, cy, w, h, vx, vy, vw, vh]
// ---------------------------------------------------------------------
struct Kalman8 {
    double x[8]{};
    double P[64]{};
    double F[64]{};
    double H[32]{};   // 4x8
    double Q[64]{};
    double R[16]{};   // 4x4

    Kalman8();
    void init(const Box& b);
    Box predict();
    void update(const Box& b);
    void velocity(float& vx, float& vy) const { vx = static_cast<float>(x[4]); vy = static_cast<float>(x[5]); }
};

// ---------------------------------------------------------------------
// 轨迹
// ---------------------------------------------------------------------
struct Track {
    Kalman8 kf;
    Box box;
    float score = 0;
    int hits = 1;
    int tsu = 0;                 // 距上次更新的帧数
    int pid = -1;                // -1 = 未确认
    float adesc[8] = {0};         // 外观 EMA（8-bin 灰度直方图）
    bool has_desc = false;

    Track() = default;
    Track(const Box& b, float s);
    void predict();
    void update(const Box& b, float s);
};

// ---------------------------------------------------------------------
// 匹配
// ---------------------------------------------------------------------
// 带门限的三级匹配：候选截断 -> 连通分量 -> 小组匈牙利 / 大组贪心
std::vector<std::pair<int, int>> gated_assign(const std::vector<double>& cost,
                                              int n_tr, int n_det,
                                              int top_k = 3, int exact_max = 12);
// 小规模精确匈牙利（行数 <= 列数时可返回 -1 表示该行没配上）
std::vector<int> hungarian(const std::vector<double>& cost, int n, int m);
std::vector<std::pair<int, int>> greedy_assign(const std::vector<double>& cost,
                                               int n, int m);

// ---------------------------------------------------------------------
// 跟踪器
// ---------------------------------------------------------------------
struct Tracker {
    int min_hits = 3;
    int max_age = 15;
    double gate = 1.0;
    int display_coast = 2;
    int next_id = 1;
    int frame = 0;
    std::vector<Track> tracks;
    struct Dead {
        int pid;
        Box box;
        float vx, vy;
        int frame;
    };
    std::deque<Dead> dead;

    Tracker(int min_hits_ = 3, int max_age_ = 15, double gate_ = 1.0,
            int display_coast_ = 2);
    void seed(const std::vector<std::tuple<int, Box, float, float, float>>& items);
    void coast();
    // 返回需要显示的轨迹（pid 已分配且 tsu <= display_coast）
    std::vector<int> update(const std::vector<Det>& dets,
                            const float* descs = nullptr, int n_desc = 0,
                            const float* flows = nullptr, int n_flow = 0);

    // 外观代价权重（0=关；docs/14 E8）
    float app_w = 0.0f;
    // 光流预测权重（0=关；docs/14 E10）：1=完全用 flow 预测位置
    float flow_w = 0.0f;
    // 当前应显示的轨迹下标（pid 已确认 + 外推帧数未超限）
    std::vector<int> shown() const;
    bool revive(const Box& box, float score);
};

}  // namespace sfly