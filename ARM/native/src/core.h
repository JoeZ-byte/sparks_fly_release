// 纯 C++ 流水线 · 基础类型与工具
#pragma once
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sfly {

struct Box {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct Det {
    Box b;
    float s = 0;
};

// Python 侧 stream_demo 的常量
constexpr int kRegMax = 16;
constexpr int kTile = 256;
// P2 模型的四个 stride
constexpr int kStrides[4] = {4, 8, 16, 32};

inline float iou(const Box& a, const Box& b) {
    const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    const float w = x2 - x1, h = y2 - y1;
    if (w <= 0 || h <= 0) return 0.f;
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / std::max(1e-6f, area_a + area_b - inter);
}

inline void logf(const char* fmt, ...);

}  // namespace sfly