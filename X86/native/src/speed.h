// 测速：多滞后中值 + EWMA +  slew 限幅（与 Python SpeedEstimator 对齐）
#pragma once
#include <deque>
#include <map>
#include <vector>

#include "core.h"
#include "tracker.h"

namespace sfly {

struct SpeedState {
    float px = 0.f;      // 像素/帧
    bool ready = false;  // 是否已由区间法算出（False 时用卡尔曼速度兜底）
    Box box;
};

class SpeedEstimator {
public:
    SpeedEstimator(float k, float fps) : k_(k), fps_(fps) {}
    void update(const std::vector<Track>& tracks, int frame,
                std::map<int, SpeedState>& out, float min_speed_px = 0.2f);

private:
    struct Sample {
        int frame;
        float cx, cy;
    };
    float k_, fps_;
    std::map<int, std::deque<Sample>> hist_;
    std::map<int, float> ewma_;
};

}  // namespace sfly