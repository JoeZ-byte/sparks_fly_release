#include "speed.h"

#include <algorithm>
#include <cmath>

namespace sfly {

namespace {
float median3(float a, float b, float c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}
}  // namespace

void SpeedEstimator::update(const std::vector<Track>& tracks, int frame,
                            std::map<int, SpeedState>& out, float /*min_speed_px*/) {
    out.clear();
    for (const auto& t : tracks) {
        const int tid = t.pid;
        if (tid < 0) continue;
        const float cx = 0.5f * (t.box.x1 + t.box.x2);
        const float cy = 0.5f * (t.box.y1 + t.box.y2);
        auto& h = hist_[tid];
        h.push_back(Sample{frame, cx, cy});
        while (h.size() > 64) h.pop_front();

        float px = -1.f;
        if (h.size() >= 4) {
            float vals[3];
            int nv = 0;
            for (int lag : {6, 12, 18}) {
                const Sample* best = nullptr;
                int best_gap = 1 << 30;
                for (const auto& q : h) {
                    const int d = std::abs(q.frame - (frame - lag));
                    if (d <= 6 && frame - q.frame >= 3 && d < best_gap) {
                        best = &q;
                        best_gap = d;
                    }
                }
                if (best) {
                    const int gap = frame - best->frame;
                    if (gap > 0) {
                        vals[nv++] = std::hypot(cx - best->cx, cy - best->cy) / static_cast<float>(gap);
                    }
                }
            }
            if (nv > 0) {
                px = (nv == 1) ? vals[0] : (nv == 2 ? 0.5f * (vals[0] + vals[1]) : median3(vals[0], vals[1], vals[2]));
                auto it = ewma_.find(tid);
                const bool has_prev = it != ewma_.end();
                const float prev = has_prev ? it->second : 0.f;
                if (has_prev) px = 0.7f * px + 0.3f * prev;
                px = std::min(px, 6.f);
                if (has_prev) {
                    const float slew = 35.f / std::max(1e-6f, 3.6f * std::max(k_, 1e-6f) * fps_);
                    px = std::clamp(px, std::max(0.f, prev - slew), prev + slew);
                }
                ewma_[tid] = px;
            }
        }
        SpeedState st;
        if (px >= 0.f) {
            st.px = px;
            st.ready = true;
        } else {
            st.px = std::hypot(t.kf.x[4], t.kf.x[5]);
            st.ready = false;
        }
        st.box = t.box;
        out[tid] = st;
    }
}

}  // namespace sfly