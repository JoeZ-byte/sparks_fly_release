#include "motion.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sfly {

namespace {
// 与 cv2.resize(INTER_AREA) 等价的面积平均（float 输入，任意缩放比）
void resize_area_float(const std::vector<float>& src, int sw, int sh,
                       std::vector<float>& dst, int dw, int dh) {
    dst.assign(static_cast<size_t>(dw) * dh, 0.f);
    // 快路径：整数倍缩小 = 方块求和（INTER_AREA 在整数倍时就是这个）
    if (sw % dw == 0 && sh % dh == 0) {
        const int bx = sw / dw, by = sh / dh;
        const double inv = 1.0 / (static_cast<double>(bx) * by);
        for (int y = 0; y < dh; ++y) {
            for (int x = 0; x < dw; ++x) {
                double acc = 0.0;
                for (int yy = 0; yy < by; ++yy) {
                    const float* row = src.data() + static_cast<size_t>(y * by + yy) * sw + x * bx;
                    for (int xx = 0; xx < bx; ++xx) acc += row[xx];
                }
                dst[static_cast<size_t>(y) * dw + x] = static_cast<float>(acc * inv);
            }
        }
        return;
    }
    const double rx = static_cast<double>(sw) / dw;
    const double ry = static_cast<double>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        const double y0 = y * ry, y1 = std::min<double>((y + 1) * ry, sh);
        for (int x = 0; x < dw; ++x) {
            const double x0 = x * rx, x1 = std::min<double>((x + 1) * rx, sw);
            double acc = 0.0, wsum = 0.0;
            for (int sy = static_cast<int>(std::floor(y0)); sy < static_cast<int>(std::ceil(y1)); ++sy) {
                if (sy < 0 || sy >= sh) continue;
                const double wy = std::min<double>(y1, sy + 1.0) - std::max<double>(y0, sy);
                if (wy <= 0) continue;
                for (int sx = static_cast<int>(std::floor(x0)); sx < static_cast<int>(std::ceil(x1)); ++sx) {
                    if (sx < 0 || sx >= sw) continue;
                    const double wx = std::min<double>(x1, sx + 1.0) - std::max<double>(x0, sx);
                    if (wx <= 0) continue;
                    const double wgt = wx * wy;
                    acc += static_cast<double>(src[static_cast<size_t>(sy) * sw + sx]) * wgt;
                    wsum += wgt;
                }
            }
            dst[static_cast<size_t>(y) * dw + x] = static_cast<float>(wsum > 0 ? acc / wsum : 0.0);
        }
    }
}
}  // namespace

MotionProposer::MotionProposer(int w_, int h_, float thr_, int max_cells_, float alpha_)
    : w(w_), h(h_), thr(thr_), max_cells(max_cells_), alpha(alpha_) {
    nx = std::max(1, (w + kTile - 1) / kTile);
    ny = std::max(1, (h + kTile - 1) / kTile);
    last_seen.assign(static_cast<size_t>(nx) * ny, 0.f);
}

void MotionProposer::update_bg(const GrayImage& gray) {
    // 关键：cv2.resize(uint8, INTER_AREA) 出来仍是 uint8（带四舍五入），
    // 再 astype(float32) 才进 accumulateWeighted。直接用浮点均值会有 ±0.5 的偏差，
    // 累计到背景模型后帧差就会明显不同 —— 这里必须先落 uint8。
    const GrayImage half = resize_area_half(gray);
    const size_t n = half.p.size();
    if (bg.empty() || bg.size() != n) {
        bg.resize(n);
        for (size_t i = 0; i < n; ++i) bg[i] = static_cast<float>(half.p[i]);
        return;
    }
    for (size_t i = 0; i < n; ++i)
        bg[i] = static_cast<float>((1.0 - alpha) * bg[i] + alpha * static_cast<double>(half.p[i]));
}

GrayImage MotionProposer::mask(const GrayImage& gray) {
    const GrayImage small = resize_area_half(gray);
    const int sw = small.w, sh = small.h;
    if (bg.empty()) {
        bg.resize(small.p.size());
        for (size_t i = 0; i < small.p.size(); ++i) bg[i] = static_cast<float>(small.p[i]);
        return GrayImage(sw, sh);
    }
    // diff = |small - uint8(bg)|（截断，与 numpy astype(uint8) 一致）
    GrayImage bg8(sw, sh);
    for (size_t i = 0; i < small.p.size(); ++i) {
        const float v = bg[i];
        bg8.p[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : static_cast<int>(v)));
    }
    const GrayImage diff = absdiff(small, bg8);
    const GrayImage m = threshold_gt(diff, static_cast<int>(thr));
    return morph_close_ellipse3(m);
}

void MotionProposer::update_bg_half(const GrayImage& half) {
    update_bg_prepared(pre_mode ? enhance(half, pre_mode) : half);
}

GrayImage MotionProposer::mask_half(const GrayImage& half) {
    return mask_prepared(pre_mode ? enhance(half, pre_mode) : half);
}

// 以下两个 _prepared 版本假定输入已经过增强，供 propose_half 复用同一份中间结果
void MotionProposer::update_bg_prepared(const GrayImage& half) {
    // 滚动前两帧缓存（三帧差分用）
    prev2 = prev1;
    prev1 = half;
    const size_t n = half.p.size();
    if (bg.empty() || bg.size() != n) {
        bg.resize(n);
        for (size_t i = 0; i < n; ++i) bg[i] = static_cast<float>(half.p[i]);
        return;
    }
    for (size_t i = 0; i < n; ++i)
        bg[i] = static_cast<float>((1.0 - alpha) * bg[i] + alpha * static_cast<double>(half.p[i]));
}

GrayImage MotionProposer::mask_prepared(const GrayImage& half) {
    // 三帧差分（docs/14 E5）：|It-It1| 与 |It1-It2| 都超过阈值才算运动
    if (diff_mode == 1 && prev1.w == half.w && prev2.w == half.w) {
        const GrayImage d1 = absdiff(half, prev1);
        const GrayImage d2 = absdiff(prev1, prev2);
        GrayImage m(half.w, half.h);
        for (size_t i = 0; i < m.p.size(); ++i)
            m.p[i] = (d1.p[i] > static_cast<int>(thr) && d2.p[i] > static_cast<int>(thr))
                         ? static_cast<uint8_t>(255)
                         : static_cast<uint8_t>(0);
        return morph_close_ellipse3(m);
    }
    if (bg.empty()) {
        bg.resize(half.p.size());
        for (size_t i = 0; i < half.p.size(); ++i) bg[i] = static_cast<float>(half.p[i]);
        return GrayImage(half.w, half.h);
    }
    GrayImage bg8(half.w, half.h);
    for (size_t i = 0; i < half.p.size(); ++i) {
        const float v = bg[i];
        bg8.p[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : static_cast<int>(v)));
    }
    const GrayImage diff = absdiff(half, bg8);
    GrayImage m_ema = threshold_gt(diff, static_cast<int>(thr));

    // 背景模型融合（docs/14 E6）
    if (bg_mode >= 1) {
        const GrayImage m_vibe = vibe_mask_prepared(half);
        if (bg_mode == 1) {
            m_ema = m_vibe;
        } else if (bg_mode == 2) {          // 并集：任一判前景即前景
            for (size_t i = 0; i < m_ema.p.size(); ++i)
                m_ema.p[i] = (m_ema.p[i] > 0 || m_vibe.p[i] > 0) ? 255 : 0;
        } else if (bg_mode == 3) {          // 交集：都判前景才算
            for (size_t i = 0; i < m_ema.p.size(); ++i)
                m_ema.p[i] = (m_ema.p[i] > 0 && m_vibe.p[i] > 0) ? 255 : 0;
        }
    }
    return morph_close_ellipse3(m_ema);
}

// ViBe：首帧把 N 个样本都填成当前值（之后靠随机替换演化）
void MotionProposer::vibe_fill(const GrayImage& half) {
    const size_t npx = half.p.size();
    vibe.assign(npx * static_cast<size_t>(vibe_n), 0);
    for (size_t i = 0; i < npx; ++i)
        for (int k = 0; k < vibe_n; ++k) vibe[i * vibe_n + k] = half.p[i];
}

GrayImage MotionProposer::vibe_mask_prepared(const GrayImage& half) {
    const size_t npx = half.p.size();
    if (vibe.size() != npx * static_cast<size_t>(vibe_n)) vibe_fill(half);
    GrayImage m(half.w, half.h);
    for (size_t i = 0; i < npx; ++i) {
        const uint8_t v = half.p[i];
        uint8_t* model = vibe.data() + i * vibe_n;
        int best = 255;
        for (int k = 0; k < vibe_n; ++k) {
            const int d = std::abs(static_cast<int>(v) - static_cast<int>(model[k]));
            if (d < best) best = d;
            if (best <= vibe_r) break;
        }
        if (best <= vibe_r) {
            m.p[i] = 0;                       // 背景：随机替换一个样本
            vibe_seed = vibe_seed * 1103515245u + 12345u;
            if ((vibe_seed >> 16) % static_cast<unsigned>(vibe_update_div) == 0)
                model[(vibe_seed >> 8) % static_cast<unsigned>(vibe_n)] = v;
        } else {
            m.p[i] = 255;                     // 前景
        }
    }
    return m;
}

void MotionProposer::propose_half(const GrayImage& half, const std::vector<Box>& track_boxes,
                                  std::vector<std::pair<int, int>>& origins, float track_bonus) {
    // 小目标增强（白顶帽 / DoG）只在提案这一层做：增强一次，掩膜与背景共用
    const GrayImage enh = pre_mode ? enhance(half, pre_mode) : half;
    const GrayImage m = mask_prepared(enh);
    std::vector<int64_t> cell;
    cells(m, cell);
    std::vector<float> scores(cell.size());
    float mx = 0.f;
    for (size_t i = 0; i < cell.size(); ++i) {
        scores[i] = static_cast<float>(cell[i]);
        mx = std::max(mx, scores[i]);
    }
    if (mx > 0) for (auto& v : scores) v /= mx;
    for (const auto& b : track_boxes) {
        const int cx = std::clamp(static_cast<int>((b.x1 + b.x2) * 0.5f), 0, w - 1) / kTile;
        const int cy = std::clamp(static_cast<int>((b.y1 + b.y2) * 0.5f), 0, h - 1) / kTile;
        if (cy < ny && cx < nx) scores[static_cast<size_t>(cy) * nx + cx] += track_bonus;
    }
    ++frame;
    for (size_t i = 0; i < scores.size(); ++i) {
        const float stale = std::clamp((frame - last_seen[i]) / 8.0f, 0.f, 1.5f);
        scores[i] += 0.9f * stale;
    }
    const int k = std::min(max_cells, static_cast<int>(scores.size()));
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    origins.clear();
    for (int i = 0; i < k; ++i) {
        const int idx = order[i];
        origins.emplace_back((idx % nx) * kTile, (idx / nx) * kTile);
        last_seen[idx] = static_cast<float>(frame);
    }
    update_bg_prepared(enh);
}

void MotionProposer::cells(const GrayImage& m, std::vector<int64_t>& out) const {
    std::vector<float> src(static_cast<size_t>(m.w) * m.h);
    for (size_t i = 0; i < m.p.size(); ++i) src[i] = (m.p[i] > 0) ? 1.f : 0.f;
    std::vector<float> dst;
    resize_area_float(src, m.w, m.h, dst, nx, ny);
    out.assign(static_cast<size_t>(nx) * ny, 0);
    const double k = static_cast<double>(m.h) / ny * static_cast<double>(m.w) / nx;
    for (size_t i = 0; i < dst.size(); ++i)
        out[i] = static_cast<int64_t>(std::nearbyint(dst[i] * k));   // numpy.rint 同银行家舍入
}

void MotionProposer::propose(const GrayImage& gray, const std::vector<Box>& track_boxes,
                             std::vector<std::pair<int, int>>& origins, float track_bonus) {
    const GrayImage m = mask(gray);
    std::vector<int64_t> cell;
    cells(m, cell);
    std::vector<float> scores(cell.size());
    float mx = 0.f;
    for (size_t i = 0; i < cell.size(); ++i) {
        scores[i] = static_cast<float>(cell[i]);
        mx = std::max(mx, scores[i]);
    }
    if (mx > 0) for (auto& v : scores) v /= mx;
    for (const auto& b : track_boxes) {
        const int cx = std::clamp(static_cast<int>((b.x1 + b.x2) * 0.5f), 0, w - 1) / kTile;
        const int cy = std::clamp(static_cast<int>((b.y1 + b.y2) * 0.5f), 0, h - 1) / kTile;
        if (cy < ny && cx < nx) scores[static_cast<size_t>(cy) * nx + cx] += track_bonus;
    }
    ++frame;
    for (size_t i = 0; i < scores.size(); ++i) {
        const float stale = std::clamp((frame - last_seen[i]) / 8.0f, 0.f, 1.5f);
        scores[i] += 0.9f * stale;
    }
    const int k = std::min(max_cells, static_cast<int>(scores.size()));
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    origins.clear();
    for (int i = 0; i < k; ++i) {
        const int idx = order[i];
        origins.emplace_back((idx % nx) * kTile, (idx / nx) * kTile);
        last_seen[idx] = static_cast<float>(frame);
    }
    update_bg(gray);
}

void MotionProposer::warmup_grid(int stride, std::vector<std::pair<int, int>>& origins) const {
    origins.clear();
    const int st = std::max(1, stride);
    for (int y = 0; y < ny; y += st)
        for (int x = 0; x < nx; x += st) origins.emplace_back(x * kTile, y * kTile);
}

}  // namespace sfly