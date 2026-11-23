#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <numeric>
#include <tuple>

namespace sfly {

// =====================================================================
// 小矩阵
// =====================================================================
void mat_mul(const double* a, const double* b, double* out, int n, int k, int m) {
    // a: n x k, b: k x m, out: n x m
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int t = 0; t < k; ++t) s += a[i * k + t] * b[t * m + j];
            out[i * m + j] = s;
        }
    }
}

void mat_add(const double* a, const double* b, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void mat_inv4(const double* a, double* out) {
    double m[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) m[i][j] = a[i * 4 + j];
        for (int j = 0; j < 4; ++j) m[i][4 + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) piv = r;
        if (std::fabs(m[piv][col]) < 1e-12) {   // 奇异：退化为零矩阵
            for (int i = 0; i < 16; ++i) out[i] = 0.0;
            return;
        }
        if (piv != col)
            for (int j = 0; j < 8; ++j) std::swap(m[col][j], m[piv][j]);
        const double d = m[col][col];
        for (int j = 0; j < 8; ++j) m[col][j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = m[r][col];
            if (f == 0.0) continue;
            for (int j = 0; j < 8; ++j) m[r][j] -= f * m[col][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = m[i][4 + j];
}

// =====================================================================
// 卡尔曼
// =====================================================================
Kalman8::Kalman8() {
    std::fill(std::begin(F), std::end(F), 0.0);
    std::fill(std::begin(P), std::end(P), 0.0);
    std::fill(std::begin(Q), std::end(Q), 0.0);
    std::fill(std::begin(R), std::end(R), 0.0);
    std::fill(std::begin(H), std::end(H), 0.0);
    for (int i = 0; i < 8; ++i) {
        F[i * 8 + i] = 1.0;
        P[i * 8 + i] = 10.0;
        Q[i * 8 + i] = 1e-2;
    }
    F[0 * 8 + 4] = 1.0;   // cx += vx
    F[1 * 8 + 5] = 1.0;   // cy += vy
    F[2 * 8 + 6] = 1.0;   // w  += vw
    F[3 * 8 + 7] = 1.0;   // h  += vh
    for (int i = 0; i < 4; ++i) {
        H[i * 8 + i] = 1.0;
        R[i * 4 + i] = 1e-1;
    }
}

void Kalman8::init(const Box& b) {
    x[0] = 0.5 * (b.x1 + b.x2);
    x[1] = 0.5 * (b.y1 + b.y2);
    x[2] = b.x2 - b.x1;
    x[3] = b.y2 - b.y1;
    x[4] = x[5] = x[6] = x[7] = 0.0;
    std::fill(std::begin(P), std::end(P), 0.0);
    for (int i = 0; i < 8; ++i) P[i * 8 + i] = 10.0;
}

Box Kalman8::predict() {
    double nx[8];
    mat_mul(F, x, nx, 8, 8, 1);
    std::copy(nx, nx + 8, x);
    double FP[64], FPFt[64], Ft[64], Pn[64];
    mat_mul(F, P, FP, 8, 8, 8);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) Ft[i * 8 + j] = F[j * 8 + i];
    mat_mul(FP, Ft, FPFt, 8, 8, 8);
    mat_add(FPFt, Q, Pn, 64);
    std::copy(Pn, Pn + 64, P);
    const double cx = x[0], cy = x[1], w = x[2], h = x[3];
    Box out;
    out.x1 = static_cast<float>(cx - w / 2);
    out.y1 = static_cast<float>(cy - h / 2);
    out.x2 = static_cast<float>(cx + w / 2);
    out.y2 = static_cast<float>(cy + h / 2);
    return out;
}

void Kalman8::update(const Box& b) {
    const double z[4] = {0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2),
                         b.x2 - b.x1, b.y2 - b.y1};
    double Hx[4];
    mat_mul(H, x, Hx, 4, 8, 1);
    double y[4];
    for (int i = 0; i < 4; ++i) y[i] = z[i] - Hx[i];

    // S = H P H^T + R
    double Ht[64];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) Ht[i * 8 + j] = H[i * 8 + j];
    double PHt[32], S[16];
    // PHt = P (8x8) * H^T (8x4)
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int t = 0; t < 8; ++t) s += P[i * 8 + t] * Ht[j * 8 + t];
            PHt[i * 4 + j] = s;
        }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int t = 0; t < 8; ++t) s += H[i * 8 + t] * PHt[t * 4 + j];
            S[i * 4 + j] = s + R[i * 4 + j];
        }
    double Sinv[16];
    mat_inv4(S, Sinv);
    // K = PHt (8x4) * Sinv (4x4) -> 8x4
    double K[32];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int t = 0; t < 4; ++t) s += PHt[i * 4 + t] * Sinv[t * 4 + j];
            K[i * 4 + j] = s;
        }
    for (int i = 0; i < 8; ++i) {
        double s = 0.0;
        for (int t = 0; t < 4; ++t) s += K[i * 4 + t] * y[t];
        x[i] += s;
    }
    // P = (I - K H) P
    double KH[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double s = 0.0;
            for (int t = 0; t < 4; ++t) s += K[i * 4 + t] * H[t * 8 + j];
            KH[i * 8 + j] = s;
        }
    double IKH[64], Pn[64];
    for (int i = 0; i < 64; ++i)
        IKH[i] = ((i / 8) == (i % 8) ? 1.0 : 0.0) - KH[i];
    mat_mul(IKH, P, Pn, 8, 8, 8);
    std::copy(Pn, Pn + 64, P);
}

// =====================================================================
// 轨迹
// =====================================================================
Track::Track(const Box& b, float s) : box(b), score(s) { kf.init(b); }

void Track::predict() {
    box = kf.predict();
    ++tsu;
}

void Track::update(const Box& b, float s) {
    kf.update(b);
    const double cx = kf.x[0], cy = kf.x[1], w = kf.x[2], h = kf.x[3];
    box.x1 = static_cast<float>(cx - w / 2);
    box.y1 = static_cast<float>(cy - h / 2);
    box.x2 = static_cast<float>(cx + w / 2);
    box.y2 = static_cast<float>(cy + h / 2);
    score = s;
    ++hits;
    tsu = 0;
}

// =====================================================================
// 匹配
// =====================================================================
std::vector<int> hungarian(const std::vector<double>& cost, int n, int m) {
    const double BIG = 1e9;
    std::vector<double> c(n * m);
    for (int i = 0; i < n * m; ++i)
        c[i] = std::isnan(cost[i]) ? BIG : cost[i];
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0), minv(m + 1);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    std::vector<char> used(m + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), BIG);
        std::fill(used.begin(), used.end(), 0);
        int iters = 0;
        for (; iters < m + 2; ++iters) {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = BIG;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = c[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            if (j1 == 0) break;
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        if (iters >= m + 2) continue;
        for (int rec = 0; rec < m + 2; ++rec) {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }
    }
    std::vector<int> ans(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j] > 0) ans[p[j] - 1] = j - 1;
    return ans;
}

std::vector<std::pair<int, int>> greedy_assign(const std::vector<double>& cost,
                                               int n, int m) {
    std::vector<double> c(cost);
    for (auto& x : c) if (std::isnan(x)) x = std::numeric_limits<double>::infinity();
    std::vector<char> row_used(n, 0), col_used(m, 0);
    std::vector<std::pair<int, int>> out;
    for (int step = 0; step < std::min(n, m); ++step) {
        double best = std::numeric_limits<double>::infinity();
        int br = -1, bc = -1;
        for (int i = 0; i < n; ++i) {
            if (row_used[i]) continue;
            for (int j = 0; j < m; ++j) {
                if (col_used[j]) continue;
                const double val = c[i * m + j];
                if (val < best) { best = val; br = i; bc = j; }
            }
        }
        if (br < 0 || !std::isfinite(best)) break;
        out.emplace_back(br, bc);
        row_used[br] = 1;
        col_used[bc] = 1;
    }
    return out;
}

std::vector<std::pair<int, int>> gated_assign(const std::vector<double>& cost,
                                              int n_tr, int n_det,
                                              int top_k, int exact_max) {
    std::vector<std::pair<int, int>> out;
    if (n_tr <= 0 || n_det <= 0) return out;

    auto at = [&](int i, int j) { return cost[i * n_det + j]; };
    auto valid = [&](int i, int j) { return !std::isnan(at(i, j)); };

    // 1) 每条轨迹保留 top_k 个候选
    std::vector<double> pruned(cost.size(), std::numeric_limits<double>::quiet_NaN());
    for (int ti = 0; ti < n_tr; ++ti) {
        std::vector<int> cols;
        for (int dj = 0; dj < n_det; ++dj)
            if (valid(ti, dj)) cols.push_back(dj);
        if (cols.empty()) continue;
        if (static_cast<int>(cols.size()) > top_k) {
            std::partial_sort(cols.begin(), cols.begin() + top_k, cols.end(),
                              [&](int a, int b) { return at(ti, a) < at(ti, b); });
            cols.resize(top_k);
        }
        for (int dj : cols) pruned[ti * n_det + dj] = cost[ti * n_det + dj];
    }

    // 2) 并查集分组
    std::vector<int> parent(n_tr + n_det);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    bool any = false;
    for (int ti = 0; ti < n_tr; ++ti)
        for (int dj = 0; dj < n_det; ++dj) {
            if (std::isnan(pruned[ti * n_det + dj])) continue;
            any = true;
            const int ra = find(ti), rb = find(n_tr + dj);
            if (ra != rb) parent[rb] = ra;
        }
    if (!any) return out;

    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> groups;
    for (int ti = 0; ti < n_tr; ++ti) {
        bool has = false;
        for (int dj = 0; dj < n_det; ++dj) if (!std::isnan(pruned[ti * n_det + dj])) { has = true; break; }
        if (has) groups[find(ti)].first.push_back(ti);
    }
    for (int dj = 0; dj < n_det; ++dj) {
        bool has = false;
        for (int ti = 0; ti < n_tr; ++ti) if (!std::isnan(pruned[ti * n_det + dj])) { has = true; break; }
        if (has) groups[find(n_tr + dj)].second.push_back(dj);
    }

    // 3) 组内求解
    for (auto& [root, g] : groups) {
        const auto& rows = g.first;
        const auto& cols = g.second;
        const int gn = static_cast<int>(rows.size()), gm = static_cast<int>(cols.size());
        std::vector<double> sub(gn * gm, std::numeric_limits<double>::quiet_NaN());
        for (int i = 0; i < gn; ++i)
            for (int j = 0; j < gm; ++j)
                sub[i * gm + j] = pruned[rows[i] * n_det + cols[j]];
        if (std::max(gn, gm) <= exact_max) {
            const auto ans = hungarian(sub, gn, gm);
            for (int i = 0; i < gn; ++i)
                if (ans[i] >= 0) out.emplace_back(rows[i], cols[ans[i]]);
        } else {
            for (auto& pr : greedy_assign(sub, gn, gm))
                out.emplace_back(rows[pr.first], cols[pr.second]);
        }
    }
    return out;
}

// =====================================================================
// Tracker
// =====================================================================
Tracker::Tracker(int min_hits_, int max_age_, double gate_, int display_coast_)
    : min_hits(min_hits_), max_age(max_age_), gate(gate_), display_coast(display_coast_) {}

void Tracker::seed(const std::vector<std::tuple<int, Box, float, float, float>>& items) {
    tracks.clear();
    dead.clear();
    int top = 0;
    for (const auto& it : items) {
        const int tid = std::get<0>(it);
        Track tk(std::get<1>(it), std::get<2>(it));
        tk.kf.x[4] = -std::get<3>(it);   // 回溯是倒着跑的，速度取负
        tk.kf.x[5] = -std::get<4>(it);
        tk.pid = tid;
        tk.hits = std::max(min_hits, 1);
        tk.tsu = 0;
        tracks.push_back(tk);
        top = std::max(top, tid);
    }
    next_id = top + 1;
    frame = 0;
}

void Tracker::coast() {
    ++frame;
    for (auto& t : tracks) t.predict();
    std::vector<Track> alive;
    for (auto& t : tracks) if (t.tsu <= max_age) alive.push_back(t);
    tracks.swap(alive);
}

std::vector<int> Tracker::update(const std::vector<Det>& dets, const float* descs, int n_desc,
                            const float* flows, int n_flow) {
    ++frame;
    for (auto& t : tracks) t.predict();
    std::vector<char> det_matched(dets.size(), 0);

    if (!tracks.empty() && !dets.empty()) {
        const int n_tr = static_cast<int>(tracks.size());
        const int n_det = static_cast<int>(dets.size());
        std::vector<double> cost(n_tr * n_det);
        for (int i = 0; i < n_tr; ++i) {
            const Track& tk = tracks[i];
            double pcx = 0.5 * (tk.box.x1 + tk.box.x2);
            double pcy = 0.5 * (tk.box.y1 + tk.box.y2);
            const double pw = std::max<double>(tk.box.x2 - tk.box.x1, 1.0);
            const double ph = std::max<double>(tk.box.y2 - tk.box.y1, 1.0);
            const double vx = tk.kf.x[4], vy = tk.kf.x[5];
            // 光流先验（docs/14 E10）：用块匹配测到的位移修正"预测中心"
            if (flow_w > 0.f && flows && (i + 1) * 2 <= n_flow) {
                const double fx = flows[i * 2], fy = flows[i * 2 + 1];
                if (std::fabs(fx) + std::fabs(fy) > 0.01) {
                    pcx += flow_w * fx;
                    pcy += flow_w * fy;
                }
            }
            const double speed = std::sqrt(vx * vx + vy * vy);
            const double base = std::max(0.5 * (pw + ph) / 2.0, 3.0);
            for (int j = 0; j < n_det; ++j) {
                const double dcx = 0.5 * (dets[j].b.x1 + dets[j].b.x2);
                const double dcy = 0.5 * (dets[j].b.y1 + dets[j].b.y2);
                const double dx = dcx - pcx, dy = dcy - pcy;
                const double dist = std::sqrt(dx * dx + dy * dy);
                const double along = (speed > 1e-6) ? (dx * vx + dy * vy) / std::max(speed, 1e-6) : 0.0;
                const double across = std::sqrt(std::max(dist * dist - along * along, 0.0));
                const double a = base * (4.5 + 0.6 * speed);
                const double b = base * 2.2;
                const double norm_e = std::sqrt((along / a) * (along / a) + (across / b) * (across / b));
                const double gate_c = std::max(10.0, 3.0 * base);
                const double norm = (speed > 1.0) ? norm_e : dist / gate_c;
                const double dw = std::max<double>(dets[j].b.x2 - dets[j].b.x1, 1.0);
                const double dh = std::max<double>(dets[j].b.y2 - dets[j].b.y1, 1.0);
                double c = norm + 0.25 * (std::fabs(std::log(pw / dw)) + std::fabs(std::log(ph / dh)));
                c += 0.3 * (1.0 - std::clamp(static_cast<double>(dets[j].s), 0.0, 1.0));
                // 外观项：L1 距离（0~1），只在候选本身已经过门限时才加
                if (app_w > 0.f && tk.has_desc && descs && (j + 1) * 8 <= n_desc) {
                    const float* d = descs + j * 8;
                    double dist_h = 0.0;
                    for (int k = 0; k < 8; ++k) dist_h += std::fabs(static_cast<double>(d[k]) - tk.adesc[k]);
                    c += app_w * 0.5 * dist_h;
                }
                cost[i * n_det + j] = (norm > gate) ? std::numeric_limits<double>::quiet_NaN() : c;
            }
        }
        for (auto& pr : gated_assign(cost, n_tr, n_det)) {
            tracks[pr.first].update(dets[pr.second].b, dets[pr.second].s);
            det_matched[pr.second] = 1;
            // 外观 EMA：匹配上就用检测的直方图更新轨迹外观
            if (descs && (pr.second + 1) * 8 <= n_desc) {
                const float* d = descs + pr.second * 8;
                Track& tk2 = tracks[pr.first];
                if (!tk2.has_desc) {
                    for (int k = 0; k < 8; ++k) tk2.adesc[k] = d[k];
                    tk2.has_desc = true;
                } else {
                    for (int k = 0; k < 8; ++k)
                        tk2.adesc[k] = 0.9f * tk2.adesc[k] + 0.1f * d[k];
                }
            }
        }
    }

    // 未匹配的检测：先尝试复活，再新建
    for (size_t j = 0; j < dets.size(); ++j) {
        if (det_matched[j]) continue;
        if (!revive(dets[j].b, dets[j].s)) tracks.emplace_back(dets[j].b, dets[j].s);
    }

    for (auto& t : tracks)
        if (t.pid < 0 && t.hits >= min_hits) { t.pid = next_id++; }

    std::vector<Track> alive;
    for (auto& t : tracks) {
        if (t.tsu > max_age) {
            if (t.pid >= 0) dead.push_back(Dead{t.pid, t.box, static_cast<float>(t.kf.x[4]),
                                                static_cast<float>(t.kf.x[5]), frame});
        } else {
            alive.push_back(t);
        }
    }
    tracks.swap(alive);
    while (!dead.empty() && frame - dead.front().frame > 12) dead.pop_front();
    while (dead.size() > 64) dead.pop_front();

    std::vector<int> shown;
    for (size_t i = 0; i < tracks.size(); ++i)
        if (tracks[i].pid >= 0 && tracks[i].tsu <= display_coast) shown.push_back(static_cast<int>(i));
    return shown;
}

std::vector<int> Tracker::shown() const {
    std::vector<int> out;
    for (size_t i = 0; i < tracks.size(); ++i)
        if (tracks[i].pid >= 0 && tracks[i].tsu <= display_coast) out.push_back(static_cast<int>(i));
    return out;
}

bool Tracker::revive(const Box& box, float score) {
    if (dead.empty()) return false;
    const double cx = 0.5 * (box.x1 + box.x2), cy = 0.5 * (box.y1 + box.y2);
    int best = -1;
    double best_d = 12.0;
    for (size_t k = 0; k < dead.size(); ++k) {
        const int age = frame - dead[k].frame;
        if (age <= 0 || age > 12) continue;
        const double dcx = 0.5 * (dead[k].box.x1 + dead[k].box.x2) + dead[k].vx * age;
        const double dcy = 0.5 * (dead[k].box.y1 + dead[k].box.y2) + dead[k].vy * age;
        const double d = std::hypot(cx - dcx, cy - dcy);
        if (d < best_d) { best = static_cast<int>(k); best_d = d; }
    }
    if (best < 0) return false;
    const Dead d = dead[best];
    dead.erase(dead.begin() + best);
    Track t(box, score);
    t.pid = d.pid;
    t.hits = std::max(t.hits, min_hits);
    t.kf.x[4] = d.vx * 0.7;
    t.kf.x[5] = d.vy * 0.7;
    tracks.push_back(t);
    return true;
}

}  // namespace sfly