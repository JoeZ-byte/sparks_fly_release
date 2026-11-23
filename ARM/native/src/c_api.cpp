/* C 接口的实现（C++ 侧）：把已通过 parity 测试的内核包成一个不透明句柄。
 *
 * 这里只做"串流程"，不含任何算法：
 *   半分辨率灰度 -> 运动提案 -> 图块预处理 -> ORT 推理 -> 解码 -> NMS
 *   -> 卡尔曼跟踪 -> 测速 -> 输出结构体
 */
#include "c_api.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core.h"
#include "decode.h"
#include "img.h"
#include "motion.h"
#include "runtime.h"
#include "speed.h"
#include "tracker.h"

struct SflyCtx {
    sfly::Session sess;
    std::vector<float> anchors, strides;
    sfly::MotionProposer* prop = nullptr;
    sfly::Tracker* trk = nullptr;
    sfly::SpeedEstimator* speed = nullptr;

    int imgsz = 320;
    int max_cells = 6;
    int warmup = 0;
    int max_coast = 8;
    float conf = 0.2f;
    int frame = 0;

    std::vector<float> blob;
    std::vector<sfly::Det> dets;
    std::vector<sfly::Track> shown_tracks;
    std::map<int, sfly::SpeedState> states;
    std::vector<SflyTrack> out_tracks;
    std::vector<SflyOrigin> out_origins;
    int pre_mode = 0;
    int min_hits = 3;
    float gate = 1.0f;
    float app_w = 0.0f;
    int diff_mode = 0;
    int bg_mode = 0;
    float flow_w = 0.0f;
    std::vector<float> flows;   // n_track x 2
    sfly::GrayImage prev_half;  // 上一帧半分辨率灰度（块匹配用）
    std::vector<float> descs;   // n_det x 8，外观描述子（docs/14 E8）

    SflyStats stats{};
};

namespace {

void set_err(char* err, int len, const std::string& msg) {
    if (!err || len <= 0) return;
    const int n = std::min<int>(len - 1, static_cast<int>(msg.size()));
    std::memcpy(err, msg.data(), static_cast<size_t>(n));
    err[n] = '\0';
}

}  // namespace

extern "C" SFLY_API const char* sfly_version(void) { return "satstream-c 1.0"; }

extern "C" SFLY_API SflyCtx* sfly_create(const SflyConfig* cfg, char* err, int err_len) {
    if (!cfg || !cfg->model_path) {
        set_err(err, err_len, "cfg 或 model_path 为空");
        return nullptr;
    }
    auto* ctx = new (std::nothrow) SflyCtx();
    if (!ctx) {
        set_err(err, err_len, "内存不足");
        return nullptr;
    }
    ctx->imgsz = cfg->imgsz > 0 ? cfg->imgsz : 320;
    ctx->max_cells = cfg->max_cells > 0 ? cfg->max_cells : 6;
    ctx->warmup = std::max(0, cfg->warmup);
    ctx->max_coast = cfg->max_coast > 0 ? cfg->max_coast : 8;
    ctx->conf = cfg->conf > 0 ? cfg->conf : 0.2f;
    ctx->pre_mode = cfg->pre_mode;   /* 0=关 1=白顶帽 2=DoG */
    ctx->min_hits = cfg->min_hits > 0 ? cfg->min_hits : 3;
    ctx->gate = cfg->gate > 0.f ? cfg->gate : 1.0f;
    ctx->app_w = cfg->app_w > 0.f ? cfg->app_w : 0.f;
    ctx->diff_mode = cfg->diff_mode;
    ctx->bg_mode = cfg->bg_mode;
    ctx->flow_w = cfg->flow_w > 0.f ? cfg->flow_w : 0.f;

    std::string e;
    if (!sfly::ort_init(cfg->ort_lib ? cfg->ort_lib : "", &e)) {
        set_err(err, err_len, "ORT 初始化失败: " + e);
        delete ctx;
        return nullptr;
    }
    if (!sfly::ort_create_session(cfg->model_path, cfg->threads > 0 ? cfg->threads : 4,
                                  ctx->sess, &e)) {
        set_err(err, err_len, "建会话失败: " + e);
        delete ctx;
        return nullptr;
    }
    sfly::make_anchors(ctx->imgsz, ctx->anchors, ctx->strides);
    ctx->blob.assign(static_cast<size_t>(3) * ctx->imgsz * ctx->imgsz * ctx->max_cells, 0.f);
    return ctx;
}

extern "C" SFLY_API void sfly_destroy(SflyCtx* ctx) {
    if (!ctx) return;
    sfly::ort_destroy(ctx->sess);
    delete ctx->prop;
    delete ctx->trk;
    delete ctx->speed;
    delete ctx;
}

extern "C" SFLY_API int sfly_step(SflyCtx* ctx, const uint8_t* bgr, int w, int h, SflyFrame* out,
                         char* err, int err_len) {
    if (!ctx || !bgr || !out || w <= 0 || h <= 0) {
        set_err(err, err_len, "参数非法");
        return -1;
    }
    // 首次调用才知道分辨率，这里再建运动提案/跟踪/测速（与 Python 版一致）
    if (!ctx->prop) {
        ctx->prop = new sfly::MotionProposer(w, h, 40.f, ctx->max_cells);
        ctx->prop->pre_mode = ctx->pre_mode;
        ctx->prop->diff_mode = ctx->diff_mode;
        ctx->prop->bg_mode = ctx->bg_mode;
        ctx->trk = new sfly::Tracker(ctx->min_hits, 15, ctx->gate, ctx->max_coast);
        ctx->trk->app_w = ctx->app_w;
        ctx->trk->flow_w = ctx->flow_w;
        ctx->speed = new sfly::SpeedEstimator(0.92f, 10.f);
    }
    ++ctx->frame;

    const auto t0 = std::chrono::steady_clock::now();
    const sfly::GrayImage half = sfly::gray_area_half(bgr, w, h);
    std::vector<std::pair<int, int>> origins;
    if (ctx->frame <= ctx->warmup) {
        ctx->prop->update_bg_half(half);
    } else {
        std::vector<sfly::Box> tb;
        tb.reserve(ctx->trk->tracks.size());
        for (const auto& t : ctx->trk->tracks) tb.push_back(t.box);
        ctx->prop->propose_half(half, tb, origins);
    }
    const auto t1 = std::chrono::steady_clock::now();

    ctx->dets.clear();
    if (!origins.empty()) {
        const int n = std::min<int>(static_cast<int>(origins.size()), ctx->max_cells);
        for (int i = 0; i < n; ++i) {
            const int x0 = std::max(0, std::min(w - sfly::kTile, origins[i].first));
            const int y0 = std::max(0, std::min(h - sfly::kTile, origins[i].second));
            origins[i] = {x0, y0};
            sfly::blob_from_bgr(bgr, w, h, x0, y0, sfly::kTile, sfly::kTile, ctx->imgsz,
                                ctx->blob.data() + static_cast<size_t>(i) * 3 * ctx->imgsz * ctx->imgsz);
        }
        const auto t2 = std::chrono::steady_clock::now();
        int64_t outc = 0, outa = 0;
        const float* pred = sfly::ort_run(ctx->sess, ctx->blob.data(), n, 3, ctx->imgsz,
                                          ctx->imgsz, outc, outa, nullptr);
        const auto t3 = std::chrono::steady_clock::now();
        ctx->stats.infer_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        if (pred) {
            const int A = static_cast<int>(outa);
            const int nc = static_cast<int>(outc) - 4 * sfly::kRegMax;
            const float k = static_cast<float>(sfly::kTile) / ctx->imgsz;
            for (int i = 0; i < n; ++i) {
                std::vector<sfly::Det> d;
                sfly::decode_topk(pred + static_cast<size_t>(i) * outc * outa, nc, A,
                                  ctx->anchors.data(), ctx->strides.data(), ctx->conf, d);
                for (auto& det : d) {
                    det.b.x1 = det.b.x1 * k + origins[i].first;
                    det.b.y1 = det.b.y1 * k + origins[i].second;
                    det.b.x2 = det.b.x2 * k + origins[i].first;
                    det.b.y2 = det.b.y2 * k + origins[i].second;
                    ctx->dets.push_back(det);
                }
            }
            if (!ctx->dets.empty()) {
                std::vector<sfly::Box> bs;
                std::vector<float> ss;
                bs.reserve(ctx->dets.size());
                ss.reserve(ctx->dets.size());
                for (const auto& d : ctx->dets) {
                    bs.push_back(d.b);
                    ss.push_back(d.s);
                }
                const std::vector<int> keep = sfly::nms(bs, ss, 0.5f);
                std::vector<sfly::Det> merged;
                merged.reserve(keep.size());
                for (int idx : keep) merged.push_back(ctx->dets[idx]);
                ctx->dets.swap(merged);
            }
        }
        ctx->stats.tiles_total += n;
    }
    ctx->stats.dets_total += static_cast<long long>(ctx->dets.size());

    const auto t4 = std::chrono::steady_clock::now();
    // 外观描述子（docs/14 E8）：检测框外扩 1px 的 8-bin 灰度直方图
    if (ctx->app_w > 0.f && !ctx->dets.empty()) {
        ctx->descs.assign(ctx->dets.size() * 8, 0.f);
        for (size_t di = 0; di < ctx->dets.size(); ++di) {
            const sfly::Box& b = ctx->dets[di].b;
            int x1 = std::max(0, static_cast<int>(b.x1) - 1);
            int y1 = std::max(0, static_cast<int>(b.y1) - 1);
            int x2 = std::min(w - 1, static_cast<int>(b.x2) + 1);
            int y2 = std::min(h - 1, static_cast<int>(b.y2) + 1);
            if (x2 <= x1) x2 = std::min(w - 1, x1 + 1);
            if (y2 <= y1) y2 = std::min(h - 1, y1 + 1);
            float hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int cnt = 0;
            for (int yy = y1; yy <= y2; ++yy) {
                const uint8_t* row = bgr + static_cast<size_t>(yy) * w * 3;
                for (int xx = x1; xx <= x2; ++xx) {
                    const int g = (row[xx * 3 + 2] * 77 + row[xx * 3 + 1] * 150 +
                                   row[xx * 3] * 29) >> 8;
                    hist[g >> 5] += 1.f;
                    ++cnt;
                }
            }
            if (cnt > 0)
                for (int k = 0; k < 8; ++k) ctx->descs[di * 8 + k] = hist[k] / cnt;
        }
    }
    // 光流先验（docs/14 E10）：对每条轨迹，用上一帧中心附近的块在当前帧做 SAD 搜索
    if (ctx->flow_w > 0.f && !ctx->prev_half.p.empty() && ctx->prev_half.w == half.w) {
        const int n_tr = static_cast<int>(ctx->trk->tracks.size());
        ctx->flows.assign(static_cast<size_t>(n_tr) * 2, 0.f);
        const int R = 6;          // 搜索半径（半分辨率像素）
        const int PW = 5;         // patch 半径
        for (int i = 0; i < n_tr; ++i) {
            const sfly::Box& b = ctx->trk->tracks[i].box;
            const int cx = static_cast<int>(0.5f * (b.x1 + b.x2) * 0.5f);   // 半分辨率中心
            const int cy = static_cast<int>(0.5f * (b.y1 + b.y2) * 0.5f);
            if (cx < R + PW || cy < R + PW || cx + R + PW >= half.w || cy + R + PW >= half.h)
                continue;
            int best_sad = 1 << 30, bdx = 0, bdy = 0;
            for (int dy = -R; dy <= R; ++dy) {
                for (int dx = -R; dx <= R; ++dx) {
                    int sad = 0;
                    for (int yy = -PW; yy <= PW && sad < best_sad; ++yy) {
                        const uint8_t* p = ctx->prev_half.p.data() +
                                           static_cast<size_t>(cy + yy) * ctx->prev_half.w + cx;
                        const uint8_t* q = half.p.data() +
                                           static_cast<size_t>(cy + yy + dy) * half.w + cx + dx;
                        for (int xx = -PW; xx <= PW; ++xx)
                            sad += std::abs(static_cast<int>(p[xx]) - static_cast<int>(q[xx]));
                    }
                    if (sad < best_sad) {
                        best_sad = sad;
                        bdx = dx;
                        bdy = dy;
                    }
                }
            }
            // 半分辨率位移 -> 全分辨率像素/帧
            ctx->flows[i * 2] = static_cast<float>(bdx) * 2.0f;
            ctx->flows[i * 2 + 1] = static_cast<float>(bdy) * 2.0f;
        }
    }
    ctx->trk->update(ctx->dets, ctx->descs.empty() ? nullptr : ctx->descs.data(),
                     static_cast<int>(ctx->descs.size()),
                     ctx->flows.empty() ? nullptr : ctx->flows.data(),
                     static_cast<int>(ctx->flows.size()));
    ctx->prev_half = half;
    const std::vector<int> shown = ctx->trk->shown();
    ctx->shown_tracks.clear();
    ctx->shown_tracks.reserve(shown.size());
    for (int idx : shown) ctx->shown_tracks.push_back(ctx->trk->tracks[idx]);
    ctx->states.clear();
    ctx->speed->update(ctx->shown_tracks, ctx->frame, ctx->states);
    const auto t5 = std::chrono::steady_clock::now();

    ctx->stats.motion_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ctx->stats.track_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();
    ++ctx->stats.frames;

    ctx->out_tracks.clear();
    ctx->out_tracks.reserve(ctx->states.size());
    for (const auto& [tid, st] : ctx->states) {
        SflyTrack t;
        t.track_id = tid;
        t.x1 = st.box.x1;
        t.y1 = st.box.y1;
        t.x2 = st.box.x2;
        t.y2 = st.box.y2;
        t.px_per_frame = st.px;
        t.speed_ready = st.ready ? 1 : 0;
        t.kmh = 3.6f * st.px * 0.92f * 10.f;
        ctx->out_tracks.push_back(t);
    }

    out->tiles = static_cast<int>(origins.size());
    out->n_dets = static_cast<int>(ctx->dets.size());
    out->n_tracks = static_cast<int>(ctx->out_tracks.size());
    out->tracks = ctx->out_tracks.data();

    ctx->out_origins.clear();
    ctx->out_origins.reserve(origins.size());
    for (const auto& o : origins) {
        SflyOrigin so;
        so.x = o.first;
        so.y = o.second;
        ctx->out_origins.push_back(so);
    }
    out->origins = ctx->out_origins.data();
    return 0;
}

extern "C" SFLY_API void sfly_get_stats(const SflyCtx* ctx, SflyStats* out) {
    if (!ctx || !out) return;
    *out = ctx->stats;
}
