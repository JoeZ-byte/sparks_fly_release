#include "decode.h"

#include <numeric>

namespace sfly {

void make_anchors(int imgsz, std::vector<float>& anchors_xy, std::vector<float>& strides) {
    anchors_xy.clear();
    strides.clear();
    anchors_xy.reserve(65536);
    strides.reserve(65536);
    for (int si = 0; si < 4; ++si) {
        const int s = kStrides[si];
        const int f = imgsz / s;
        for (int y = 0; y < f; ++y) {
            for (int x = 0; x < f; ++x) {
                // 与 Python make_anchors 一致：网格整数坐标 × stride（没有 +0.5）
                anchors_xy.push_back(static_cast<float>(x) * s);
                anchors_xy.push_back(static_cast<float>(y) * s);
                strides.push_back(static_cast<float>(s));
            }
        }
    }
}

void dfl(const float* x, int n, float* out) {
    // x: (n, kRegMax) 连续
    for (int i = 0; i < n; ++i) {
        const float* row = x + static_cast<size_t>(i) * kRegMax;
        float mx = row[0];
        for (int k = 1; k < kRegMax; ++k) mx = std::max(mx, row[k]);
        float sum = 0.f;
        float acc = 0.f;
        for (int k = 0; k < kRegMax; ++k) {
            const float e = std::exp(row[k] - mx);
            sum += e;
            acc += e * static_cast<float>(k);
        }
        out[i] = acc / sum;
    }
}

std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                     float iou_thr) {
    std::vector<int> keep;
    if (boxes.empty()) return keep;
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<char> removed(boxes.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        const int idx = order[i];
        if (removed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            const int jdx = order[j];
            if (removed[jdx]) continue;
            if (iou(boxes[idx], boxes[jdx]) > iou_thr) removed[jdx] = 1;
        }
    }
    return keep;
}

void decode_topk(const float* pred, int nc, int A, const float* anchors_xy,
                 const float* strides, float conf_thr,
                 std::vector<Det>& out, int max_pre_nms) {
    out.clear();
    const int box_ch = 4 * kRegMax;
    const size_t stride_ch = static_cast<size_t>(box_ch + nc) * A;

    // 1) 找过阈值的 anchor（pred 是通道优先：(box_ch+nc) × A）
    std::vector<int> idx;
    idx.reserve(256);
    for (int a = 0; a < A; ++a) {
        float c = pred[static_cast<size_t>(box_ch) * A + a];
        for (int k = 1; k < nc; ++k) {
            c = std::max(c, pred[static_cast<size_t>(box_ch + k) * A + a]);
        }
        if (c >= conf_thr) idx.push_back(a);
    }
    (void)stride_ch;
    if (idx.empty()) return;

    // 2) 超过 max_pre_nms 时按分数截断（与 Python 的 argpartition 语义一致：只保分数最高的）
    if (static_cast<int>(idx.size()) > max_pre_nms) {
        std::vector<std::pair<float, int>> sc;
        sc.reserve(idx.size());
        for (int a : idx) {
            float c = pred[static_cast<size_t>(box_ch) * A + a];
            for (int k = 1; k < nc; ++k) c = std::max(c, pred[static_cast<size_t>(box_ch + k) * A + a]);
            sc.emplace_back(c, a);
        }
        std::partial_sort(sc.begin(), sc.begin() + max_pre_nms, sc.end(),
                          [](const auto& x, const auto& y) { return x.first > y.first; });
        idx.resize(max_pre_nms);
        for (int i = 0; i < max_pre_nms; ++i) idx[i] = sc[i].second;
    }

    // 3) DFL + dist2bbox
    std::vector<Box> boxes;
    std::vector<float> scores;
    boxes.reserve(idx.size());
    scores.reserve(idx.size());
    std::vector<float> raw(4 * kRegMax);
    float lt[2], rb[2];
    for (int a : idx) {
        for (int q = 0; q < 4; ++q) {
            for (int k = 0; k < kRegMax; ++k) {
                raw[static_cast<size_t>(q) * kRegMax + k] =
                    pred[static_cast<size_t>(q * kRegMax + k) * A + a];
            }
        }
        float d[4];
        dfl(raw.data(), 2, lt);        // 左上 (x, y)
        dfl(raw.data() + 2 * kRegMax, 2, rb);  // 右下 (x, y)
        d[0] = lt[0]; d[1] = lt[1]; d[2] = rb[0]; d[3] = rb[1];
        const float cx = anchors_xy[2 * static_cast<size_t>(a)];
        const float cy = anchors_xy[2 * static_cast<size_t>(a) + 1];
        const float st = strides[a];
        Box b;
        b.x1 = cx - d[0] * st;
        b.y1 = cy - d[1] * st;
        b.x2 = cx + d[2] * st;
        b.y2 = cy + d[3] * st;
        boxes.push_back(b);
        float c = pred[static_cast<size_t>(box_ch) * A + a];
        for (int k = 1; k < nc; ++k) c = std::max(c, pred[static_cast<size_t>(box_ch + k) * A + a]);
        scores.push_back(c);
    }

    const std::vector<int> keep = nms(boxes, scores, 0.5f);
    out.reserve(keep.size());
    for (int k : keep) out.push_back(Det{boxes[k], scores[k]});
}

}  // namespace sfly