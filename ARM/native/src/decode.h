// DFL 解码 + NMS：与 Python 版 stream_demo.decode_topk / nms 逐值对齐
#pragma once
#include <vector>

#include "core.h"

namespace sfly {

// anchor 网格（与 Python make_anchors 一致：每个 stride 一个 ceil(imgsz/s)^2 网格，中心 +0.5 格）
void make_anchors(int imgsz, std::vector<float>& anchors_xy, std::vector<float>& strides);

// DFL：把 reg_max 长度的分布期望成距离
void dfl(const float* x, int n, float* out);

// 解码一帧的全部 anchor。
// pred 布局与 ONNX 输出一致：pred[(4*reg_max + nc) * A]，即通道优先。
// 只对 c >= conf_thr 的 anchor 做解码，再按 IoU 0.5 做 NMS。
void decode_topk(const float* pred, int nc, int A, const float* anchors_xy,
                 const float* strides, float conf_thr,
                 std::vector<Det>& out, int max_pre_nms = 1000);

// 与 Python nms() 相同语义（按分数降序、贪心抑制，返回保留索引）
std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                     float iou_thr);

}  // namespace sfly