// 图像算子：灰度 / INTER_AREA 降采样 / 帧差 / 阈值 / 3x3 椭圆闭运算
// 语义对齐 OpenCV（cv2.cvtColor BGR2GRAY、cv2.resize INTER_AREA、
// cv2.absdiff、cv2.threshold、cv2.morphologyEx MORPH_CLOSE）
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace sfly {

struct GrayImage {
    int w = 0, h = 0;
    std::vector<uint8_t> p;
    GrayImage() = default;
    GrayImage(int w_, int h_) : w(w_), h(h_), p(static_cast<size_t>(w_) * h_, 0) {}
    uint8_t& at(int x, int y) { return p[static_cast<size_t>(y) * w + x]; }
    uint8_t at(int x, int y) const { return p[static_cast<size_t>(y) * w + x]; }
};

// BGR24 -> 灰度（BT.601，与 cv2 相同的定点系数）
GrayImage bgr_to_gray(const uint8_t* bgr, int w, int h);

// 一步得到"灰度 + 2x2 面积降采样"（等价 bgr_to_gray 后再 resize_area_half，
// 但省掉 1MB 的全分辨率中间图 —— ARM 上这是主要开销）
GrayImage gray_area_half(const uint8_t* bgr, int w, int h);

// 2x2 盒平均降采样（INTER_AREA 在整数倍缩小时的等价实现）
GrayImage resize_area_half(const GrayImage& src);

// |a - b|
GrayImage absdiff(const GrayImage& a, const GrayImage& b);

// out = (in > thr) ? 255 : 0
GrayImage threshold_gt(const GrayImage& in, int thr);

// 3x3 椭圆结构元的闭运算（先膨胀后腐蚀）
GrayImage morph_close_ellipse3(const GrayImage& in);

// BGR -> 网络输入 blob（等价 cv2.dnn.blobFromImage(crop, 1/255, (imgsz,imgsz), None, True, False)）
// dst 需预先分配 3*imgsz*imgsz，布局 CHW、RGB、0~1
// bgr 是整幅图；crop 由 (x0,y0,tw,th) 指定，缩放目标 imgsz。越界自动 clamp。
// 8U 定点双线性缩放（复刻 cv2.resize INTER_LINEAR：11 位系数 + 四舍五入）
void resize_linear_u8(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

void blob_from_bgr(const uint8_t* bgr, int w, int h, int x0, int y0, int tw, int th,
                   int imgsz, float* dst);

// bg = (1-alpha)*bg + alpha*g（与 cv2.accumulateWeighted 相同）
void accumulate_weighted(float* bg, const uint8_t* g, int n, double alpha);

/* ---------------- 小目标增强（一档优化，见 docs/14） ----------------
 *
 * 卫星视频里车只有 2~26 px，云/水面/道路是低频大块。两个算子都只保留
 * "比背景亮的 2~5 px 斑点"这一频段，成本在 512² 半图上只有 1~2 ms。
 */

// 3x3 方形结构元的腐蚀 / 膨胀（边界按复制延拓，避免边缘假响应）
GrayImage erode3x3(const GrayImage& in);
GrayImage dilate3x3(const GrayImage& in);

// 白顶帽：I - opening(I,3x3) = I - dilate(erode(I))
// 输出为 0~255 的"亮斑强度"，平坦背景接近 0。
GrayImage tophat_white(const GrayImage& in);

// 可分离高斯（σ 给定，核半径 = ceil(3σ)，边界复制延拓）
GrayImage gaussian_blur(const GrayImage& in, double sigma);

// DoG 带通：clamp(G(σ1) - G(σ2) + 128)，平坦区接近 128
GrayImage dog_band(const GrayImage& in, double sigma1, double sigma2);

// 预处理模式：0=关闭 1=白顶帽 2=DoG
GrayImage enhance(const GrayImage& in, int mode);

}  // namespace sfly