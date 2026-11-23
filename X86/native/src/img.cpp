#include "img.h"

#include <algorithm>
#include <cmath>

namespace sfly {

GrayImage bgr_to_gray(const uint8_t* bgr, int w, int h) {
    GrayImage g(w, h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgr + static_cast<size_t>(y) * w * 3;
        uint8_t* out = g.p.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            // OpenCV 的 BGR2GRAY 定点系数：R*4899 + G*9617 + B*1868，>>14
            const int b = row[x * 3 + 0];
            const int gr = row[x * 3 + 1];
            const int r = row[x * 3 + 2];
            out[x] = static_cast<uint8_t>((r * 4899 + gr * 9617 + b * 1868 + (1 << 13)) >> 14);
        }
    }
    return g;
}

GrayImage gray_area_half(const uint8_t* bgr, int w, int h) {
    const int dw = w / 2, dh = h / 2;
    GrayImage out(dw, dh);
    auto gray_at = [&](int x, int y) -> int {
        const uint8_t* p = bgr + (static_cast<size_t>(y) * w + x) * 3;
        return (p[2] * 4899 + p[1] * 9617 + p[0] * 1868 + (1 << 13)) >> 14;
    };
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const int s = gray_at(2 * x, 2 * y) + gray_at(2 * x + 1, 2 * y) +
                          gray_at(2 * x, 2 * y + 1) + gray_at(2 * x + 1, 2 * y + 1);
            out.at(x, y) = static_cast<uint8_t>((s + 2) >> 2);
        }
    }
    return out;
}

GrayImage resize_area_half(const GrayImage& src) {
    const int dw = src.w / 2, dh = src.h / 2;
    GrayImage out(dw, dh);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const int s = src.at(2 * x, 2 * y) + src.at(2 * x + 1, 2 * y) +
                          src.at(2 * x, 2 * y + 1) + src.at(2 * x + 1, 2 * y + 1);
            out.at(x, y) = static_cast<uint8_t>((s + 2) >> 2);   // 四舍五入
        }
    }
    return out;
}

GrayImage absdiff(const GrayImage& a, const GrayImage& b) {
    GrayImage out(a.w, a.h);
    for (size_t i = 0; i < out.p.size(); ++i) {
        const int d = static_cast<int>(a.p[i]) - static_cast<int>(b.p[i]);
        out.p[i] = static_cast<uint8_t>(d < 0 ? -d : d);
    }
    return out;
}

GrayImage threshold_gt(const GrayImage& in, int thr) {
    GrayImage out(in.w, in.h);
    for (size_t i = 0; i < in.p.size(); ++i)
        out.p[i] = (in.p[i] > thr) ? 255 : 0;
    return out;
}

namespace {
// 3x3 椭圆结构元：十字 + 四个角中的 (0,1)(1,0)(1,2)(2,1) 与中心 —— cv2 的
// getStructuringElement(MORPH_ELLIPSE,(3,3)) 实际是 3x3 十字形
constexpr int kEll[3][3] = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};

void dilate3(const GrayImage& in, GrayImage& out) {
    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            uint8_t m = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= in.h) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!kEll[dy + 1][dx + 1]) continue;
                    const int xx = x + dx;
                    if (xx < 0 || xx >= in.w) continue;
                    m = std::max(m, in.at(xx, yy));
                }
            }
            out.at(x, y) = m;
        }
    }
}

void erode3(const GrayImage& in, GrayImage& out) {
    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            uint8_t m = 255;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= in.h) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!kEll[dy + 1][dx + 1]) continue;
                    const int xx = x + dx;
                    if (xx < 0 || xx >= in.w) continue;
                    m = std::min(m, in.at(xx, yy));
                }
            }
            out.at(x, y) = m;
        }
    }
}
}  // namespace

GrayImage morph_close_ellipse3(const GrayImage& in) {
    GrayImage tmp(in.w, in.h), out(in.w, in.h);
    dilate3(in, tmp);
    erode3(tmp, out);
    return out;
}

void resize_linear_u8(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    // 复刻 cv2.resize INTER_LINEAR(8U)：先水平插值（11 位系数，int 中间值），再垂直合成
    const int kCoef = 11;
    const int kScale = 1 << kCoef;
    const int kShift = 2 * kCoef;
    const int kRound = 1 << (kShift - 1);
    const double sx = static_cast<double>(sw) / dw;
    const double sy = static_cast<double>(sh) / dh;
    std::vector<int> xa0(dw), xa1(dw), xi0(dw);
    for (int x = 0; x < dw; ++x) {
        double fx = (x + 0.5) * sx - 0.5;
        if (fx < 0) fx = 0;
        int i0 = static_cast<int>(fx);
        if (i0 > sw - 2) i0 = std::max(0, sw - 2);
        const double frac = fx - i0;
        xi0[x] = i0;
        xa0[x] = static_cast<int>(std::lround((1.0 - frac) * kScale));
        xa1[x] = static_cast<int>(std::lround(frac * kScale));
    }
    std::vector<int> ya0(dh), ya1(dh), yi0(dh);
    for (int y = 0; y < dh; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        if (fy < 0) fy = 0;
        int i0 = static_cast<int>(fy);
        if (i0 > sh - 2) i0 = std::max(0, sh - 2);
        const double frac = fy - i0;
        yi0[y] = i0;
        ya0[y] = static_cast<int>(std::lround((1.0 - frac) * kScale));
        ya1[y] = static_cast<int>(std::lround(frac * kScale));
    }
    // 水平插值结果缓存两行
    std::vector<int> ha(dw), hb(dw);
    for (int y = 0; y < dh; ++y) {
        const int y0 = yi0[y], y1 = std::min(y0 + 1, sh - 1);
        const uint8_t* r0 = src + static_cast<size_t>(y0) * sw;
        const uint8_t* r1 = src + static_cast<size_t>(y1) * sw;
        for (int x = 0; x < dw; ++x) {
            const int x0 = xi0[x], x1 = std::min(x0 + 1, sw - 1);
            ha[x] = r0[x0] * xa0[x] + r0[x1] * xa1[x];
            hb[x] = r1[x0] * xa0[x] + r1[x1] * xa1[x];
        }
        const int b0 = ya0[y], b1 = ya1[y];
        uint8_t* out = dst + static_cast<size_t>(y) * dw;
        for (int x = 0; x < dw; ++x) {
            const int v = ha[x] * b0 + hb[x] * b1;
            out[x] = static_cast<uint8_t>(std::clamp((v + kRound) >> kShift, 0, 255));
        }
    }
}

void blob_from_bgr(const uint8_t* bgr, int w, int h, int x0, int y0, int tw, int th,
                   int imgsz, float* dst) {
    // 复用缓冲：每帧要处理 6 个图块，反复分配会直接吃掉 3 倍性能
    static thread_local std::vector<uint8_t> crop, resized, plane, plane_out;
    crop.resize(static_cast<size_t>(tw) * th * 3);
    resized.resize(static_cast<size_t>(imgsz) * imgsz * 3);
    plane.resize(static_cast<size_t>(tw) * th);
    plane_out.resize(static_cast<size_t>(imgsz) * imgsz);
    for (int y = 0; y < th; ++y) {
        for (int x = 0; x < tw; ++x) {
            const int gx = std::clamp(x0 + x, 0, w - 1);
            const int gy = std::clamp(y0 + y, 0, h - 1);
            for (int ch = 0; ch < 3; ++ch)
                crop[(static_cast<size_t>(y) * tw + x) * 3 + ch] =
                    bgr[(static_cast<size_t>(gy) * w + gx) * 3 + ch];
        }
    }
    // 2) 定点双线性缩放到 imgsz（与 cv2.resize INTER_LINEAR 最多差 ±2 LSB）
    for (int ch = 0; ch < 3; ++ch) {
        for (size_t i = 0; i < plane.size(); ++i) plane[i] = crop[i * 3 + ch];
        resize_linear_u8(plane.data(), tw, th, plane_out.data(), imgsz, imgsz);
        const int cidx = 2 - ch;   // BGR -> RGB
        for (size_t i = 0; i < plane_out.size(); ++i)
            resized[i * 3 + cidx] = plane_out[i];
    }
    // 3) HWC(BGR->RGB) uint8 -> CHW float /255
    const size_t sz = static_cast<size_t>(imgsz) * imgsz;
    for (int ch = 0; ch < 3; ++ch)
        for (size_t i = 0; i < sz; ++i)
            dst[ch * sz + i] = static_cast<float>(resized[i * 3 + ch]) / 255.0f;
}

void accumulate_weighted(float* bg, const uint8_t* g, int n, double alpha) {
    for (int i = 0; i < n; ++i)
        bg[i] = static_cast<float>((1.0 - alpha) * bg[i] + alpha * static_cast<double>(g[i]));
}

/* ---------------- 小目标增强（docs/14 一档第 1 项） ---------------- */

namespace {

inline uint8_t clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

}  // namespace

GrayImage erode3x3(const GrayImage& in) {
    // 可分离：先横向 min3，再纵向 min3（3x3 min 可分解为 1x3 与 3x1 的串联）
    const int w = in.w, h = in.h;
    std::vector<uint8_t> tmp(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = in.p.data() + static_cast<size_t>(y) * w;
        uint8_t* dst = tmp.data() + static_cast<size_t>(y) * w;
        const uint8_t left0 = row[0];
        dst[0] = std::min(left0, row[w > 1 ? 1 : 0]);
        for (int x = 1; x < w - 1; ++x)
            dst[x] = std::min(row[x - 1], std::min(row[x], row[x + 1]));
        if (w > 1) dst[w - 1] = std::min(row[w - 1], row[w - 2]);
    }
    GrayImage out(w, h);
    for (int y = 0; y < h; ++y) {
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y + 1 < h ? y + 1 : h - 1;
        const uint8_t* r0 = tmp.data() + static_cast<size_t>(y0) * w;
        const uint8_t* r1 = tmp.data() + static_cast<size_t>(y) * w;
        const uint8_t* r2 = tmp.data() + static_cast<size_t>(y1) * w;
        uint8_t* dst = out.p.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) dst[x] = std::min(r0[x], std::min(r1[x], r2[x]));
    }
    return out;
}

GrayImage dilate3x3(const GrayImage& in) {
    const int w = in.w, h = in.h;
    std::vector<uint8_t> tmp(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = in.p.data() + static_cast<size_t>(y) * w;
        uint8_t* dst = tmp.data() + static_cast<size_t>(y) * w;
        const uint8_t left0 = row[0];
        dst[0] = std::max(left0, row[w > 1 ? 1 : 0]);
        for (int x = 1; x < w - 1; ++x)
            dst[x] = std::max(row[x - 1], std::max(row[x], row[x + 1]));
        if (w > 1) dst[w - 1] = std::max(row[w - 1], row[w - 2]);
    }
    GrayImage out(w, h);
    for (int y = 0; y < h; ++y) {
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y + 1 < h ? y + 1 : h - 1;
        const uint8_t* r0 = tmp.data() + static_cast<size_t>(y0) * w;
        const uint8_t* r1 = tmp.data() + static_cast<size_t>(y) * w;
        const uint8_t* r2 = tmp.data() + static_cast<size_t>(y1) * w;
        uint8_t* dst = out.p.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) dst[x] = std::max(r0[x], std::max(r1[x], r2[x]));
    }
    return out;
}

GrayImage tophat_white(const GrayImage& in) {
    const GrayImage eroded = erode3x3(in);
    const GrayImage opened = dilate3x3(eroded);
    GrayImage out(in.w, in.h);
    for (size_t i = 0; i < in.p.size(); ++i)
        out.p[i] = clamp_u8(static_cast<int>(in.p[i]) - static_cast<int>(opened.p[i]));
    return out;
}

GrayImage gaussian_blur(const GrayImage& in, double sigma) {
    if (sigma <= 0.01) return in;
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<float> k(static_cast<size_t>(2 * r + 1));
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        const double v = std::exp(-(static_cast<double>(i) * i) / (2.0 * sigma * sigma));
        k[static_cast<size_t>(i + r)] = static_cast<float>(v);
        sum += v;
    }
    for (auto& v : k) v = static_cast<float>(v / sum);

    const int w = in.w, h = in.h;
    std::vector<float> tmp(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = in.p.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) {
                const int xx = std::clamp(x + i, 0, w - 1);
                acc += static_cast<float>(row[xx]) * k[static_cast<size_t>(i + r)];
            }
            tmp[static_cast<size_t>(y) * w + x] = acc;
        }
    }
    GrayImage out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) {
                const int yy = std::clamp(y + i, 0, h - 1);
                acc += tmp[static_cast<size_t>(yy) * w + x] * k[static_cast<size_t>(i + r)];
            }
            out.p[static_cast<size_t>(y) * w + x] = clamp_u8(static_cast<int>(acc + 0.5f));
        }
    }
    return out;
}

GrayImage dog_band(const GrayImage& in, double sigma1, double sigma2) {
    const GrayImage g1 = gaussian_blur(in, sigma1);
    const GrayImage g2 = gaussian_blur(in, sigma2);
    GrayImage out(in.w, in.h);
    for (size_t i = 0; i < in.p.size(); ++i)
        out.p[i] = clamp_u8(static_cast<int>(g1.p[i]) - static_cast<int>(g2.p[i]) + 128);
    return out;
}

GrayImage enhance(const GrayImage& in, int mode) {
    switch (mode) {
        case 1: return tophat_white(in);
        case 2: return dog_band(in, 1.0, 3.0);
        default: return in;
    }
}

}  // namespace sfly