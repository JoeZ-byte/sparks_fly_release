// ONNX Runtime C API 的动态加载封装（不链接、不要 dev 包）
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sfly {

// 动态加载 onnxruntime 库并取到 OrtApi 表。
// search_hint 为空时按平台默认路径尝试（pip 安装位置 / 系统路径）。
bool ort_init(const std::string& search_hint = "", std::string* err = nullptr);

struct Session {
    void* sess = nullptr;      // OrtSession*
    void* allocator = nullptr; // OrtAllocator*
    int64_t input_h = 0, input_w = 0;
    std::string out_name;          // 模型第 0 个输出的名字（ORT 不允许传空名）
    std::vector<float> out_buf;
};

// 创建会话（intra_op 线程数 = threads）
bool ort_create_session(const std::string& model_path, int threads, Session& out,
                        std::string* err = nullptr);
void ort_destroy(Session& s);

// 批量推理：输入 (n, 3, h, w) 的 NCHW float，输出指向模型第 0 个输出
// 返回的指针生命周期到下一次调用为止。
const float* ort_run(Session& s, const float* input, int n, int c, int h, int w,
                     int64_t& out_channels, int64_t& out_anchors, std::string* err = nullptr);

}  // namespace sfly