#include "runtime.h"

#include <cstring>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <glob.h>
#endif

// MinGW 只定义 __stdcall，没有 _stdcall；ORT 头在 _WIN32 下用 _stdcall
#ifdef _WIN32
#  ifndef _stdcall
#    define _stdcall __stdcall
#  endif
#endif
#include "onnxruntime_c_api.h"

namespace sfly {
namespace {

const OrtApi* g_api = nullptr;

#ifdef _WIN32
using LibHandle = HMODULE;
LibHandle dl_open(const char* p) { return ::LoadLibraryA(p); }
void* dl_sym(LibHandle h, const char* n) { return reinterpret_cast<void*>(::GetProcAddress(h, n)); }
#else
using LibHandle = void*;
LibHandle dl_open(const char* p) { return ::dlopen(p, RTLD_NOW | RTLD_GLOBAL); }
void* dl_sym(LibHandle h, const char* n) { return ::dlsym(h, n); }
#endif

}  // namespace

bool ort_init(const std::string& hint, std::string* err) {
    if (g_api) return true;
    std::vector<std::string> candidates;
    if (!hint.empty()) candidates.push_back(hint);
#ifdef _WIN32
    candidates.push_back("D:\\venvs\\viso\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime.dll");
    candidates.push_back("onnxruntime.dll");
#else
    // POSIX：libonnxruntime 的真名带版本号（libonnxruntime.so.1.32.0），用 glob 展开
    const char* patterns[] = {
        "/home/*/.local/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so*",
        "/usr/lib/*/libonnxruntime.so*",
        "/usr/local/lib/libonnxruntime.so*",
        "./libonnxruntime.so*",
    };
    for (const char* pat : patterns) {
        glob_t g{};
        if (::glob(pat, 0, nullptr, &g) == 0) {
            for (size_t i = 0; i < g.gl_pathc; ++i) candidates.push_back(g.gl_pathv[i]);
            globfree(&g);
        }
    }
    if (candidates.empty()) candidates.push_back("libonnxruntime.so");
#endif
    for (const auto& p : candidates) {
        LibHandle h = dl_open(p.c_str());
        if (!h) continue;
        auto get_base = reinterpret_cast<const OrtApiBase* (*)()>(dl_sym(h, "OrtGetApiBase"));
        if (!get_base) continue;
        const OrtApiBase* base = get_base();
        // 运行库可能比头文件旧/新：从高到低找一个双方都支持的版本
        for (int v = ORT_API_VERSION; v >= 1; --v) {
            g_api = base->GetApi(static_cast<uint32_t>(v));
            if (g_api) {
                std::printf("  ORT 已加载: %s (API v%d, 运行库 %s)\n", p.c_str(), v,
                            base->GetVersionString());
                return true;
            }
        }
    }
    if (err) *err = "找不到可用的 onnxruntime 动态库（检查 pip 安装路径或传入 --ort-lib）";
    return false;
}

bool ort_create_session(const std::string& model_path, int threads, Session& out,
                        std::string* err) {
    if (!g_api) {
        if (err) *err = "ort_init 未调用";
        return false;
    }
    OrtEnv* env = nullptr;
    if (g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sfly", &env) != nullptr) {
        if (err) *err = "CreateEnv 失败";
        return false;
    }
    OrtSessionOptions* opts = nullptr;
    g_api->CreateSessionOptions(&opts);
    g_api->SetIntraOpNumThreads(opts, threads);
    g_api->SetInterOpNumThreads(opts, 1);
    g_api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    // 与 Python 版一致：关掉自旋等待（否则满载时会和主线程抢核）
    g_api->AddSessionConfigEntry(opts, "session.intra_op.allow_spinning", "0");
    g_api->AddSessionConfigEntry(opts, "session.use_env_allocators", "1");
    OrtSession* sess = nullptr;
#ifdef _WIN32
    // Windows 版 ORT 用宽字符路径
    const int wlen = ::MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, wpath.data(), wlen);
    OrtStatus* st = g_api->CreateSession(env, wpath.c_str(), opts, &sess);
#else
    OrtStatus* st = g_api->CreateSession(env, model_path.c_str(), opts, &sess);
#endif
    if (st) {
        if (err) *err = std::string("CreateSession 失败: ") + g_api->GetErrorMessage(st);
        g_api->ReleaseStatus(st);
        return false;
    }
    OrtAllocator* alloc = nullptr;
    g_api->GetAllocatorWithDefaultOptions(&alloc);
    char* name = nullptr;
    if (g_api->SessionGetOutputName(sess, 0, alloc, &name) == nullptr && name) {
        out.out_name = name;
        g_api->AllocatorFree(alloc, name);
    }
    out.sess = sess;
    out.allocator = alloc;
    return true;
}

void ort_destroy(Session& s) {
    if (s.sess && g_api) g_api->ReleaseSession(reinterpret_cast<::OrtSession*>(s.sess));
    s.sess = nullptr;
}

const float* ort_run(Session& s, const float* input, int n, int c, int h, int w,
                     int64_t& out_channels, int64_t& out_anchors, std::string* err) {
    static const char* kInput = "batch";
    const int64_t in_shape[4] = {n, c, h, w};
    OrtMemoryInfo* mem = nullptr;
    g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
    OrtValue* in_tensor = nullptr;
    g_api->CreateTensorWithDataAsOrtValue(mem, const_cast<float*>(input),
                                          sizeof(float) * n * c * h * w,
                                          in_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                          &in_tensor);
    const char* in_names[1] = {kInput};
    const char* out_names[1] = {s.out_name.c_str()};
    OrtValue* out_tensor = nullptr;
    OrtStatus* st = g_api->Run(reinterpret_cast<::OrtSession*>(s.sess), nullptr,
                               in_names, reinterpret_cast<const OrtValue* const*>(&in_tensor), 1,
                               out_names, 1, &out_tensor);
    g_api->ReleaseValue(in_tensor);
    g_api->ReleaseMemoryInfo(mem);
    if (st) {
        if (err) *err = std::string("Run 失败: ") + g_api->GetErrorMessage(st);
        g_api->ReleaseStatus(st);
        return nullptr;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    g_api->GetTensorTypeAndShape(out_tensor, &info);
    size_t dims = 0;
    g_api->GetDimensionsCount(info, &dims);
    int64_t shape[4] = {0, 0, 0, 0};
    g_api->GetDimensions(info, shape, dims);
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    out_channels = dims >= 2 ? shape[1] : 0;
    out_anchors = dims >= 3 ? shape[2] : 0;
    void* data = nullptr;
    g_api->GetTensorMutableData(out_tensor, &data);
    s.out_buf.assign(static_cast<const float*>(data),
                     static_cast<const float*>(data) + out_channels * out_anchors * n);
    g_api->ReleaseValue(out_tensor);
    return s.out_buf.data();
}

}  // namespace sfly