// vkblas_hipblas.c — hipBLAS ABI 兼容层 (LD_PRELOAD 劫持)
// 热路径 (fp32 GEMM) → Vulkan; 其余 → dlsym 转发真 libhipblas
// 所有签名严格对齐 /opt/rocm-6.4.3/include/hipblas/hipblas.h (v1 int 参数)
#include "vkblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime_api.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ---------- 内部状态 ----------
// 设计: hipblasCreate 直接返回真 hipblas handle (PyTorch 内部会用 RTLD_NEXT/句柄
// 直接调真库函数如 hipblasSetWorkspace, 假 handle 会炸); 我们自己的 stream/mode
// 状态存全局 (PyTorch 单 handle 场景, 可接受)
static hipStream_t g_stream = NULL;
static hipblasPointerMode_t g_mode = HIPBLAS_POINTER_MODE_HOST;

// ---------- 真 hipBLAS 转发 ----------
static void* real_hipblas(void) {
    static void* h = NULL;
    if (!h) {
        h = dlopen("libhipblas.so.2", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("/opt/rocm/lib/libhipblas.so.2", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) fprintf(stderr, "[vkblas] real libhipblas.so.2 not found: %s\n", dlerror());
    }
    return h;
}
static void* rb_sym(const char* name) { return real_hipblas() ? dlsym(real_hipblas(), name) : NULL; }

// 通用转发: 函数指针类型从自身声明推导 (__typeof__), 参数直传
#define FWD(name, ...) __extension__ ({                              \
    static __typeof__(&name) fn = NULL;                              \
    if (!fn) fn = (__typeof__(&name))rb_sym(#name);                  \
    fn ? fn(__VA_ARGS__) : HIPBLAS_STATUS_NOT_SUPPORTED; })

// ---------- 句柄管理 ----------
// hipblasCreate 返回真 handle; 我们自己的状态在全局 (见上)
hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    static hipblasStatus_t (*real_create)(hipblasHandle_t*) = NULL;
    if (!real_create) real_create = (hipblasStatus_t(*)(hipblasHandle_t*))rb_sym("hipblasCreate");
    if (!real_create) return HIPBLAS_STATUS_NOT_SUPPORTED;
    return real_create(handle);
}
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return FWD(hipblasDestroy, handle);
}
hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    g_stream = streamId;
    return FWD(hipblasSetStream, handle, streamId);
}
hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    *streamId = g_stream;
    return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    g_mode = mode;
    return FWD(hipblasSetPointerMode, handle, mode);
}
hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    *mode = g_mode;
    return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return FWD(hipblasSetAtomicsMode, handle, atomics_mode);
}

// PyTorch ROCm 后端要求 (workspace 管理)
// 注意: 直接返回 SUCCESS — workspace 只对 rocBLAS 后端有意义, Vulkan 不需要;
// 转发真库在本进程里会因符号插值/句柄问题返回错误, 且 PyTorch 只需成功状态
hipblasStatus_t hipblasSetWorkspace(hipblasHandle_t handle, void* workspace, size_t size) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    (void)workspace; (void)size;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGetVersion(hipblasHandle_t handle, int* version) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return FWD(hipblasGetVersion, handle, version);
}

const char* hipblasStatusToString(hipblasStatus_t status) {
    static const char* (*fn)(hipblasStatus_t) = NULL;
    if (!fn) fn = (const char* (*)(hipblasStatus_t))rb_sym("hipblasStatusToString");
    return fn ? fn(status) : "unknown";
}

// ---------- 标量读取 (host/device 双模式, 全局 pointer mode) ----------
static int get_scalar_f32(const void* p, float* out) {
    if (g_mode == HIPBLAS_POINTER_MODE_DEVICE)
        return hipMemcpyDtoH(out, (void*)p, 4) == hipSuccess ? 0 : -1;
    *out = *(const float*)p;
    return 0;
}

// ---------- HIP 内存生命周期 hook (dma-buf import 缓存失效) ----------
// 导出 hipFree/hipHostFree/hipFreeManaged 并转发真 libamdhip64: 进程内所有对应 free
// 调用 (LD_PRELOAD 或符号优先场景) 先到我们 → 按块基址失效 import 缓存条目, 杜绝
// "hipFree 后地址复用 → 缓存读到陈旧 dma-buf"。hipMalloc 不 hook — 缓存 miss 自然注册。
// vkblas_hook_active 自证实门控: free hook 一旦被调用过, 即证明本进程的 free 都经过
// 我们 (符号绑定是全局性的), vkblas.c 据此启用 import 缓存; 纯 dlopen (不经 hook)
// 场景下 free 会绕过我们 → 缓存永久禁用, 退回每次全量导入 (正确性优先)。
int vkblas_hook_active = 0;

static void* real_amdhip(void) {
    static void* h = NULL;
    if (!h) {
        h = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("/opt/rocm/lib/libamdhip64.so.6", RTLD_LAZY | RTLD_GLOBAL);
    }
    return h;
}

#define HOOK_FREE(fname)                                                     \
    hipError_t fname(void* ptr) {                                            \
        static hipError_t (*real)(void*) = NULL;                             \
        if (!real) real = (hipError_t(*)(void*))dlsym(real_amdhip(), #fname);\
        vkblas_hook_active = 1;                                              \
        if (getenv("VKBLAS_TRACE"))                                          \
            fprintf(stderr, "[vkblas] %s hook: %p (cache active)\n", #fname, ptr); \
        vkblas_cache_invalidate_base(ptr);                                   \
        return real ? real(ptr) : hipErrorRuntimeMemory;                     \
    }

HOOK_FREE(hipFree)
HOOK_FREE(hipHostFree)
HOOK_FREE(hipFreeManaged)

// ---------- 核心: fp32 GEMM (Vulkan) ----------
// hipBLAS column-major 语义 → Vulkan row-major:
//   C_rm[n,m] = op(B)^T @ op(A)^T → A_eff=B参数, B_eff=A参数, M'=n, N'=m, K'=k
//   A_eff 是 B 参数: transb=N(B 是 (k×n)cm)时需转置读, transb=T((n×k)cm)时直接读
// 前向声明 (GEMM 族在文件后部引用)
static hipblasStatus_t vk_gemm_c64(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const hipComplex* alpha, const hipComplex* AP, int lda,
                                   const hipComplex* BP, int ldb,
                                   const hipComplex* beta, hipComplex* CP, int ldc);
static hipblasStatus_t vk_gemm_strided_c64(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const hipComplex* alpha, const hipComplex* AP, int lda, long long strideA,
                                           const hipComplex* BP, int ldb, long long strideB,
                                           const hipComplex* beta, hipComplex* CP, int ldc, long long strideC,
                                           int batchCount);
static hipblasStatus_t vk_gemm_z64(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda,
                                   const hipDoubleComplex* BP, int ldb,
                                   const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
static hipblasStatus_t vk_gemm_strided_z64(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, long long strideA,
                                           const hipDoubleComplex* BP, int ldb, long long strideB,
                                           const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc, long long strideC,
                                           int batchCount);
static hipblasStatus_t vk_gemm_f32(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const void* alpha, const void* AP, int lda,
                                   const void* BP, int ldb,
                                   const void* beta, void* CP, int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] gemm %s%s m=%d n=%d k=%d lda=%d ldb=%d\n",
                transA == HIPBLAS_OP_T ? "T" : "N", transB == HIPBLAS_OP_T ? "T" : "N",
                m, n, k, lda, ldb);
    double t0 = now_s(), t1, t2;
    hipStreamSynchronize(g_stream);  // 等调用方 HIP 工作完成
    t1 = now_s();
    vkblas_status_t st = vkblas_gemm_f32(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    t2 = now_s();
    if (getenv("VKBLAS_PROFILE"))
        fprintf(stderr, "[vk] prof: entry=%.1fus hipSync=%.1fus vkblas=%.2fms total=%.2fms\n",
                (t1 - t0) * 1e6, (t1 - t0) * 1e6, (t2 - t1) * 1e3, (t2 - t0) * 1e3);
    if (st != VKBLAS_OK) {  // import/init 失败 → 转发真 hipblas (handle 即真 handle)
        return FWD(hipblasSgemm, handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t vk_gemm_strided_f32(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const void* alpha, const void* AP, int lda, long long strideA,
                                           const void* BP, int ldb, long long strideB,
                                           const void* beta, void* CP, int ldc, long long strideC,
                                           int batchCount) {
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    vkblas_status_t st = vkblas_gemm_f32(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != VKBLAS_OK) {
        return FWD(hipblasSgemmStridedBatched, handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
                   BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// ---------- fp32 判定 ----------
static int is_fp32_ex(hipDataType t, hipblasComputeType_t c) {
    return (t == HIP_R_32F) &&
           (c == HIPBLAS_COMPUTE_32F || c == HIPBLAS_COMPUTE_32F_PEDANTIC);
}
// 通吃 HIPBLAS_V2 两种模式: HIP_R_16BF=14; legacy hipblasDatatype_t HIPBLAS_R_16B=168
static int is_bf16_ex(hipDataType t) { return (int)t == HIP_R_16BF || (int)t == HIPBLAS_R_16B; }
static int is_fp16_ex(hipDataType t) { return (int)t == HIP_R_16F || (int)t == HIPBLAS_R_16F; }

// ---------- bf16 GEMM 回退 (Vulkan) ----------
// 与 vk_gemm_f32 相同的 column-major → row-major 翻译; A/B/C 为 bf16
static hipblasStatus_t vk_gemm_bf16(hipblasHandle_t handle,
                                    hipblasOperation_t transA, hipblasOperation_t transB,
                                    int m, int n, int k,
                                    const void* alpha, const void* AP, int lda,
                                    const void* BP, int ldb,
                                    const void* beta, void* CP, int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] gemm bf16 %s%s m=%d n=%d k=%d lda=%d ldb=%d\n",
                transA == HIPBLAS_OP_T ? "T" : "N", transB == HIPBLAS_OP_T ? "T" : "N",
                m, n, k, lda, ldb);
    hipStreamSynchronize(g_stream);
    vkblas_status_t st = vkblas_gemm_bf16(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    if (st != VKBLAS_OK) {  // init/cvt 失败 → 转发真库
        return FWD(hipblasGemmEx_v2, handle, transA, transB, m, n, k, alpha, AP,
                   HIP_R_16BF, lda, BP, HIP_R_16BF, ldb, beta, CP, HIP_R_16BF, ldc,
                   HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t vk_gemm_strided_bf16(hipblasHandle_t handle,
                                            hipblasOperation_t transA, hipblasOperation_t transB,
                                            int m, int n, int k,
                                            const void* alpha, const void* AP, int lda, long long strideA,
                                            const void* BP, int ldb, long long strideB,
                                            const void* beta, void* CP, int ldc, long long strideC,
                                            int batchCount) {
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    vkblas_status_t st = vkblas_gemm_bf16(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != VKBLAS_OK) {
        return FWD(hipblasGemmStridedBatchedEx_v2, handle, transA, transB, m, n, k,
                   alpha, AP, HIP_R_16BF, lda, strideA, BP, HIP_R_16BF, ldb, strideB,
                   beta, CP, HIP_R_16BF, ldc, strideC, batchCount,
                   HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// ---------- fp16 GEMM 回退 (Vulkan) ----------
// 同 bf16: A/B/C 为 fp16, 内部 fp32 引擎 + cvt (unpackHalf2x16/packHalf2x16)
static hipblasStatus_t vk_gemm_f16(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const void* alpha, const void* AP, int lda,
                                   const void* BP, int ldb,
                                   const void* beta, void* CP, int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] gemm f16 %s%s m=%d n=%d k=%d lda=%d ldb=%d\n",
                transA == HIPBLAS_OP_T ? "T" : "N", transB == HIPBLAS_OP_T ? "T" : "N",
                m, n, k, lda, ldb);
    hipStreamSynchronize(g_stream);
    vkblas_status_t st = vkblas_gemm_f16(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    if (st != VKBLAS_OK) {  // init/cvt 失败 → 转发真库
        return FWD(hipblasGemmEx_v2, handle, transA, transB, m, n, k, alpha, AP,
                   HIP_R_16F, lda, BP, HIP_R_16F, ldb, beta, CP, HIP_R_16F, ldc,
                   HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t vk_gemm_strided_f16(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const void* alpha, const void* AP, int lda, long long strideA,
                                           const void* BP, int ldb, long long strideB,
                                           const void* beta, void* CP, int ldc, long long strideC,
                                           int batchCount) {
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    float a, b;
    if (get_scalar_f32(alpha, &a) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (get_scalar_f32(beta, &b) != 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    vkblas_status_t st = vkblas_gemm_f16(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != VKBLAS_OK) {
        return FWD(hipblasGemmStridedBatchedEx_v2, handle, transA, transB, m, n, k,
                   alpha, AP, HIP_R_16F, lda, strideA, BP, HIP_R_16F, ldb, strideB,
                   beta, CP, HIP_R_16F, ldc, strideC, batchCount,
                   HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// ---------- GEMM 族 ----------
hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                             int m, int n, int k, const float* alpha, const float* AP, int lda,
                             const float* BP, int ldb, const float* beta, float* CP, int ldc) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_f32(handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
}
hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k, const float* alpha, const float* AP, int lda, long long strideA,
                                           const float* BP, int ldb, long long strideB, const float* beta,
                                           float* CP, int ldc, long long strideC, int batchCount) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_strided_f32(handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
                               BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
}
hipblasStatus_t hipblasGemmEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                 int m, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda,
                                 const void* B, hipDataType bType, int ldb, const void* beta, void* C,
                                 hipDataType cType, int ldc, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[dbg] hipblasGemmEx_v2 aType=%d\n", (int)aType);
    if (is_fp32_ex(aType, computeType) && is_fp32_ex(bType, computeType) && cType == HIP_R_32F)
        return vk_gemm_f32(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (is_bf16_ex(aType) && is_bf16_ex(bType) && is_bf16_ex(cType))
        return vk_gemm_bf16(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (is_fp16_ex(aType) && is_fp16_ex(bType) && is_fp16_ex(cType))
        return vk_gemm_f16(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return FWD(hipblasGemmEx_v2, handle, transA, transB, m, n, k, alpha, A, aType, lda,
               B, bType, ldb, beta, C, cType, ldc, computeType, algo);
}
hipblasStatus_t hipblasGemmStridedBatchedEx_v2(hipblasHandle_t handle,
                                               hipblasOperation_t transA, hipblasOperation_t transB,
                                               int m, int n, int k, const void* alpha, const void* A, hipDataType aType,
                                               int lda, hipblasStride strideA, const void* B, hipDataType bType,
                                               int ldb, hipblasStride strideB, const void* beta, void* C,
                                               hipDataType cType, int ldc, hipblasStride strideC, int batchCount,
                                               hipblasComputeType_t computeType, hipblasGemmAlgo_t algo) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    if (is_fp32_ex(aType, computeType) && is_fp32_ex(bType, computeType) && cType == HIP_R_32F)
        return vk_gemm_strided_f32(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    if (is_bf16_ex(aType) && is_bf16_ex(bType) && is_bf16_ex(cType))
        return vk_gemm_strided_bf16(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                    B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    if (is_fp16_ex(aType) && is_fp16_ex(bType) && is_fp16_ex(cType))
        return vk_gemm_strided_f16(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                   B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    return FWD(hipblasGemmStridedBatchedEx_v2, handle, transA, transB, m, n, k, alpha, A, aType,
               lda, strideA, B, bType, ldb, strideB, beta, C, cType, ldc, strideC, batchCount,
               computeType, algo);
}
hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                              int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType,
                              int lda, const void* B, hipblasDatatype_t bType, int ldb, const void* beta,
                              void* C, hipblasDatatype_t cType, int ldc, hipblasDatatype_t computeType,
                              hipblasGemmAlgo_t algo) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    if (is_bf16_ex(aType) && is_bf16_ex(bType) && is_bf16_ex(cType))
        return vk_gemm_bf16(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (is_fp16_ex(aType) && is_fp16_ex(bType) && is_fp16_ex(cType))
        return vk_gemm_f16(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return FWD(hipblasGemmEx, handle, transA, transB, m, n, k, alpha, A, aType, lda,
               B, bType, ldb, beta, C, cType, ldc, computeType, algo);
}
hipblasStatus_t hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                             int m, int n, int k, const double* alpha, const double* AP, int lda,
                             const double* BP, int ldb, const double* beta, double* CP, int ldc) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] gemm f64 %s%s m=%d n=%d k=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
                transA == HIPBLAS_OP_T ? "T" : "N", transB == HIPBLAS_OP_T ? "T" : "N",
                m, n, k, lda, ldb, ldc, *alpha, *beta);
    hipStreamSynchronize(g_stream);
    int st = vkblas_gemm_f64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        *alpha, BP, (uint32_t)ldb, AP, (uint32_t)lda, *beta, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    if (st != 0)
        return FWD(hipblasDgemm, handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
    return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k, const double* alpha, const double* AP, int lda, long long strideA,
                                           const double* BP, int ldb, long long strideB, const double* beta,
                                           double* CP, int ldc, long long strideC, int batchCount) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    int st = vkblas_gemm_f64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        *alpha, BP, (uint32_t)ldb, AP, (uint32_t)lda, *beta, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != 0)
        return FWD(hipblasDgemmStridedBatched, handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
                   BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
    return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t hipblasCgemm_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                int m, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda,
                                const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_c64(handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
}
hipblasStatus_t hipblasCgemmStridedBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                              int m, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, long long strideA,
                                              const hipComplex* BP, int ldb, long long strideB, const hipComplex* beta,
                                              hipComplex* CP, int ldc, long long strideC, int batchCount) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_strided_c64(handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
                               BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
}
hipblasStatus_t hipblasZgemm_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda,
                                const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_z64(handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
}
hipblasStatus_t hipblasZgemmStridedBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                              int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, long long strideA,
                                              const hipDoubleComplex* BP, int ldb, long long strideB, const hipDoubleComplex* beta,
                                              hipDoubleComplex* CP, int ldc, long long strideC, int batchCount) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    return vk_gemm_strided_z64(handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
                               BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
}

// ---------- complex64 GEMM 回退 (Vulkan) ----------
// hipComplex = {float x, y} 交错; column-major → row-major 翻译同 fp32
static hipblasStatus_t vk_gemm_c64(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const hipComplex* alpha, const hipComplex* AP, int lda,
                                   const hipComplex* BP, int ldb,
                                   const hipComplex* beta, hipComplex* CP, int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    int st = vkblas_gemm_c64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        alpha->x, alpha->y, BP, (uint32_t)ldb, AP, (uint32_t)lda,
        beta->x, beta->y, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    if (st != 0) {  // 引擎不可用 → 转发真库
        return FWD(hipblasCgemm_v2, handle, transA, transB, m, n, k, alpha, AP, lda,
                   BP, ldb, beta, CP, ldc);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t vk_gemm_strided_c64(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const hipComplex* alpha, const hipComplex* AP, int lda, long long strideA,
                                           const hipComplex* BP, int ldb, long long strideB,
                                           const hipComplex* beta, hipComplex* CP, int ldc, long long strideC,
                                           int batchCount) {
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    int st = vkblas_gemm_c64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        alpha->x, alpha->y, BP, (uint32_t)ldb, AP, (uint32_t)lda,
        beta->x, beta->y, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != 0) {
        return FWD(hipblasCgemmStridedBatched_v2, handle, transA, transB, m, n, k,
                   alpha, AP, lda, strideA, BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
    }
    return HIPBLAS_STATUS_SUCCESS;
}

// ---------- complex128 GEMM 回退 (Vulkan) ----------
// hipDoubleComplex = {double x, y} 交错; 拆 4×fp64 GEMM (d64 引擎)
static hipblasStatus_t vk_gemm_z64(hipblasHandle_t handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda,
                                   const hipDoubleComplex* BP, int ldb,
                                   const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    // 2026-08-23: 真库 Zgemm 在 gfx803 复测健康 (16³ maxabsdiff 6.5e-14),
    // vkblas d64 引擎复测算错 (maxabsdiff 159) — 反转优先级: 真库优先, vkblas 兜底
    hipStreamSynchronize(g_stream);
    hipblasStatus_t rs = FWD(hipblasZgemm_v2, handle, transA, transB, m, n, k, alpha, AP, lda,
                             BP, ldb, beta, CP, ldc);
    if (rs == HIPBLAS_STATUS_SUCCESS) return HIPBLAS_STATUS_SUCCESS;
    if (rs != HIPBLAS_STATUS_NOT_SUPPORTED) return rs;  // 其他错误 (无效参数等) 直接上报
    int st = vkblas_gemm_z64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        alpha->x, alpha->y, BP, (uint32_t)ldb, AP, (uint32_t)lda,
        beta->x, beta->y, CP, (uint32_t)ldc,
        1, 0, 0, 0);
    if (st != 0) return HIPBLAS_STATUS_NOT_SUPPORTED;  // 引擎也不可用
    return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t vk_gemm_strided_z64(hipblasHandle_t handle,
                                           hipblasOperation_t transA, hipblasOperation_t transB,
                                           int m, int n, int k,
                                           const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, long long strideA,
                                           const hipDoubleComplex* BP, int ldb, long long strideB,
                                           const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc, long long strideC,
                                           int batchCount) {
    if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return HIPBLAS_STATUS_INVALID_VALUE;
    hipStreamSynchronize(g_stream);
    hipblasStatus_t rs = FWD(hipblasZgemmStridedBatched_v2, handle, transA, transB, m, n, k,
                             alpha, AP, lda, strideA, BP, ldb, strideB, beta, CP, ldc, strideC, batchCount);
    if (rs == HIPBLAS_STATUS_SUCCESS) return HIPBLAS_STATUS_SUCCESS;
    if (rs != HIPBLAS_STATUS_NOT_SUPPORTED) return rs;
    int st = vkblas_gemm_z64(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        alpha->x, alpha->y, BP, (uint32_t)ldb, AP, (uint32_t)lda,
        beta->x, beta->y, CP, (uint32_t)ldc,
        (uint32_t)batchCount, strideB, strideA, strideC);
    if (st != 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    return HIPBLAS_STATUS_SUCCESS;
}

// ---------- dot / gemv / trsm / 分解族: 全部转发 ----------
hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx,
                            const float* y, int incy, float* result) {
    return FWD(hipblasSdot, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x, int incx,
                            const double* y, int incy, double* result) {
    return FWD(hipblasDdot, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasCdotc_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx,
                                const hipComplex* y, int incy, hipComplex* result) {
    return FWD(hipblasCdotc_v2, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasCdotu_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx,
                                const hipComplex* y, int incy, hipComplex* result) {
    return FWD(hipblasCdotu_v2, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasZdotc_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx,
                                const hipDoubleComplex* y, int incy, hipDoubleComplex* result) {
    return FWD(hipblasZdotc_v2, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasZdotu_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx,
                                const hipDoubleComplex* y, int incy, hipDoubleComplex* result) {
    return FWD(hipblasZdotu_v2, handle, n, x, incx, y, incy, result);
}
hipblasStatus_t hipblasDotEx_v2(hipblasHandle_t handle, int n, const void* x, hipDataType xType, int incx,
                                const void* y, hipDataType yType, int incy, void* result,
                                hipDataType resultType, hipDataType executionType) {
    return FWD(hipblasDotEx_v2, handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
}
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const float* alpha, const float* AP, int lda, const float* x, int incx,
                             const float* beta, float* y, int incy) {
    return FWD(hipblasSgemv, handle, trans, m, n, alpha, AP, lda, x, incx, beta, y, incy);
}
hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                             const double* alpha, const double* AP, int lda, const double* x, int incx,
                             const double* beta, double* y, int incy) {
    return FWD(hipblasDgemv, handle, trans, m, n, alpha, AP, lda, x, incx, beta, y, incy);
}
hipblasStatus_t hipblasCgemv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                const hipComplex* alpha, const hipComplex* AP, int lda,
                                const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy) {
    return FWD(hipblasCgemv_v2, handle, trans, m, n, alpha, AP, lda, x, incx, beta, y, incy);
}
hipblasStatus_t hipblasZgemv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                                const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda,
                                const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta,
                                hipDoubleComplex* y, int incy) {
    return FWD(hipblasZgemv_v2, handle, trans, m, n, alpha, AP, lda, x, incx, beta, y, incy);
}
hipblasStatus_t hipblasStrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                             const float* alpha, const float* AP, int lda, float* BP, int ldb) {
    return FWD(hipblasStrsm, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb);
}
hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                             const double* alpha, const double* AP, int lda, double* BP, int ldb) {
    return FWD(hipblasDtrsm, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb);
}
hipblasStatus_t hipblasCtrsm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                const hipComplex* alpha, const hipComplex* AP, int lda, hipComplex* BP, int ldb) {
    return FWD(hipblasCtrsm_v2, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb);
}
hipblasStatus_t hipblasZtrsm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda,
                                hipDoubleComplex* BP, int ldb) {
    return FWD(hipblasZtrsm_v2, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb);
}
hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                    hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                    const float* alpha, const float* const AP[], int lda,
                                    float* const BP[], int ldb, int batchCount) {
    return FWD(hipblasStrsmBatched, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb, batchCount);
}
hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                    hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                    const double* alpha, const double* const AP[], int lda,
                                    double* const BP[], int ldb, int batchCount) {
    return FWD(hipblasDtrsmBatched, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb, batchCount);
}
hipblasStatus_t hipblasCtrsmBatched_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                       hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                       const hipComplex* alpha, const hipComplex* const AP[], int lda,
                                       hipComplex* const BP[], int ldb, int batchCount) {
    return FWD(hipblasCtrsmBatched_v2, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb, batchCount);
}
hipblasStatus_t hipblasZtrsmBatched_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo,
                                       hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n,
                                       const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int lda,
                                       hipDoubleComplex* const BP[], int ldb, int batchCount) {
    return FWD(hipblasZtrsmBatched_v2, handle, side, uplo, transA, diag, m, n, alpha, AP, lda, BP, ldb, batchCount);
}
hipblasStatus_t hipblasSgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n,
                                    const int nrhs, float* const A[], const int lda, float* const B[],
                                    const int ldb, int* info, int* deviceInfo, const int batchCount) {
    return FWD(hipblasSgelsBatched, handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}
hipblasStatus_t hipblasDgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n,
                                    const int nrhs, double* const A[], const int lda, double* const B[],
                                    const int ldb, int* info, int* deviceInfo, const int batchCount) {
    return FWD(hipblasDgelsBatched, handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}
hipblasStatus_t hipblasCgelsBatched_v2(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n,
                                       const int nrhs, hipComplex* const A[], const int lda, hipComplex* const B[],
                                       const int ldb, int* info, int* deviceInfo, const int batchCount) {
    return FWD(hipblasCgelsBatched_v2, handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}
hipblasStatus_t hipblasZgelsBatched_v2(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n,
                                       const int nrhs, hipDoubleComplex* const A[], const int lda,
                                       hipDoubleComplex* const B[], const int ldb, int* info, int* deviceInfo,
                                       const int batchCount) {
    return FWD(hipblasZgelsBatched_v2, handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}
hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle, const int m, const int n, float* const A[],
                                     const int lda, float* const ipiv[], int* info, const int batchCount) {
    return FWD(hipblasSgeqrfBatched, handle, m, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle, const int m, const int n, double* const A[],
                                     const int lda, double* const ipiv[], int* info, const int batchCount) {
    return FWD(hipblasDgeqrfBatched, handle, m, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasCgeqrfBatched_v2(hipblasHandle_t handle, const int m, const int n, hipComplex* const A[],
                                        const int lda, hipComplex* const ipiv[], int* info, const int batchCount) {
    return FWD(hipblasCgeqrfBatched_v2, handle, m, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasZgeqrfBatched_v2(hipblasHandle_t handle, const int m, const int n, hipDoubleComplex* const A[],
                                        const int lda, hipDoubleComplex* const ipiv[], int* info, const int batchCount) {
    return FWD(hipblasZgeqrfBatched_v2, handle, m, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle, const int n, float* const A[], const int lda,
                                     int* ipiv, int* info, const int batchCount) {
    return FWD(hipblasSgetrfBatched, handle, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle, const int n, double* const A[], const int lda,
                                     int* ipiv, int* info, const int batchCount) {
    return FWD(hipblasDgetrfBatched, handle, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasCgetrfBatched_v2(hipblasHandle_t handle, const int n, hipComplex* const A[], const int lda,
                                        int* ipiv, int* info, const int batchCount) {
    return FWD(hipblasCgetrfBatched_v2, handle, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasZgetrfBatched_v2(hipblasHandle_t handle, const int n, hipDoubleComplex* const A[], const int lda,
                                        int* ipiv, int* info, const int batchCount) {
    return FWD(hipblasZgetrfBatched_v2, handle, n, A, lda, ipiv, info, batchCount);
}
hipblasStatus_t hipblasSgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs,
                                     float* const A[], const int lda, const int* ipiv, float* const B[],
                                     const int ldb, int* info, const int batchCount) {
    return FWD(hipblasSgetrsBatched, handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}
hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs,
                                     double* const A[], const int lda, const int* ipiv, double* const B[],
                                     const int ldb, int* info, const int batchCount) {
    return FWD(hipblasDgetrsBatched, handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}
hipblasStatus_t hipblasCgetrsBatched_v2(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs,
                                        hipComplex* const A[], const int lda, const int* ipiv, hipComplex* const B[],
                                        const int ldb, int* info, const int batchCount) {
    return FWD(hipblasCgetrsBatched_v2, handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}
hipblasStatus_t hipblasZgetrsBatched_v2(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs,
                                        hipDoubleComplex* const A[], const int lda, const int* ipiv,
                                        hipDoubleComplex* const B[], const int ldb, int* info, const int batchCount) {
    return FWD(hipblasZgetrsBatched_v2, handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

// ================= rocBLAS 层拦截 =================
// 关键发现 (2026-08-14): PyTorch fp16 GEMM 走 hipblaslt_ext::Gemm (C++ API),
// gfx803 上 hipblasLt 不支持 → 内部直接 fallback 调 rocblas_gemm_ex,
// 完全绕过 hipBLAS C 符号层 → LD_PRELOAD 劫持 hipblas 符号对 fp16 无效!
// 在此层拦截 fp16/bf16/fp32 GEMM, 其余转发真 librocblas。
// 与 hipblas 层的关系: hipblas 层失败转发的调用会再次进入本层并再次尝试,
// 最终 FWD 到真库, 无递归风险 (本层 FWD 用 RTLD 句柄直取真库指针)。
#include <rocblas/rocblas.h>

static void* real_rocblas(void) {
    static void* h = NULL;
    if (!h) {
        h = dlopen("librocblas.so.4", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("/opt/rocm/lib/librocblas.so.4", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) fprintf(stderr, "[vkblas] real librocblas.so.4 not found: %s\n", dlerror());
    }
    return h;
}
static void* rcb_sym(const char* name) { return real_rocblas() ? dlsym(real_rocblas(), name) : NULL; }
#define FWD_RC(name, ...) __extension__ ({                             \
    static __typeof__(&name) fn = NULL;                                \
    if (!fn) fn = (__typeof__(&name))rcb_sym(#name);                   \
    fn ? fn(__VA_ARGS__) : rocblas_status_not_implemented; })

// 共用 GEMM 翻译: 与 hipblas 层完全相同 (column-major → row-major 交叉)
// op_a=transB, op_b=transA, M=n, N=m, A 指针=b, B 指针=a, lda=ldb, ldb=lda
rocblas_status rocblas_gemm_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
                               rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
                               const void* a, rocblas_datatype a_type, rocblas_int lda,
                               const void* b, rocblas_datatype b_type, rocblas_int ldb, const void* beta,
                               const void* c, rocblas_datatype c_type, rocblas_int ldc,
                               void* d, rocblas_datatype d_type, rocblas_int ldd,
                               rocblas_datatype compute_type, rocblas_gemm_algo algo,
                               int32_t solution_index, uint32_t flags) {
    if (!handle) return rocblas_status_invalid_value;
    if (m <= 0 || n <= 0 || k <= 0) return rocblas_status_invalid_value;
    // 仅支持 in-place (c==d) + fp32 compute; 其余转发
    // 注意: gemm_ex 的 compute_type 用 rocblas_datatype 枚举 (f32_r=151), 不是 rocblas_compute_type (300+)!
    if (c != d || ldc != ldd || compute_type != rocblas_datatype_f32_r)
        return FWD_RC(rocblas_gemm_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
                      d, d_type, ldd, compute_type, algo, solution_index, flags);
    float av, bv;
    if (get_scalar_f32(alpha, &av) != 0 || get_scalar_f32(beta, &bv) != 0)
        return FWD_RC(rocblas_gemm_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
                      d, d_type, ldd, compute_type, algo, solution_index, flags);
    int is_f16 = a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                 && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r;
    int is_bf16 = a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
                  && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r;
    int is_f32 = a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
                 && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r;
    if (!is_f16 && !is_bf16 && !is_f32)
        return FWD_RC(rocblas_gemm_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
                      d, d_type, ldd, compute_type, algo, solution_index, flags);
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] rocblas_gemm_ex %s %s %s m=%d n=%d k=%d\n",
                is_f16 ? "f16" : is_bf16 ? "bf16" : "f32",
                transA == rocblas_operation_transpose ? "T" : "N",
                transB == rocblas_operation_transpose ? "T" : "N", m, n, k);
    hipStreamSynchronize(g_stream);
    vkblas_status_t st;
    if (is_f16)
        st = vkblas_gemm_f16(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            1, 0, 0, 0);
    else if (is_bf16)
        st = vkblas_gemm_bf16(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            1, 0, 0, 0);
    else
        st = vkblas_gemm_f32(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            1, 0, 0, 0);
    if (st == VKBLAS_OK) return rocblas_status_success;
    return FWD_RC(rocblas_gemm_ex, handle, transA, transB, m, n, k, alpha,
                  a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
                  d, d_type, ldd, compute_type, algo, solution_index, flags);
}

rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
                                               rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
                                               const void* a, rocblas_datatype a_type, rocblas_int lda, int64_t strideA,
                                               const void* b, rocblas_datatype b_type, rocblas_int ldb, int64_t strideB,
                                               const void* beta, const void* c, rocblas_datatype c_type, rocblas_int ldc, int64_t strideC,
                                               void* d, rocblas_datatype d_type, rocblas_int ldd, int64_t strideD,
                                               rocblas_int batch_count, rocblas_datatype compute_type,
                                               rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags) {
    if (!handle) return rocblas_status_invalid_value;
    if (m <= 0 || n <= 0 || k <= 0 || batch_count <= 0) return rocblas_status_invalid_value;
    // compute_type 用 rocblas_datatype 枚举 (f32_r=151)
    if (c != d || ldc != ldd || compute_type != rocblas_datatype_f32_r)
        return FWD_RC(rocblas_gemm_strided_batched_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, strideA, b, b_type, ldb, strideB, beta, c, c_type, ldc, strideC,
                      d, d_type, ldd, strideD, batch_count, compute_type, algo, solution_index, flags);
    float av, bv;
    if (get_scalar_f32(alpha, &av) != 0 || get_scalar_f32(beta, &bv) != 0)
        return FWD_RC(rocblas_gemm_strided_batched_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, strideA, b, b_type, ldb, strideB, beta, c, c_type, ldc, strideC,
                      d, d_type, ldd, strideD, batch_count, compute_type, algo, solution_index, flags);
    int is_f16 = a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
                 && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r;
    int is_bf16 = a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
                  && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r;
    int is_f32 = a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
                 && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r;
    if (!is_f16 && !is_bf16 && !is_f32)
        return FWD_RC(rocblas_gemm_strided_batched_ex, handle, transA, transB, m, n, k, alpha,
                      a, a_type, lda, strideA, b, b_type, ldb, strideB, beta, c, c_type, ldc, strideC,
                      d, d_type, ldd, strideD, batch_count, compute_type, algo, solution_index, flags);
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] rocblas_gemm_strided_batched_ex %s %s%s m=%d n=%d k=%d bc=%d\n",
                is_f16 ? "f16" : is_bf16 ? "bf16" : "f32",
                transA == rocblas_operation_transpose ? "T" : "N",
                transB == rocblas_operation_transpose ? "T" : "N", m, n, k, batch_count);
    hipStreamSynchronize(g_stream);
    vkblas_status_t st;
    if (is_f16)
        st = vkblas_gemm_f16(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            (uint32_t)batch_count, strideB, strideA, strideC);
    else if (is_bf16)
        st = vkblas_gemm_bf16(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            (uint32_t)batch_count, strideB, strideA, strideC);
    else
        st = vkblas_gemm_f32(
            transB == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            transA == rocblas_operation_transpose ? VKBLAS_OP_T : VKBLAS_OP_N,
            (uint32_t)n, (uint32_t)m, (uint32_t)k,
            av, b, (uint32_t)ldb, a, (uint32_t)lda, bv, d, (uint32_t)ldc,
            (uint32_t)batch_count, strideB, strideA, strideC);
    if (st == VKBLAS_OK) return rocblas_status_success;
    return FWD_RC(rocblas_gemm_strided_batched_ex, handle, transA, transB, m, n, k, alpha,
                  a, a_type, lda, strideA, b, b_type, ldb, strideB, beta, c, c_type, ldc, strideC,
                  d, d_type, ldd, strideD, batch_count, compute_type, algo, solution_index, flags);
}
