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
    hipStreamSynchronize(g_stream);  // 等调用方 HIP 工作完成
    vkblas_status_t st = vkblas_gemm_f32(
        transB == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        transA == HIPBLAS_OP_T ? VKBLAS_OP_T : VKBLAS_OP_N,
        (uint32_t)n, (uint32_t)m, (uint32_t)k,
        a, BP, (uint32_t)ldb, AP, (uint32_t)lda, b, CP, (uint32_t)ldc,
        1, 0, 0, 0);
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
static int is_bf16_ex(hipDataType t) { return t == HIP_R_16BF; }

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
    if (is_fp32_ex(aType, computeType) && is_fp32_ex(bType, computeType) && cType == HIP_R_32F)
        return vk_gemm_f32(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (is_bf16_ex(aType) && is_bf16_ex(bType) && is_bf16_ex(cType))
        return vk_gemm_bf16(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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
    return FWD(hipblasZgemm_v2, handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
}
hipblasStatus_t hipblasZgemmStridedBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
                                              int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, long long strideA,
                                              const hipDoubleComplex* BP, int ldb, long long strideB, const hipDoubleComplex* beta,
                                              hipDoubleComplex* CP, int ldc, long long strideC, int batchCount) {
    return FWD(hipblasZgemmStridedBatched_v2, handle, transA, transB, m, n, k, alpha, AP, lda, strideA,
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
