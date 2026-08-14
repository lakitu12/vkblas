// vkblas.h — Vulkan BLAS 核心(引擎 + GEMM)
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef enum {
    VKBLAS_OK = 0,
    VKBLAS_ERR_INIT,     // Vulkan/HSA 初始化失败
    VKBLAS_ERR_IMPORT,   // dma-buf 导入失败(调用方应 fallback 转发)
    VKBLAS_ERR_PARAM,    // 参数错误
} vkblas_status_t;

typedef enum { VKBLAS_OP_N = 0, VKBLAS_OP_T = 1 } vkblas_op_t;

// 核心 GEMM, row-major 语义(host 侧已完成 hipBLAS 参数翻译):
//   C[M,N] = alpha * A_eff[M,K] @ B_eff[K,N] + beta * C_old
//   A_eff: op_a==T 时读 A[k*lda+m], 否则 A[m*lda+k]; B_eff 同理
// 指针必须是 HIP 设备指针(HSA GPU agent 内存), 可导出 dma-buf
// batch: 批量数; stride_*: 各 batch 指针步长(元素数)
// beta==0 时不读 C 内存
vkblas_status_t vkblas_gemm_f32(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    float alpha,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    float beta,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c);

// bf16 GEMM: 语义同 f32, 但 A/B/C 为 bf16 (2B/元素)
// 内部: bf16 → fp32 (cvt shader) → fp32 GEMM → fp32 → bf16 回写
// 所有 stride/ld 均为元素单位
vkblas_status_t vkblas_gemm_bf16(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    float alpha,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    float beta,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c);
