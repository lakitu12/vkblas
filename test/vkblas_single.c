// vkblas_single.c — 单 case 复现: 直接调 vkblas_gemm_f32
// 用法: vkblas_single <M> <N> <K> [op_a] [op_b]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include "../src/vkblas.h"

int main(int argc, char** argv) {
    uint32_t M = 4096, N = 4096, K = 4096;
    vkblas_op_t op_a = VKBLAS_OP_N, op_b = VKBLAS_OP_N;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) op_a = atoi(argv[4]) ? VKBLAS_OP_T : VKBLAS_OP_N;
    if (argc >= 6) op_b = atoi(argv[5]) ? VKBLAS_OP_T : VKBLAS_OP_N;

    hipInit(0);
    hipSetDevice(0);
    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, (size_t)M * K * 4);
    hipMalloc((void**)&dB, (size_t)K * N * 4);
    hipMalloc((void**)&dC, (size_t)M * N * 4);
    hipMemset(dA, 1, (size_t)M * K * 4);
    hipMemset(dB, 1, (size_t)K * N * 4);

    fprintf(stderr, "run vkblas_gemm_f32 M=%u N=%u K=%u op_a=%d op_b=%d ...\n", M, N, K, op_a, op_b);
    vkblas_status_t st = vkblas_gemm_f32(op_a, op_b, M, N, K, 1.0f, dA, M, dB, K, 0.0f, dC, M, 1, 0, 0, 0);
    fprintf(stderr, "done, status=%d\n", st);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
