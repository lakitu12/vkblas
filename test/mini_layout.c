// mini_layout.c — 直调 vkblas_gemm_bf16_hipblas (col-major 物理) 定位/回归工具
// 用法: ./mini_layout <dtype:0=fp16 1=bf16> [M N K] [op_a op_b] [lda ldb ldc]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include "../src/vkblas.h"

typedef vkblas_status_t (*gemm_fn)(vkblas_op_t, vkblas_op_t, uint32_t, uint32_t, uint32_t,
                                   float, const void*, uint32_t, const void*, uint32_t, float,
                                   void*, uint32_t, uint32_t, int64_t, int64_t, int64_t);

static double refd[512][512];
int main(int argc, char** argv) {
    int dtype = argc > 1 ? atoi(argv[1]) : 1;
    uint32_t M = argc > 4 ? atoi(argv[2]) : 256, N = argc > 4 ? atoi(argv[3]) : 256, K = argc > 4 ? atoi(argv[4]) : 256;
    vkblas_op_t op_a = argc > 6 ? atoi(argv[5]) : VKBLAS_OP_N, op_b = argc > 6 ? atoi(argv[6]) : VKBLAS_OP_T;
    uint32_t lda = argc > 9 ? atoi(argv[7]) : M, ldb = argc > 9 ? atoi(argv[8]) : K, ldc = argc > 9 ? atoi(argv[9]) : N;
    if (M > 8192 || N > 8192 || K > 8192) { fprintf(stderr, "<=8192\n"); return 1; }
    int do_ref = M <= 512 && N <= 512;
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    gemm_fn vk = (gemm_fn)dlsym(so, dtype ? "vkblas_gemm_bf16_hipblas" : "vkblas_gemm_f16");
    if (!vk) { fprintf(stderr, "dlsym fail\n"); return 2; }
    hipInit(0); hipSetDevice(0);
    unsigned short *hA = malloc((size_t)M * K * 2), *hB = malloc((size_t)K * N * 2);
    srand(777);
    for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = (unsigned short)(rand() % 0x400 + 0x3800) | ((rand() & 1) << 15);
    for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = (unsigned short)(rand() % 0x400 + 0x3800) | ((rand() & 1) << 15);
    if (do_ref) {
        for (int m = 0; m < (int)M; m++) for (int n = 0; n < (int)N; n++) {
            double s = 0;
            for (int k = 0; k < (int)K; k++) {
                uint32_t ea = (op_a == VKBLAS_OP_T) ? (uint32_t)k * lda + m : (uint32_t)m * lda + k;
                uint32_t eb = (op_b == VKBLAS_OP_T) ? (uint32_t)n * ldb + k : (uint32_t)k * ldb + n;
                unsigned ua = hA[ea], ub = hB[eb];
                double fa, fb;
                if (dtype) { fa = (double)((ua & 0x8000) ? -1 : 1) * (double)(ua & 0x7FFF); fb = (double)((ub & 0x8000) ? -1 : 1) * (double)(ub & 0x7FFF); }
                else {
                    int sa = (ua >> 15) ? -1 : 1, sb = (ub >> 15) ? -1 : 1;
                    int ea_ = ((ua >> 10) & 0x1F) - 15, eb_ = ((ub >> 10) & 0x1F) - 15;
                    fa = sa * ldexp(1.0 + ((double)(ua & 0x3FF) / 1024.0), ea_);
                    fb = sb * ldexp(1.0 + ((double)(ub & 0x3FF) / 1024.0), eb_);
                }
                s += fa * fb;
            }
            refd[m][n] = s;
        }
    }
    void *dA, *dB, *dC;
    hipMalloc(&dA, (size_t)M * K * 2); hipMalloc(&dB, (size_t)K * N * 2); hipMalloc(&dC, (size_t)M * N * 2);
    hipMemcpy(dA, hA, (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, (size_t)K * N * 2, hipMemcpyHostToDevice);
    memset(hA, 0, (size_t)M * K * 2);
    vkblas_status_t st = vk(op_a, op_b, M, N, K, 1.0f, dA, lda, dB, ldb, 0.0f, dC, ldc, 1, 0, 0, 0);
    hipDeviceSynchronize();
    hipMemcpy(hA, dC, (size_t)M * N * 2, hipMemcpyDeviceToHost);
    if (!do_ref) {
        printf("dtype=%s %s%s M=%u N=%u K=%u lda=%u ldb=%u ldc=%u: st=%d (no ref, ran ok)\n",
               dtype ? "bf16" : "f16 ", op_a ? "T" : "N", op_b ? "T" : "N", M, N, K, lda, ldb, ldc, (int)st);
        return 0;
    }
    int nbad = 0, shown = 0; double maxr = 0;
    for (int m = 0; m < (int)M; m++) for (int n = 0; n < (int)N; n++) {
        unsigned uc = hA[(size_t)m * ldc + n];
        double got = (double)((uc & 0x8000) ? -1 : 1) * (double)(uc & 0x7FFF);
        if (!dtype) {
            int sc = (uc >> 15) ? -1 : 1; int ec = ((uc >> 10) & 0x1F) - 15;
            got = sc * ldexp(1.0 + ((double)(uc & 0x3FF) / 1024.0), ec);
        }
        double r = refd[m][n] != 0 ? (got - refd[m][n]) / refd[m][n] : got;
        if (r > maxr) maxr = r;
        if (r > 0.01) { nbad++; if (shown < 12) { printf("  bad[%d][%d] got=%.2f ref=%.2f\n", m, n, got, refd[m][n]); shown++; } }
    }
    printf("dtype=%s %s%s M=%u N=%u K=%u lda=%u ldb=%u ldc=%u: st=%d nbad=%d max_rel=%.4f\n",
           dtype ? "bf16" : "f16 ", op_a ? "T" : "N", op_b ? "T" : "N", M, N, K, lda, ldb, ldc, (int)st, nbad, maxr);
    return 0;
}