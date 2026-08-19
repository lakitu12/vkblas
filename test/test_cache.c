// test_cache.c — import 缓存 (HIT 路径) 正确性回归。LD_PRELOAD 运行:
//   LD_PRELOAD=./libvkblas_hipblas.so ./test/test_cache
// 覆盖: ① 同 ptr 多次 GEMM (hit) 且内容变化 ② free→同址复用 (hook 失效) ③ size 变化
#include <hipblas/hipblas.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float ref_cm(const float* A, const float* B, int m, int n, int k, int lda, int ldb) {
    // hipBLAS 列主序参照: A(m×k, lda), B(k×n, ldb), C[r + c*ldc] = sum_q A[r+q*lda]*B[q+c*ldb]
    // 返回 C 的最大元素 (全非负输入下 max 与索引布局无关, 但求和顺序必须列主序)
    float mx = 0;
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < m; r++) {
            float s = 0;
            for (int q = 0; q < k; q++) s += A[r + q * lda] * B[q + c * ldb];
            if (s > mx) mx = s;
        }
    }
    return mx;
}

static int run_gemm(hipblasHandle_t h, int m, int n, int k,
                    const float* hA, const float* hB, float expect_max) {
    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, (size_t)m * k * 4);
    hipMalloc((void**)&dB, (size_t)k * n * 4);
    hipMalloc((void**)&dC, (size_t)m * n * 4);
    hipMemcpy(dA, hA, (size_t)m * k * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, (size_t)k * n * 4, hipMemcpyHostToDevice);
    hipMemset(dC, 0, (size_t)m * n * 4);
    float alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t st = hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                      m, n, k, &alpha, dA, m, dB, k, &beta, dC, m);
    hipDeviceSynchronize();
    float* hGot = malloc((size_t)m * n * 4);
    hipMemcpy(hGot, dC, (size_t)m * n * 4, hipMemcpyDeviceToHost);
    float mx = 0;
    for (int i = 0; i < m * n; i++) if (hGot[i] > mx) mx = hGot[i];
    int ok = (st == HIPBLAS_STATUS_SUCCESS) && mx == expect_max;
    printf("  gemm %dx%dx%d: st=%d max=%.0f expect=%.0f -> %s\n",
           m, n, k, st, mx, expect_max, ok ? "PASS" : "FAIL");
    free(hGot);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return ok ? 0 : 1;
}

int main(void) {
    hipblasHandle_t h;
    hipblasCreate(&h);
    int fails = 0;

    // --- ① 持久指针: 同一 ptr 连续 3 次 GEMM, 内容每次改写 (命中路径 + dma-buf 读到新数据)
    printf("[A] same-ptr reuse, content changed between calls:\n");
    float* dAp; float* dBp; float* dCp;
    hipMalloc((void**)&dAp, 64 * 64 * 4);
    hipMalloc((void**)&dBp, 64 * 64 * 4);
    hipMalloc((void**)&dCp, 64 * 64 * 4);
    for (int it = 0; it < 3; it++) {
        float* hA = malloc(64 * 64 * 4); float* hB = malloc(64 * 64 * 4);
        for (int i = 0; i < 64 * 64; i++) { hA[i] = (float)((i + it) % 7); hB[i] = (float)((i * 3 + it) % 5); }
        hipMemcpy(dAp, hA, 64 * 64 * 4, hipMemcpyHostToDevice);
        hipMemcpy(dBp, hB, 64 * 64 * 4, hipMemcpyHostToDevice);
        hipMemset(dCp, 0, 64 * 64 * 4);
        float alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t st = hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, 64,
                                          &alpha, dAp, 64, dBp, 64, &beta, dCp, 64);
        hipDeviceSynchronize();
        float mx = 0;
        float* hGot = malloc(64 * 64 * 4);
        hipMemcpy(hGot, dCp, 64 * 64 * 4, hipMemcpyDeviceToHost);
        for (int i = 0; i < 64 * 64; i++) if (hGot[i] > mx) mx = hGot[i];
        float exp = ref_cm(hA, hB, 64, 64, 64, 64, 64);
        int ok = (st == HIPBLAS_STATUS_SUCCESS) && mx == exp;
        printf("  iter %d: max=%.0f expect=%.0f -> %s\n", it, mx, exp, ok ? "PASS" : "FAIL");
        fails += !ok;
        free(hA); free(hB); free(hGot);
    }
    // B 是不变权重 (同 ptr 两种用途 mix)
    hipFree(dBp);

    // --- ② free → 同址复用 (hook 失效后必须重导) ---
    printf("[B] free -> same-address reuse, different content/size:\n");
    float* q1; hipMalloc((void**)&q1, 100 * 100 * 4);   // 40KB
    hipFree(q1);
    float* q2; hipMalloc((void**)&q2, 100 * 100 * 4);   // 期望同址
    fails += (q2 != q1);
    printf("  addr reuse: q1=%p q2=%p %s\n", q1, q2, q2 == q1 ? "(same, as expected)" : "(DIFFERENT)");
    if (q2 == q1) {
        float* hA2 = malloc(100 * 200 * 4); float* hB2 = malloc(200 * 50 * 4);
        for (int i = 0; i < 100 * 200; i++) hA2[i] = (float)(i % 3);
        for (int i = 0; i < 200 * 50; i++) hB2[i] = (float)((i % 4) + 1.0f);
        fails += run_gemm(h, 100, 50, 200, hA2, hB2, ref_cm(hA2, hB2, 100, 50, 200, 100, 200));
        free(hA2); free(hB2);
    }
    hipFree(q2);

    // --- ③ 大 size 变化: 同址 4KB → 64KB (size 不够必须重导) ---
    printf("[C] size growth on reused address:\n");
    float* r1; hipMalloc((void**)&r1, 32 * 32 * 4);   // 4KB
    hipFree(r1);
    float* r2; hipMalloc((void**)&r2, 128 * 128 * 4); // 64KB, 期望同址
    printf("  addr: r1=%p r2=%p %s\n", r1, r2, r2 == r1 ? "(same)" : "(different)");
    if (r2 == r1) {
        float* hA3 = malloc(128 * 128 * 4); float* hB3 = malloc(128 * 128 * 4);
        for (int i = 0; i < 128 * 128; i++) { hA3[i] = (float)((i * 7) % 9); hB3[i] = (float)((i * 11) % 6); }
        fails += run_gemm(h, 128, 128, 128, hA3, hB3, ref_cm(hA3, hB3, 128, 128, 128, 128, 128));
        free(hA3); free(hB3);
    }
    hipFree(r2);

    hipFree(dAp); hipFree(dCp);
    hipblasDestroy(h);
    printf("%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}