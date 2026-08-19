// test_gemm.c — vkblas 正确性 + 性能测试
// 直链真 hipblas 做参照; dlopen 我们的 libvkblas_hipblas.so 取 Vulkan 实现
// 用法: ./test_gemm [--perf]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime_api.h>

// 我们 .so 里 hipblasSgemm / hipblasCreate / hipblasDestroy 的签名 (与真库一致)
typedef hipblasStatus_t (*sgemm_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                    int64_t, int64_t, int64_t, const float*, const float*, int64_t,
                                    const float*, int64_t, const float*, float*, int64_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);
typedef hipblasStatus_t (*destroy_fn)(hipblasHandle_t);
typedef hipblasStatus_t (*sgemm_batch_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                          int64_t, int64_t, int64_t, const float*, const float*, int64_t,
                                          int64_t, const float*, int64_t, int64_t, const float*,
                                          float*, int64_t, int64_t, int64_t);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// CPU 参照: hipBLAS column-major 语义 (ground truth, 不依赖真 hipblas)
// 大矩阵降采样: 只算 sample_rows 行 (0 = 全部), 避免 4096³ 的 O(n³) 拖死测试
static void ref_gemm(hipblasOperation_t transa, hipblasOperation_t transb,
                     int m, int n, int k, float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc,
                     int sample_rows) {
    int rows[32];
    int nrows = m;
    if (sample_rows > 0 && sample_rows < m) {
        nrows = sample_rows;
        for (int r = 0; r < sample_rows; r++)
            rows[r] = (int)((int64_t)r * m / sample_rows);  // 均匀采样
        rows[sample_rows - 1] = m - 1;
    }
    for (int ri = 0; ri < nrows; ri++) {
        int i = sample_rows > 0 && sample_rows < m ? rows[ri] : ri;
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int t = 0; t < k; t++) {
                // hipBLAS 语义: op(A)[i][t] = A[i + t*lda] (N) / A[t + i*lda] (T)
                //               op(B)[t][j] = B[t + j*ldb] (N) / B[j + t*ldb] (T)
                float av = (transa == HIPBLAS_OP_T) ? A[t + (size_t)i * lda] : A[i + (size_t)t * lda];
                float bv = (transb == HIPBLAS_OP_T) ? B[j + (size_t)t * ldb] : B[t + (size_t)j * ldb];
                s += (double)av * bv;
            }
            C[i + (size_t)j * ldc] = alpha * (float)s + beta * C[i + (size_t)j * ldc];
        }
    }
}

static int run_case(sgemm_fn vk_sgemm, create_fn vk_create, destroy_fn vk_destroy, const char* name,
                    hipblasOperation_t transa, hipblasOperation_t transb,
                    int m, int n, int k, int lda, int ldb, int ldc,
                    float alpha, float beta, int seed) {
    // host 数据 (column-major 语义存储; 大小按 trans 决定的内存布局)
    size_t szA = (transa == HIPBLAS_OP_T) ? (size_t)lda * m : (size_t)lda * k;
    size_t szB = (transb == HIPBLAS_OP_T) ? (size_t)ldb * k : (size_t)ldb * n;
    size_t szC = (size_t)ldc * n;
    float *hA = malloc(szA * 4), *hB = malloc(szB * 4), *hC = malloc(szC * 4);
    float *hRef = malloc(szC * 4), *hGot = malloc(szC * 4);
    srand(seed);
    for (size_t i = 0; i < szA; i++) hA[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (size_t i = 0; i < szB; i++) hB[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (size_t i = 0; i < szC; i++) hC[i] = (float)(rand() % 200 - 100) / 10.0f;

    float *dA, *dB, *dC2;
    hipMalloc((void**)&dA, szA * 4); hipMalloc((void**)&dB, szB * 4);
    hipMalloc((void**)&dC2, szC * 4);
    if (getenv("VKBLAS_PROBE")) {
        fprintf(stderr, "[probe] %s: dA=%p(%zu) dB=%p(%zu) dC=%p(%zu)\n",
                name, (void*)dA, szA * 4, (void*)dB, szB * 4, (void*)dC2, szC * 4);
    }
    hipMemcpy(dA, hA, szA * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, szB * 4, hipMemcpyHostToDevice);
    hipMemcpy(dC2, hC, szC * 4, hipMemcpyHostToDevice);

    // CPU ground truth (大矩阵降采样)
    int sample = ((int64_t)m * n * k > 5000000) ? 12 : 0;
    memcpy(hRef, hC, szC * 4);
    ref_gemm(transa, transb, m, n, k, alpha, hA, lda, hB, ldb, beta, hRef, ldc, sample);

    // 我们的 Vulkan 实现 (host pointer mode) — 必须配我们自己的 handle
    hipblasHandle_t vh;
    vk_create(&vh);
    hipblasStatus_t st2 = vk_sgemm(vh, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC2, ldc);
    hipDeviceSynchronize();
    vk_destroy(vh);

    hipMemcpy(hGot, dC2, szC * 4, hipMemcpyDeviceToHost);

    double maxd = 0;
    if (sample > 0) {
        // 只对比采样行 (与 ref_gemm 相同的采样)
        for (int r = 0; r < sample; r++) {
            int i = (int)((int64_t)r * m / sample);
            if (r == sample - 1) i = m - 1;
            for (int j = 0; j < n; j++) {
                double d = fabs((double)hRef[i + (size_t)j * ldc] - hGot[i + (size_t)j * ldc]);
                if (d > maxd) maxd = d;
            }
        }
    } else {
        for (size_t i = 0; i < (size_t)m * n; i++) {
            double d = fabs((double)hRef[i] - hGot[i]);
            if (d > maxd) maxd = d;
        }
    }
    int ok = (st2 == HIPBLAS_STATUS_SUCCESS && maxd < 0.1);  // 索引错位时 diff≥100, 精度误差≤0.03
    printf("%-38s M=%-4d N=%-4d K=%-4d | vk=%s | maxdiff=%.2e | %s\n",
           name, m, n, k,
           st2 == HIPBLAS_STATUS_SUCCESS ? "ok" : "ERR",
           maxd, ok ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef); free(hGot);
    hipFree(dA); hipFree(dB); hipFree(dC2);
    return ok ? 0 : 1;
}

// ---- strided_batched 多 batch 测试: 单次 hipblasSgemmStridedBatched vs CPU 逐 batch 参照 ----
static int run_batch_case(sgemm_batch_fn vk_sgemm_batch, create_fn vk_create, destroy_fn vk_destroy,
                          const char* name,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k, int lda, int ldb, int ldc,
                          int64_t strideA, int64_t strideB, int64_t strideC, int batch,
                          float alpha, float beta, int seed) {
    // 单 batch 尺寸 (元素) + 全 batch 总大小
    size_t szA1 = (transa == HIPBLAS_OP_T) ? (size_t)lda * m : (size_t)lda * k;
    size_t szB1 = (transb == HIPBLAS_OP_T) ? (size_t)ldb * k : (size_t)ldb * n;
    size_t szC1 = (size_t)ldc * n;
    size_t szA = (strideA > 0 ? (size_t)strideA * (batch - 1) : 0) + szA1;
    size_t szB = (strideB > 0 ? (size_t)strideB * (batch - 1) : 0) + szB1;
    size_t szC = (strideC > 0 ? (size_t)strideC * (batch - 1) : 0) + szC1;
    float *hA = malloc(szA * 4), *hB = malloc(szB * 4), *hC = malloc(szC * 4);
    float *hRef = malloc(szC * 4), *hGot = malloc(szC * 4);
    srand(seed);
    for (size_t i = 0; i < szA; i++) hA[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (size_t i = 0; i < szB; i++) hB[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (size_t i = 0; i < szC; i++) hC[i] = (float)(rand() % 200 - 100) / 10.0f;
    memcpy(hRef, hC, szC * 4);
    memcpy(hGot, hC, szC * 4);

    // CPU 参照: 逐 batch 调 ref_gemm (stride 为 0 时广播 = 所有 batch 共享同一份)
    int sample = ((int64_t)m * n * k > 5000000) ? 12 : 0;
    for (int b = 0; b < batch; b++) {
        const float* Ab = hA + (strideA > 0 ? strideA * b : 0);
        const float* Bb = hB + (strideB > 0 ? strideB * b : 0);
        float* Cb = hRef + (strideC > 0 ? strideC * b : 0);
        ref_gemm(transa, transb, m, n, k, alpha, Ab, lda, Bb, ldb, beta, Cb, ldc, sample);
    }

    float *dA, *dB, *dC2;
    hipMalloc((void**)&dA, szA * 4); hipMalloc((void**)&dB, szB * 4);
    hipMalloc((void**)&dC2, szC * 4);
    hipMemcpy(dA, hA, szA * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, szB * 4, hipMemcpyHostToDevice);
    hipMemcpy(dC2, hGot, szC * 4, hipMemcpyHostToDevice);

    hipblasHandle_t vh;
    vk_create(&vh);
    hipblasStatus_t st2 = vk_sgemm_batch(vh, transa, transb, m, n, k,
                                         &alpha, dA, lda, strideA, dB, ldb, strideB,
                                         &beta, dC2, ldc, strideC, batch);
    hipDeviceSynchronize();
    vk_destroy(vh);

    hipMemcpy(hGot, dC2, szC * 4, hipMemcpyDeviceToHost);

    double maxd = 0;
    if (sample > 0) {
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < sample; r++) {
                int i = (int)((int64_t)r * m / sample);
                if (r == sample - 1) i = m - 1;
                for (int j = 0; j < n; j++) {
                    size_t off = (strideC > 0 ? strideC * b : 0) + i + (size_t)j * ldc;
                    double d = fabs((double)hRef[off] - hGot[off]);
                    if (d > maxd) maxd = d;
                }
            }
        }
    } else {
        for (size_t i = 0; i < szC; i++) {
            double d = fabs((double)hRef[i] - hGot[i]);
            if (d > maxd) maxd = d;
        }
    }
    int ok = (st2 == HIPBLAS_STATUS_SUCCESS && maxd < 0.1);
    printf("BATCH %-32s M=%-4d N=%-4d K=%-4d b=%-3d | vk=%s | maxdiff=%.2e | %s\n",
           name, m, n, k, batch,
           st2 == HIPBLAS_STATUS_SUCCESS ? "ok" : "ERR",
           maxd, ok ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hRef); free(hGot);
    hipFree(dA); hipFree(dB); hipFree(dC2);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    int perf = argc > 1 && strcmp(argv[1], "--perf") == 0;

    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!so) { fprintf(stderr, "dlopen libvkblas_hipblas.so failed: %s\n", dlerror()); return 2; }
    sgemm_fn vk_sgemm = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn vk_create = (create_fn)dlsym(so, "hipblasCreate");
    destroy_fn vk_destroy = (destroy_fn)dlsym(so, "hipblasDestroy");
    sgemm_batch_fn vk_sgemm_batch = (sgemm_batch_fn)dlsym(so, "hipblasSgemmStridedBatched");
    if (!vk_sgemm || !vk_create || !vk_destroy || !vk_sgemm_batch) {
        fprintf(stderr, "dlsym hipblas symbols failed\n");
        return 2;
    }

    hipInit(0);
    hipSetDevice(0);

    if (!perf) {
        int fails = 0;
        hipblasOperation_t ops[2] = { HIPBLAS_OP_N, HIPBLAS_OP_T };
        struct { int m, n, k; } shapes[] = {
            {32, 32, 32}, {64, 64, 64}, {128, 128, 128}, {257, 129, 64}, {100, 200, 300},
            {256, 256, 4096}, {512, 512, 4096}, {512, 640, 4096}, {512, 768, 4096},
            {4096, 4096, 4096}, {4096, 1024, 4096}, {1024, 4096, 4096}, {512, 512, 512},
            {33, 65, 17}, {1, 64, 64}, {64, 1, 64}, {3, 5, 7},
        };
        for (int s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++) {
            for (int ta = 0; ta < 2; ta++) {
                for (int tb = 0; tb < 2; tb++) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "%s%s %dx%dx%d",
                             ta ? "T" : "N", tb ? "T" : "N", shapes[s].m, shapes[s].n, shapes[s].k);
                    // 紧密 ld (hipBLAS 合法性: N→lda≥m / T→lda≥k, 同理 ldb, ldc≥m)
                    int lda = ta ? shapes[s].k : shapes[s].m;
                    int ldb = tb ? shapes[s].n : shapes[s].k;
                    int ldc = shapes[s].m;
                    fails += run_case(vk_sgemm, vk_create, vk_destroy, nm, ops[ta], ops[tb],
                                      shapes[s].m, shapes[s].n, shapes[s].k,
                                      lda, ldb, ldc,
                                      1.0f, 0.0f, 42 + s * 10 + ta * 2 + tb);
                }
            }
        }
        // alpha/beta 非平凡 + 非紧密 ld
        fails += run_case(vk_sgemm, vk_create, vk_destroy, "alpha=0.5 beta=0.7", HIPBLAS_OP_T, HIPBLAS_OP_T,
                          256, 128, 64, 64, 128, 256, 0.5f, 0.7f, 7);
        fails += run_case(vk_sgemm, vk_create, vk_destroy, "padded lda=140 ldb=80 ldc=300", HIPBLAS_OP_N, HIPBLAS_OP_T,
                          128, 64, 64, 140, 80, 300, 1.0f, 0.0f, 8);
        fails += run_case(vk_sgemm, vk_create, vk_destroy, "padded both T", HIPBLAS_OP_T, HIPBLAS_OP_T,
                          100, 50, 80, 120, 90, 150, 1.0f, 1.0f, 9);

        // ---- strided_batched 多 batch (batch 合并快路径) ----
        // 变体覆盖: NN (转置合并) / TT (直接合并) / 非 128 倍数 (v6) / 256+ (v7-128)
        // alpha/beta / 非紧密 stride / 广播 stride=0
        hipblasOperation_t ops2[2] = { HIPBLAS_OP_N, HIPBLAS_OP_T };
        struct { int m, n, k, lda, ldb, ldc; int64_t sa, sb, sc; int batch; } bshapes[] = {
            {64, 64, 64, 64, 64, 64, 4096, 4096, 4096, 8},      // NN/TT 紧密
            {128, 128, 128, 128, 128, 128, 16384, 16384, 16384, 6},  // v7-128 边界
            {256, 256, 256, 256, 256, 256, 65536, 65536, 65536, 4},  // v7-128
            {33, 65, 17, 40, 70, 40, 2800, 1190, 2600, 5},      // 奇数+padding, v6 (strideC=ldc*n)
            {100, 200, 300, 320, 310, 110, 32000, 62000, 22000, 3},  // 大 K, 非紧密 stride (lda≥K 320/310)
        };
        for (int s = 0; s < (int)(sizeof(bshapes) / sizeof(bshapes[0])); s++) {
            for (int ta = 0; ta < 2; ta++) {
                for (int tb = 0; tb < 2; tb++) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "%s%s %dx%dx%d", ta ? "T" : "N", tb ? "T" : "N",
                             bshapes[s].m, bshapes[s].n, bshapes[s].k);
                    fails += run_batch_case(vk_sgemm_batch, vk_create, vk_destroy, nm,
                                            ops2[ta], ops2[tb],
                                            bshapes[s].m, bshapes[s].n, bshapes[s].k,
                                            bshapes[s].lda, bshapes[s].ldb, bshapes[s].ldc,
                                            bshapes[s].sa, bshapes[s].sb, bshapes[s].sc,
                                            bshapes[s].batch, 1.0f, 0.0f, 100 + s * 10 + ta * 2 + tb);
                }
            }
        }
        // alpha/beta 非平凡 + 广播 stride=0 (A/B 共享, C 独立)
        fails += run_batch_case(vk_sgemm_batch, vk_create, vk_destroy, "alpha=0.5 beta=0.7 b=4",
                                HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, 64, 64, 64, 64,
                                4096, 4096, 4096, 4, 0.5f, 0.7f, 7);
        fails += run_batch_case(vk_sgemm_batch, vk_create, vk_destroy, "broadcast strideA=0",
                                HIPBLAS_OP_N, HIPBLAS_OP_N, 96, 48, 32, 96, 32, 96,
                                0, 1536, 4608, 6, 1.0f, 0.0f, 8);
        fails += run_batch_case(vk_sgemm_batch, vk_create, vk_destroy, "broadcast strideB=0",
                                HIPBLAS_OP_T, HIPBLAS_OP_T, 48, 96, 32, 32, 96, 48,
                                1536, 0, 4608, 6, 1.0f, 1.0f, 9);
        printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
        return fails == 0 ? 0 : 1;
    }

    // ---- 性能 ----
    const int M = 4096, N = 4096, K = 4096;
    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, (size_t)M * K * 4); hipMalloc((void**)&dB, (size_t)K * N * 4);
    hipMalloc((void**)&dC, (size_t)M * N * 4);
    hipMemset(dA, 1, (size_t)M * K * 4); hipMemset(dB, 1, (size_t)K * N * 4);

    hipblasHandle_t vh;
    vk_create(&vh);
    float alpha = 1.0f, beta = 0.0f;
    double best_vk = 1e9;
    for (int it = 0; it < 3; it++) {
        double t0 = now_s();
        vk_sgemm(vh, HIPBLAS_OP_T, HIPBLAS_OP_T, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        hipDeviceSynchronize();
        double dt = now_s() - t0;
        if (dt < best_vk) best_vk = dt;
    }
    vk_destroy(vh);
    printf("Vulkan hipblasSgemm %dx%dx%d (TT, PyTorch 常用): %.1f ms | %.2f TFLOPS\n",
           M, N, K, best_vk * 1000, 2.0 * M * N * K / best_vk / 1e12);
    return 0;
}
