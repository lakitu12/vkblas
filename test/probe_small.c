// probe_small.c — 分解小 batch GEMM 的时间瓶颈
// 测量: 首次调用, 缓存命中, 拆分为 import vs compute
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*sgemm_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                    int64_t, int64_t, int64_t, const float*, const float*, int64_t,
                                    const float*, int64_t, const float*, float*, int64_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);
typedef hipblasStatus_t (*destroy_fn)(hipblasHandle_t);

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double call_nn(sgemm_fn f, hipblasHandle_t h,
                      int M, int N, int K,
                      float* dA, float* dB, float* dC) {
    float al = 1.0f, be = 0.0f;
    double t0 = now_s();
    hipblasStatus_t st = f(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
    hipDeviceSynchronize();
    double dt = now_s() - t0;
    if (st != HIPBLAS_STATUS_SUCCESS) fprintf(stderr, "Sgemm failed: %d\n", st);
    return dt;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    unsetenv("VKBLAS_TILE128");  // 默认路径

    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!so) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 2; }
    sgemm_fn vk = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    destroy_fn vk_destroy = (destroy_fn)dlsym(so, "hipblasDestroy");
    if (!vk || !cr || !vk_destroy) { fprintf(stderr, "dlsym failed\n"); return 2; }

    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);

    // 测试 shape: decode-1 (1×4096×4096)
    int M = 1, N = 4096, K = 4096;
    size_t sa = (size_t)M * K, sb = (size_t)K * N, sc = (size_t)M * N;
    float *hA = malloc(sa * 4), *hB = malloc(sb * 4), *hC = malloc(sc * 4);
    for (size_t i = 0; i < sa; i++) hA[i] = (rand() % 1000 - 500) / 500.0f;
    for (size_t i = 0; i < sb; i++) hB[i] = (rand() % 1000 - 500) / 500.0f;

    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, sa * 4);
    hipMalloc((void**)&dB, sb * 4);
    hipMalloc((void**)&dC, sc * 4);
    hipMemcpy(dA, hA, sa * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sb * 4, hipMemcpyHostToDevice);

    printf("=== decode-1 (M=1 N=4096 K=4096) 时间分解 ===\n");
    printf("调用 #  时间(ms)  备注\n");
    printf("----------------------------------------\n");

    // 连续调用 10 次, 观察缓存加热效果
    for (int i = 0; i < 10; i++) {
        double dt = call_nn(vk, h, M, N, K, dA, dB, dC);
        const char* note = (i == 0) ? "首次 (含引擎初始化+import)" :
                           (i == 1) ? "缓存命中" : "";
        printf("%4d    %8.3f   %s\n", i + 1, dt * 1000, note);
    }

    // 对比: 不同 M 值的开销
    printf("\n=== M 伸缩性 ===\n");
    int ms[] = {1, 16, 128, 256, 512, 1024, 2048, 4096};
    printf("M      时间(ms)  TFLOPS\n");
    printf("------------------------\n");
    for (int mi = 0; mi < 8; mi++) {
        int m = ms[mi];
        size_t sa2 = (size_t)m * K, sc2 = (size_t)m * N;
        float *dA2, *dC2;
        hipMalloc((void**)&dA2, sa2 * 4);
        hipMalloc((void**)&dC2, sc2 * 4);
        hipMemcpy(dA2, hA, sa2 * 4, hipMemcpyHostToDevice);
        // 预热: 2 次调用建立缓存
        call_nn(vk, h, m, N, K, dA2, dB, dC2);
        call_nn(vk, h, m, N, K, dA2, dB, dC2);
        // 计时: best of 3
        double best = 1e9;
        for (int it = 0; it < 3; it++) {
            double dt = call_nn(vk, h, m, N, K, dA2, dB, dC2);
            if (dt < best) best = dt;
        }
        double flops = 2.0 * m * N * K;
        printf("%-6d   %8.3f   %.2f\n", m, best * 1000, flops / best / 1e12);
        hipFree(dA2); hipFree(dC2);
    }

    // 测试: 只用 vkQueueWaitIdle 的耗时 (空同步)
    printf("\n=== 空同步开销 (vkQueueWaitIdle) ===\n");
    double t0 = now_s();
    for (int i = 0; i < 5; i++) {
        hipDeviceSynchronize();
    }
    double sync_time = (now_s() - t0) / 5;
    printf("hipDeviceSynchronize: %.3f ms\n", sync_time * 1000);

    free(hA); free(hB); free(hC);
    hipFree(dA); hipFree(dB); hipFree(dC);
    vk_destroy(h);
    return 0;
}