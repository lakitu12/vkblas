// bench_shapes.c — v6/v7 tile 对比扫描 (同一进程内切 VKBLAS_TILE128 环境变量)
// 每个 shape: 正确性校验 + best-of-3 计时
//   校验策略 (控制 CPU 参考成本):
//     M*N*K <= 1024³: 全量 CPU 参考 (ikj 缓存友好序)
//     更大: v6 vs v7 输出等价性 + 512 个随机元素 spot-check 对 CPU dot
// 用法: HSA_OVERRIDE_GFX_VERSION=8.0.3 LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib ./test/bench_shapes
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

// row-major C(M×N) = A(M×K)·B(K×N) 的 hipblas 列主序翻译 (NN 约定):
//   vkblas_hipblas.c 翻译层: op_a=transB, op_b=transA, M_vk=n, N_vk=m, A_vk=BP, B_vk=AP
//   NN 下: A_vk=dA(M×K), B_vk=dB(K×N), M_vk=M, N_vk=N → C=dA·dB=A_rm·B_rm=C_rm ✓
//   (TT 仅对方阵 (M=K) 合法, 非方阵 op_a=T 时访问 A[k*lda+m] 越界)
static hipblasStatus_t call_nn(sgemm_fn f, hipblasHandle_t h,
                               int M, int N, int K,
                               float* dA, float* dB, float* dC) {
    float al = 1.0f, be = 0.0f;
    return f(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
}

// CPU 参考 (ikj 序: A 行/B 行连续读, L1 友好)
static void cpu_gemm(float* ref, const float* A, const float* B, int M, int N, int K) {
    memset(ref, 0, (size_t)M * N * 4);
    for (int i = 0; i < M; i++) {
        const float* arow = A + (size_t)i * K;
        float* crow = ref + (size_t)i * N;
        for (int kk = 0; kk < K; kk++) {
            float a = arow[kk];
            const float* brow = B + (size_t)kk * N;
            for (int j = 0; j < N; j++) crow[j] += a * brow[j];
        }
    }
}

static int spot_ok(const float* C, const float* A, const float* B,
                   int M, int N, int K, unsigned seed) {
    srand(seed);
    for (int s = 0; s < 512; s++) {
        int i = rand() % M, j = rand() % N;
        double acc = 0.0;
        for (int kk = 0; kk < K; kk++) acc += (double)A[(size_t)i * K + kk] * B[(size_t)kk * N + j];
        float c = C[(size_t)i * N + j];
        if (fabsf(c - (float)acc) > 1e-2f * (1.0f + fabsf(c))) return 0;
    }
    return 1;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!so) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 2; }
    sgemm_fn vk = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    destroy_fn vk_destroy = (destroy_fn)dlsym(so, "hipblasDestroy");
    if (!vk || !cr || !vk_destroy) { fprintf(stderr, "dlsym failed\n"); return 2; }

    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);

    // shape (M,N,K) — LLM 推理常见 + 已知胜负区域
    struct { int m, n, k; const char* tag; } sh[] = {
        {4096, 4096, 4096, "square"},
        {8192, 8192, 8192, "square-big"},
        {2048, 2048, 4096, "square-med"},
        {1,    4096, 4096, "decode-1"},
        {1,   11008, 4096, "decode-ffn"},
        {16,   4096, 4096, "bs16"},
        {128,  4096, 4096, "bs128"},
        {256,  4096, 4096, "bs256"},
        {256, 11008, 4096, "bs256-ffn"},
        {512,  4096, 4096, "bs512"},
        {512,  8192, 4096, "M<<N"},
        {512,  512,  4096, "small"},
        {4096, 512,  4096, "M>>N"},
        {4096,1024,  4096, "M>>N1k"},
        {8192, 512,  4096, "M>>N known-lose"},
        {8192,1024,  4096, "M>>N big"},
        {4096, 256,  4096, "M>>N narrow"},
        {1024,2048,  4096, "wide"},
        {2048,1024,  4096, "tall"},
        {1024,1024,  4096, "med"},
    };
    int nsh = sizeof(sh) / sizeof(sh[0]);

    printf("%-6s %-18s %8s %8s %8s %8s %6s %8s\n",
           "M", "tag", "v6_ms", "v7_ms", "v6_TF", "v7_TF", "win", "speed");
    printf("-------------------------------------------------------------------\n");

    int total_bad = 0;
    for (int s = 0; s < nsh; s++) {
        int M = sh[s].m, N = sh[s].n, K = sh[s].k;
        size_t sa = (size_t)M * K, sb = (size_t)K * N, sc = (size_t)M * N;
        float* hA = malloc(sa * 4); float* hB = malloc(sb * 4); float* hC = malloc(sc * 4);
        srand(1000 + s);
        for (size_t i = 0; i < sa; i++) hA[i] = (rand() % 1000 - 500) / 500.0f;
        for (size_t i = 0; i < sb; i++) hB[i] = (rand() % 1000 - 500) / 500.0f;
        float* dA, *dB, *dC;
        hipMalloc((void**)&dA, sa * 4); hipMalloc((void**)&dB, sb * 4); hipMalloc((void**)&dC, sc * 4);
        hipMemcpy(dA, hA, sa * 4, hipMemcpyHostToDevice);
        hipMemcpy(dB, hB, sb * 4, hipMemcpyHostToDevice);

        // ---- v6 (env unset) ----
        unsetenv("VKBLAS_TILE128");
        call_nn(vk, h, M, N, K, dA, dB, dC); hipDeviceSynchronize();
        hipMemcpy(hC, dC, sc * 4, hipMemcpyDeviceToHost);
        float* hC6 = malloc(sc * 4); memcpy(hC6, hC, sc * 4);

        // ---- v7 (env set) ----
        setenv("VKBLAS_TILE128", "1", 1);
        call_nn(vk, h, M, N, K, dA, dB, dC); hipDeviceSynchronize();
        hipMemcpy(hC, dC, sc * 4, hipMemcpyDeviceToHost);
        unsetenv("VKBLAS_TILE128");

        // ---- 校验 ----
        int bad = 0;
        if ((long long)M * N * K <= 1024LL * 1024 * 1024) {
            float* ref = malloc(sc * 4);
            cpu_gemm(ref, hA, hB, M, N, K);
            for (size_t i = 0; i < sc; i++) {
                if (fabsf(hC6[i] - ref[i]) > 1e-2f * (1 + fabsf(ref[i]))) bad += 1;  // v6
                if (fabsf(hC[i] - ref[i]) > 1e-2f * (1 + fabsf(ref[i]))) bad += 10; // v7
            }
            free(ref);
        } else {
            // v6 vs v7 等价性 (v6 已被 test_gemm 全量验证) + v7 输出 CPU spot-check
            for (size_t i = 0; i < sc; i++) {
                float a = hC6[i], b = hC[i];
                if (fabsf(a - b) > 1e-2f * (1 + fabsf(a))) bad++;
            }
            if (!spot_ok(hC, hA, hB, M, N, K, 42 + s)) bad += 1000; // spot 挂 → 大标记
        }

        // ---- 计时 (重新跑, 数据已在 device) ----
        double best6 = 1e9, best7 = 1e9;
        unsetenv("VKBLAS_TILE128");
        for (int it = 0; it < 3; it++) {
            double t0 = now_s();
            call_nn(vk, h, M, N, K, dA, dB, dC); hipDeviceSynchronize();
            double dt = now_s() - t0; if (dt < best6) best6 = dt;
        }
        setenv("VKBLAS_TILE128", "1", 1);
        for (int it = 0; it < 3; it++) {
            double t0 = now_s();
            call_nn(vk, h, M, N, K, dA, dB, dC); hipDeviceSynchronize();
            double dt = now_s() - t0; if (dt < best7) best7 = dt;
        }
        unsetenv("VKBLAS_TILE128");

        double flops = 2.0 * M * N * K;
        const char* win = (best6 <= best7) ? "v6" : "v7";
        double sp = best6 > 0 ? best6 / best7 : 0;
        printf("%-6d %-18s %8.3f %8.3f %8.2f %8.2f %5s %7.2fx %s\n",
               M, sh[s].tag, best6 * 1000, best7 * 1000,
               flops / best6 / 1e12, flops / best7 / 1e12, win, sp,
               bad ? "<<< BAD" : "");
        total_bad += bad ? 1 : 0;
        fflush(stdout);

        free(hA); free(hB); free(hC); free(hC6);
        hipFree(dA); hipFree(dB); hipFree(dC);
    }
    vk_destroy(h);
    printf("-------------------------------------------------------------------\n");
    printf("%s (speed>1.05 才值得切 v7; BAD = 正确性失败)\n",
           total_bad == 0 ? "ALL CORRECT" : "CORRECTNESS FAILURES");
    return 0;
}
