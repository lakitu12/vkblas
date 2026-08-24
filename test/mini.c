// mini.c — 单 shape GEMM 计时 + shader dump 用
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*sgemm_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                    int64_t, int64_t, int64_t, const float*, const float*, int64_t,
                                    const float*, int64_t, const float*, float*, int64_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    sgemm_fn vk = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);
    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, (size_t)M * K * 4); hipMalloc((void**)&dB, (size_t)K * N * 4); hipMalloc((void**)&dC, (size_t)M * N * 4);
    hipMemset(dA, 0x3f, (size_t)M * K * 4); hipMemset(dB, 0x3f, (size_t)K * N * 4);
    float al = 1.0f, be = 0.0f;
    for (int it = 0; it < 5; it++) {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        double t0 = ts.tv_sec + ts.tv_nsec / 1e9;
        vk(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
        hipDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double dt = ts.tv_sec + ts.tv_nsec / 1e9 - t0;
        printf("M=%d N=%d K=%d: %.3f ms (%.2f TFLOPS)\n", M, N, K, dt * 1e3, 2.0 * M * N * K / dt / 1e12);
    }
    return 0;
}