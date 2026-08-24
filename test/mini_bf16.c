// mini_bf16.c — bf16 GEMM 计时 (hipblasGemmEx R_16B, 经 vkblas 直通库)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*gemmex_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                     int, int, int, const void*, const void*, hipblasDatatype_t, int,
                                     const void*, hipblasDatatype_t, int, const void*, void*,
                                     hipblasDatatype_t, int, hipblasDatatype_t, hipblasGemmAlgo_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    gemmex_fn vk = (gemmex_fn)dlsym(so, "hipblasGemmEx");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);
    void *dA, *dB, *dC;
    hipMalloc(&dA, (size_t)M * K * 2); hipMalloc(&dB, (size_t)K * N * 2); hipMalloc(&dC, (size_t)M * N * 2);
    hipMemset(dA, 0x3f, (size_t)M * K * 2); hipMemset(dB, 0x3f, (size_t)K * N * 2);
    float al = 1.0f, be = 0.0f;
    for (int it = 0; it < 5; it++) {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        double t0 = ts.tv_sec + ts.tv_nsec / 1e9;
        vk(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, HIPBLAS_R_16B, N, dA, HIPBLAS_R_16B, K,
           &be, dC, HIPBLAS_R_16B, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        hipDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double dt = ts.tv_sec + ts.tv_nsec / 1e9 - t0;
        printf("bf16 M=%d N=%d K=%d: %.3f ms (%.2f TFLOPS)\n", M, N, K, dt * 1e3, 2.0 * M * N * K / dt / 1e12);
    }
    return 0;
}