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
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec / 1e9; }

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    int reps = 10;
    if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc > 4) reps = atoi(argv[4]);
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    sgemm_fn vk = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);
    float *dA, *dB, *dC;
    hipMalloc((void**)&dA, (size_t)M * K * 4); hipMalloc((void**)&dB, (size_t)K * N * 4); hipMalloc((void**)&dC, (size_t)M * N * 4);
    hipMemset(dA, 0x3f, (size_t)M * K * 4); hipMemset(dB, 0x3f, (size_t)K * N * 4);
    float al = 1.0f, be = 0.0f;
    // warmup
    for (int it = 0; it < 2; it++) vk(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
    hipDeviceSynchronize();
    // 单次 (含 sync)
    double t0 = now_s();
    vk(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
    hipDeviceSynchronize();
    double t1 = now_s();
    printf("single+sync: %.3f ms\n", (t1 - t0) * 1e3);
    // N 连发 (1 次 sync)
    t0 = now_s();
    for (int it = 0; it < reps; it++) vk(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, N, dA, K, &be, dC, N);
    hipDeviceSynchronize();
    t1 = now_s();
    printf("%d 连发+1sync: %.3f ms (每发 %.3f ms)\n", reps, (t1 - t0) * 1e3, (t1 - t0) * 1e3 / reps);
    return 0;
}