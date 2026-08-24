// mini2.c — bf16/f16 GemmEx 计时
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*gemmex_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int, int, int, const void*, const void*, hipblasDatatype_t, int,
    const void*, hipblasDatatype_t, int, const void*, void*, hipblasDatatype_t, int,
    hipblasDatatype_t, hipblasGemmAlgo_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    const char* dtype = "bf16";
    if (argc > 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); dtype = argv[4]; }
    int elem = !strcmp(dtype, "f32") ? 4 : 2;
    int celem = elem;
    if (argc > 5 && !strcmp(argv[5], "c2b")) celem = 2;  // C 输出 2B (直通条件)
    hipblasDatatype_t dt = !strcmp(dtype, "f32") ? HIPBLAS_R_32F
                        : !strcmp(dtype, "bf16") ? HIPBLAS_R_16B : HIPBLAS_R_16F;
    hipblasDatatype_t dct = celem == 2 ? dt : HIPBLAS_R_32F;
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    gemmex_fn gx = (gemmex_fn)dlsym(so, "hipblasGemmEx");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);
    void *dA, *dB, *dC;
    hipMalloc(&dA, (size_t)M * K * elem); hipMalloc(&dB, (size_t)K * N * elem); hipMalloc(&dC, (size_t)M * N * celem);
    hipMemset(dA, 0x3f, (size_t)M * K * elem); hipMemset(dB, 0x3f, (size_t)K * N * elem);
    float al = 1.0f, be = 0.0f;
    hipblasComputeType_t ct = HIPBLAS_COMPUTE_32F;
    for (int it = 0; it < 5; it++) {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        double t0 = ts.tv_sec + ts.tv_nsec / 1e9;
        gx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &al, dB, dt, N, dA, dt, K, &be, dC, dct, N, ct, HIPBLAS_GEMM_DEFAULT);
        hipDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double dtm = ts.tv_sec + ts.tv_nsec / 1e9 - t0;
        printf("%s M=%d N=%d K=%d: %.3f ms (%.2f TFLOPS)\n", dtype, M, N, K, dtm * 1e3, 2.0 * M * N * K / dtm / 1e12);
    }
    return 0;
}