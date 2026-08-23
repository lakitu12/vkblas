// bench_decode.c - PyTorch style decode (hipblas T,N) timing benchmark
// Usage: HSA_OVERRIDE_GFX_VERSION=8.0.3 LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib
//        [VKBLAS_CACHE_TRANSPOSE=1] ./test/bench_decode [m [dtype]]
//   m default 1, one of 1/2/4/8/16; dtype: f32|bf16|f16 (default bf16)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*gemmex_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int, int, int, const void*, const void*, hipblasDatatype_t, int,
    const void*, hipblasDatatype_t, int, const void*, void*, hipblasDatatype_t, int,
    hipblasDatatype_t, hipblasGemmAlgo_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    int m = argc > 1 ? atoi(argv[1]) : 1;
    const char* dtype = argc > 2 ? argv[2] : "bf16";
    int O = 4096, K = 4096;
    void* so = dlopen("/home/lakitu/code/vkblas/libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!so) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    gemmex_fn gx = (gemmex_fn)dlsym(so, "hipblasGemmEx");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);

    int elem = strcmp(dtype, "f32") == 0 ? 4 : 2;
    hipblasDatatype_t dt = strcmp(dtype, "f32") == 0 ? HIPBLAS_R_32F
                          : strcmp(dtype, "bf16") == 0 ? HIPBLAS_R_16B : HIPBLAS_R_16F;

    void *dX, *dW, *dC;
    hipMalloc(&dX, (size_t)m * K * elem);
    hipMalloc(&dW, (size_t)O * K * elem);
    hipMalloc(&dC, (size_t)m * O * elem);
    hipMemset(dX, 0x3f, (size_t)m * K * elem);  // 0.5-ish
    hipMemset(dW, 0x3f, (size_t)O * K * elem);
    hipMemset(dC, 0, (size_t)m * O * elem);

    float al = 1.0f, be = 0.0f;
    hipblasComputeType_t ct = elem == 4 ? HIPBLAS_COMPUTE_32F : HIPBLAS_COMPUTE_32F;

    // warmup (cache build)
    gx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, O, m, K, &al, dW, dt, K, dX, dt, K, &be, dC, dt, O,
       ct, HIPBLAS_GEMM_DEFAULT);
    hipDeviceSynchronize();
    gx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, O, m, K, &al, dW, dt, K, dX, dt, K, &be, dC, dt, O,
       ct, HIPBLAS_GEMM_DEFAULT);
    hipDeviceSynchronize();

    // 20 iterations timed
    double best = 1e9;
    for (int it = 0; it < 20; it++) {
        double t0 = now_s();
        gx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, O, m, K, &al, dW, dt, K, dX, dt, K, &be, dC, dt, O,
           ct, HIPBLAS_GEMM_DEFAULT);
        hipDeviceSynchronize();
        double dt_ = now_s() - t0;
        if (dt_ < best) best = dt_;
    }
    printf("%s m=%-2d O=4096 K=4096: %.3f ms\n", dtype, m, best * 1e3);
    return 0;
}