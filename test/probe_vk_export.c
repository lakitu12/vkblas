// probe_vk_export.c — 判别: vkblas 大尺寸 dma-buf 导入后是否污染后续导出
// PHASES=64,32  → 4096³ (64MB×3) 后 32MB GEMM
// PHASES=512,32 → 8192³ (512MB×3) 后 32MB GEMM
// PHASES=64,512,32 → 完整复现 bench 序列
// 每个 phase: hipMalloc → 1 次 GEMM → 打印返回/4099/前后 fd 数 → hipFree
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*sgemm_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                    int64_t, int64_t, int64_t, const float*, const float*, int64_t,
                                    const float*, int64_t, const float*, float*, int64_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);
typedef hipblasStatus_t (*destroy_fn)(hipblasHandle_t);

static int fd_count(void) {
    // 数 /proc/self/fd 条目
    int n = 0;
    char path[64];
    for (int i = 0; i < 4096; i++) {
        snprintf(path, sizeof(path), "/proc/self/fd/%d", i);
        if (access(path, F_OK) == 0) n++;
    }
    return n;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* phases = getenv("PHASES") ? getenv("PHASES") : "64,512,32";
    printf("PHASES=%s  fd_start=%d\n", phases, fd_count());

    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    if (!so) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 2; }
    sgemm_fn vk = (sgemm_fn)dlsym(so, "hipblasSgemm");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    destroy_fn vk_destroy = (destroy_fn)dlsym(so, "hipblasDestroy");
    if (!vk || !cr || !vk_destroy) { fprintf(stderr, "dlsym failed\n"); return 2; }

    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);
    float al = 1.0f, be = 0.0f;

    char buf[256]; strncpy(buf, phases, sizeof(buf));
    for (char* tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int M, N, K;
        if (!strcmp(tok, "64")) { M = N = K = 4096; }
        else if (!strcmp(tok, "512")) { M = N = K = 8192; }
        else if (!strcmp(tok, "32")) { M = 2048; N = 2048; K = 4096; }
        else if (!strcmp(tok, "16")) { M = 4096; N = 1024; K = 4096; }
        else { fprintf(stderr, "unknown phase %s\n", tok); return 2; }

        size_t sa = (size_t)M * K * 4, sb = (size_t)K * N * 4, sc = (size_t)M * N * 4;
        float *dA, *dB, *dC;
        hipMalloc((void**)&dA, sa); hipMalloc((void**)&dB, sb); hipMalloc((void**)&dC, sc);
        hipMemset(dA, 0, sa); hipMemset(dB, 0, sb); hipMemset(dC, 0, sc);
        int fdb = fd_count();
        // hipBLAS 列主序技巧: TT 交换
        hipblasStatus_t st = vk(h, HIPBLAS_OP_T, HIPBLAS_OP_T, N, M, K,
                                &al, dB, N, dA, K, &be, dC, N);
        hipDeviceSynchronize();
        int fda = fd_count();
        printf("[%s] %dx%dx%d  A=%zuMB B=%zuMB C=%zuMB  st=%d  fd %d→%d\n",
               tok, M, N, K, sa >> 20, sb >> 20, sc >> 20, (int)st, fdb, fda);
        hipFree(dA); hipFree(dB); hipFree(dC);
    }
    printf("end fd=%d\n", fd_count());
    vk_destroy(h);
    return 0;
}
