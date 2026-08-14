// probe_loop.c — 连续 hipMalloc/hipFree + 导出, 复现 4099
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

typedef uint32_t hsa_status_t;
typedef hsa_status_t (*hsa_init_fn)(void);
typedef hsa_status_t (*hsa_export_fn)(const void*, size_t, int*, uint64_t*);

int main(void) {
    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    hsa_init_fn hsa_init = (hsa_init_fn)dlsym(hsa, "hsa_init");
    hsa_export_fn hsa_export = (hsa_export_fn)dlsym(hsa, "hsa_amd_portable_export_dmabuf");
    hsa_init();

    hipInit(0);
    hipSetDevice(0);

    int fails = 0;
    for (int i = 0; i < 60; i++) {
        // 模拟 run_case: 3 个分配 + 导出
        size_t szA = 257 * 64, szB = 64 * 129, szC = 257 * 129;
        void *dA, *dB, *dC;
        hipError_t e1 = hipMalloc(&dA, szA * 4);
        hipError_t e2 = hipMalloc(&dB, szB * 4);
        hipError_t e3 = hipMalloc(&dC, szC * 4);
        int fd = -1; uint64_t off = 0;
        hsa_status_t st = hsa_export(dB, ((size_t)63 * 64 + 129) * 4, &fd, &off);
        if (st != 0) {
            printf("iter %d: export FAILED st=%u (hipMalloc %d/%d/%d)\n", i, st, e1, e2, e3);
            fails++;
        } else {
            if (fd >= 0) close(fd);
        }
        hipFree(dA); hipFree(dB); hipFree(dC);
        if (fails >= 3) break;
    }
    printf("done, %d failures\n", fails);
    return 0;
}
