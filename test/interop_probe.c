// interop_probe.c — 最小互操作探针: hipMalloc → hsa_amd_portable_export_dmabuf
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

typedef uint32_t hsa_status_t;
typedef hsa_status_t (*hsa_init_fn)(void);
typedef hsa_status_t (*hsa_export_fn)(const void*, size_t, int*, uint64_t*);

int main(void) {
    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!hsa) { printf("dlopen hsa failed\n"); return 1; }
    hsa_init_fn hsa_init = (hsa_init_fn)dlsym(hsa, "hsa_init");
    hsa_export_fn hsa_export = (hsa_export_fn)dlsym(hsa, "hsa_amd_portable_export_dmabuf");

    hipInit(0);
    hipSetDevice(0);

    void* d;
    hipError_t e = hipMalloc(&d, 65536);
    printf("hipMalloc: %s (%p)\n", hipGetErrorString(e), d);

    // 方案1: HIP 已初始化 HSA 后直接导出
    int fd = -1; uint64_t off = 0;
    hsa_status_t st = hsa_export(d, 65536, &fd, &off);
    printf("export (after hipInit): st=%u fd=%d off=%llu\n", st, fd, (unsigned long long)off);
    if (fd >= 0) close(fd);

    // 方案2: 先显式 hsa_init 再导出
    hsa_status_t i1 = hsa_init();
    printf("hsa_init: %u\n", i1);
    fd = -1; off = 0;
    st = hsa_export(d, 65536, &fd, &off);
    printf("export (after hsa_init): st=%u fd=%d off=%llu\n", st, fd, (unsigned long long)off);
    if (fd >= 0) close(fd);

    // 方案3: 小 size (4 字节)
    fd = -1; off = 0;
    st = hsa_export(d, 4, &fd, &off);
    printf("export (4B): st=%u fd=%d off=%llu\n", st, fd, (unsigned long long)off);
    if (fd >= 0) close(fd);

    // 方案4: 分配内偏移 (中间 4KB 处)
    fd = -1; off = 0;
    st = hsa_export((char*)d + 4096, 4096, &fd, &off);
    printf("export (+4096, 4096B): st=%u fd=%d off=%llu\n", st, fd, (unsigned long long)off);
    if (fd >= 0) close(fd);

    hipFree(d);
    return 0;
}
