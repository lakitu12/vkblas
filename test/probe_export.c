// probe_export.c — 探测 hsa_amd_portable_export_dmabuf 对 hipMalloc 子分配的失败模式
// 假设: hip 池 (rpp/VMM) 从已释放块切出的子块导出返回 4099 (INVALID_ARGUMENT)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

typedef uint32_t hsa_status_t;
typedef hsa_status_t (*hsa_init_fn)(void);
typedef hsa_status_t (*hsa_shut_fn)(void);
typedef hsa_status_t (*hsa_export_fn)(const void* ptr, size_t size, int* dmabuf, uint64_t* offset);

static hsa_init_fn hsa_init;
static hsa_shut_fn hsa_shut;
static hsa_export_fn hsa_export;

static void report(const char* tag, const void* ptr, size_t size) {
    int fd = -1; uint64_t off = 0;
    hsa_status_t st = hsa_export(ptr, size, &fd, &off);
    if (st == 0) {
        hipPointerAttribute_t attr;
        hipPointerGetAttributes(&attr, (hipDeviceptr_t)ptr);
        size_t sub = (const char*)ptr - (const char*)attr.devicePointer;
        printf("%-28s ptr=%-18p size=%8zu  OK   off=%-8lu  sub-off-within-alloc=%zu  fd=%d\n",
               tag, ptr, size, (unsigned long)off, sub, fd);
        close(fd);
    } else {
        hipPointerAttribute_t attr;
        hipPointerGetAttributes(&attr, (hipDeviceptr_t)ptr);
        size_t sub = (const char*)ptr - (const char*)attr.devicePointer;
        printf("%-28s ptr=%-18p size=%8zu  FAIL=%u  (alloc-base=%p, sub-off=%zu)\n",
               tag, ptr, size, st, attr.devicePointer, sub);
    }
    fflush(stdout);
}

int main(void) {
    void* h = dlopen("/opt/rocm-6.4.3/lib/libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen hsa failed\n"); return 2; }
    hsa_init = (hsa_init_fn)dlsym(h, "hsa_init");
    hsa_shut = (hsa_shut_fn)dlsym(h, "hsa_shut_down");
    hsa_export = (hsa_export_fn)dlsym(h, "hsa_amd_portable_export_dmabuf");
    if (!hsa_init || !hsa_export) { fprintf(stderr, "symbols missing\n"); return 2; }
    if (hsa_init() != 0) { fprintf(stderr, "hsa_init failed\n"); return 2; }

    hipInit(0); hipSetDevice(0);

    printf("=== 阶段 1: 全新大块 (无池历史) ===\n");
    void* p1; hipMalloc(&p1, 64u << 20);
    report("fresh 64MB whole", p1, 64u << 20);
    report("fresh 64MB sub 16MB@0", p1, 16u << 20);
    report("fresh 64MB sub 16MB@16M", (char*)p1 + (16u << 20), 16u << 20);
    hipFree(p1);

    printf("\n=== 阶段 2: free 后重新分配 (可能命中池子分配) ===\n");
    void* a, *b, *c, *d;
    hipMalloc(&a, 16u << 20); hipMalloc(&b, 16u << 20);
    hipMalloc(&c, 16u << 20); hipMalloc(&d, 16u << 20);
    report("realloc #1 16MB", a, 16u << 20);
    report("realloc #2 16MB", b, 16u << 20);
    report("realloc #3 16MB", c, 16u << 20);
    report("realloc #4 16MB", d, 16u << 20);
    hipFree(a); hipFree(b); hipFree(c); hipFree(d);

    printf("\n=== 阶段 3: 交错分配/释放 (池碎片化) ===\n");
    void* x[8];
    for (int i = 0; i < 8; i++) hipMalloc(&x[i], 8u << 20);
    hipFree(x[0]); hipFree(x[2]); hipFree(x[4]); hipFree(x[6]);
    report("x[1] 8MB (奇数幸存)", x[1], 8u << 20);
    report("x[3] 8MB (奇数幸存)", x[3], 8u << 20);
    for (int i = 0; i < 8; i++) hipFree(x[i]);

    printf("\n=== 阶段 4: 精确复现 bench 序列 ===\n");
    // bench: 4096³ (64+64+64MB) → free → 8192³ (512×3) → free → 2048²×4096 (32+32+16MB)
    {
        void *a1, *b1, *c1;
        hipMalloc(&a1, 64u << 20); hipMalloc(&b1, 64u << 20); hipMalloc(&c1, 64u << 20);
        report("big1 A 64MB", a1, 64u << 20);
        report("big1 B 64MB", b1, 64u << 20);
        report("big1 C 64MB", c1, 64u << 20);
        hipFree(a1); hipFree(b1); hipFree(c1);
    }
    {
        void *a1, *b1, *c1;
        hipMalloc(&a1, 512u << 20); hipMalloc(&b1, 512u << 20); hipMalloc(&c1, 512u << 20);
        report("big2 A 512MB", a1, 512u << 20);
        hipFree(a1); hipFree(b1); hipFree(c1);
    }
    {
        void *a2, *b2, *c2;
        hipMalloc(&a2, 32u << 20); hipMalloc(&b2, 32u << 20); hipMalloc(&c2, 16u << 20);
        report("small A 32MB (bench 失败点)", a2, 32u << 20);
        report("small B 32MB", b2, 32u << 20);
        report("small C 16MB", c2, 16u << 20);
        hipFree(a2); hipFree(b2); hipFree(c2);
    }

    hsa_shut();
    return 0;
}
