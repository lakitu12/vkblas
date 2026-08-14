// vkblas.c — Vulkan compute 引擎 + HSA dma-buf 零拷贝互操作
// 懒初始化单例; 线程安全(全局锁, GEMM 串行执行)
#include "vkblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <vulkan/vulkan.h>

#ifndef VKBLAS_SHADER_DIR
#define VKBLAS_SHADER_DIR "/home/lakitu/code/vkblas/src/shaders"
#endif

// ---- HSA 动态加载 ----
typedef uint32_t hsa_status_t;
typedef hsa_status_t (*hsa_init_fn)(void);
typedef hsa_status_t (*hsa_shut_fn)(void);
typedef hsa_status_t (*hsa_export_dmabuf_fn)(const void* ptr, size_t size,
                                             int* dmabuf, uint64_t* offset);

// ---- 引擎状态 ----
// 注意: 不做 dma-buf 导入缓存 — hipFree 后地址会被 HIP 复用, 缓存会读到陈旧的
// dma-buf 引用; 每次调用重新 export/import, 用完即释放 (正确性优先)
static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkDescriptorPool dpool;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipe[4];   // idx = op_a*2 + op_b: nn,tn,nt,tt
    VkDescriptorSet dset;
    // 转置 (TB=0 快路径支持)
    VkDescriptorSetLayout tdsl;
    VkPipelineLayout tpl;
    VkPipeline tpipe;
    VkDescriptorSet tdset;
    VkDeviceMemory tmem;
    VkBuffer tbuf;
    size_t tbuf_size;

    void* hsa_so;
    hsa_init_fn hsa_init;
    hsa_shut_fn hsa_shut;
    hsa_export_dmabuf_fn hsa_export;

    pthread_mutex_t lock;
    int init_done;
} g;

static void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[vkblas] FAIL %s: %d\n", what, r);
        abort();
    }
}

static void* load_hsa(void) {
    void* h = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("/opt/rocm/lib/libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("/opt/rocm-6.4.3/lib/libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "[vkblas] libhsa-runtime64 not found: %s\n", dlerror()); return NULL; }
    g.hsa_init = (hsa_init_fn)dlsym(h, "hsa_init");
    g.hsa_export = (hsa_export_dmabuf_fn)dlsym(h, "hsa_amd_portable_export_dmabuf");
    if (!g.hsa_init || !g.hsa_export) {
        fprintf(stderr, "[vkblas] hsa symbols missing\n");
        dlclose(h);
        return NULL;
    }
    g.hsa_shut = (hsa_shut_fn)dlsym(h, "hsa_shut_down");
    return h;
}

static int load_spv(const char* name, VkShaderModule* sm) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", VKBLAS_SHADER_DIR, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[vkblas] shader %s not found\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(sz);
    if (fread(code, 1, sz, f) != (size_t)sz) { fclose(f); free(code); return -1; }
    fclose(f);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)sz, .pCode = code };
    VkResult r = vkCreateShaderModule(g.dev, &smci, NULL, sm);
    free(code);
    return r == VK_SUCCESS ? 0 : -1;
}

static void init_vkblas(void) {
    memset(&g, 0, sizeof(g));
    pthread_mutex_init(&g.lock, NULL);

    g.hsa_so = load_hsa();
    if (!g.hsa_so) { g.init_done = 1; return; }
    g.hsa_init();  // 幂等; 失败也无妨(导出时会报错)

    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2 };
    const char* inst_ext[] = { VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai, .enabledExtensionCount = 1, .ppEnabledExtensionNames = inst_ext };
    if (vkCreateInstance(&ici, NULL, &g.inst) != VK_SUCCESS) {
        fprintf(stderr, "[vkblas] create instance failed\n");
        g.init_done = 1; return;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.inst, &n, NULL);
    VkPhysicalDevice* devs = malloc(n * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g.inst, &n, devs);
    g.phys = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (strstr(props.deviceName, "RADV") || strstr(props.deviceName, "AMD")) {
            g.phys = devs[i];
            break;
        }
    }
    free(devs);
    if (g.phys == VK_NULL_HANDLE) { fprintf(stderr, "[vkblas] no AMD device\n"); g.init_done = 1; return; }

    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &n, NULL);
    VkQueueFamilyProperties* qps = malloc(n * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &n, qps);
    g.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++)
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qfam = i; break; }
    free(qps);

    const char* dev_ext[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g.qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &dqci,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = dev_ext };
    if (vkCreateDevice(g.phys, &dci, NULL, &g.dev) != VK_SUCCESS) {
        fprintf(stderr, "[vkblas] create device failed\n");
        g.init_done = 1; return;
    }
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g.qfam };
    vk_check(vkCreateCommandPool(g.dev, &cpci, NULL, &g.cpool), "cmd pool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    vk_check(vkAllocateCommandBuffers(g.dev, &cbai, &g.cmd), "cmd buffer");

    VkDescriptorSetLayoutBinding bnds[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnds };
    vk_check(vkCreateDescriptorSetLayout(g.dev, &dslci, NULL, &g.dsl), "ds layout");

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 44 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    vk_check(vkCreatePipelineLayout(g.dev, &plci, NULL, &g.pl), "pipeline layout");

    static const char* names[4] = { "gemm_nn.spv", "gemm_nt.spv", "gemm_tn.spv", "gemm_tt.spv" };
    VkShaderModule sm;
    for (int i = 0; i < 4; i++) {
        if (load_spv(names[i], &sm) != 0) { g.init_done = 1; return; }
        VkComputePipelineCreateInfo cpci2 = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.pl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpci2, NULL, &g.pipe[i]), "pipeline");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };  // GEMM 3 + 转置 2
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &ps };
    vk_check(vkCreateDescriptorPool(g.dev, &dpci, NULL, &g.dpool), "desc pool");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.dsl };
    vk_check(vkAllocateDescriptorSets(g.dev, &dsai, &g.dset), "desc set");

    // ---- 转置 pipeline (2 个 storage buffer) ----
    VkDescriptorSetLayoutBinding tbnds[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo tdslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = tbnds };
    vk_check(vkCreateDescriptorSetLayout(g.dev, &tdslci, NULL, &g.tdsl), "t dsl");
    VkPushConstantRange tpcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 20 };
    VkPipelineLayoutCreateInfo tplci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.tdsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &tpcr };
    vk_check(vkCreatePipelineLayout(g.dev, &tplci, NULL, &g.tpl), "t pl");
    if (load_spv("transpose.spv", &sm) == 0) {
        VkComputePipelineCreateInfo tcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.tpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &tcp, NULL, &g.tpipe), "t pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }
    VkDescriptorSetAllocateInfo tdsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.tdsl };
    vk_check(vkAllocateDescriptorSets(g.dev, &tdsai, &g.tdset), "t dset");

    g.ready = 1;
    g.init_done = 1;
    fprintf(stderr, "[vkblas] engine ready (Vulkan + HSA dma-buf interop)\n");
}

static void ensure_init(void) {
    if (!g.init_done) init_vkblas();
}

// 导出 HIP 指针为 dma-buf 并导入 Vulkan; 用完即释放 (不缓存)
static int import_ptr(const void* ptr, size_t size, VkBuffer* buf,
                      VkDeviceMemory* mem, VkDeviceSize* offset) {
    if (!g.ready) return -1;

    int fd = -1;
    uint64_t off = 0;
    hsa_status_t st = g.hsa_export(ptr, size, &fd, &off);
    if (st != 0) {
        fprintf(stderr, "[vkblas] hsa_amd_portable_export_dmabuf failed: %u\n", st);
        return -1;
    }

    VkBuffer b;
    VkExternalMemoryBufferCreateInfo embci = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &embci, .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(g.dev, &bci, NULL, &b) != VK_SUCCESS) { close(fd); return -1; }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.dev, b, &mr);

    // 找支持 dma-buf 导入的 memory type (逐 type 尝试, 不依赖旧版 memoryTypeBits 查询)
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &mprops);
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    VkImportMemoryFdInfoKHR imfi = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd };
    for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
        if (!(mr.memoryTypeBits & (1u << i))) continue;
        // 优先 device-local; 第二遍再试其余
        if (!(mprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = i;
        mai.pNext = &imfi;
        if (vkAllocateMemory(g.dev, &mai, NULL, &m) == VK_SUCCESS) break;
    }
    if (m == VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
            if (!(mr.memoryTypeBits & (1u << i))) continue;
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = i;
            mai.pNext = &imfi;
            if (vkAllocateMemory(g.dev, &mai, NULL, &m) == VK_SUCCESS) break;
        }
    }
    if (m == VK_NULL_HANDLE) {
        vkDestroyBuffer(g.dev, b, NULL); close(fd); return -1;
    }
    // fd 已由 Vulkan 接管 (导入成功), 可关闭
    close(fd);

    if (vkBindBufferMemory(g.dev, b, m, (VkDeviceSize)off) != VK_SUCCESS) {
        vkFreeMemory(g.dev, m, NULL); vkDestroyBuffer(g.dev, b, NULL); return -1;
    }
    // 诊断: 打印实际使用的 memory type 属性
    VkPhysicalDeviceMemoryProperties dmp;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &dmp);
    for (uint32_t i = 0; i < dmp.memoryTypeCount; i++) {
        if (mai.memoryTypeIndex == i) {
            fprintf(stderr, "[vkblas] dma-buf import -> memoryType %u flags=0x%x (%s%s%s%s)\n",
                    i, dmp.memoryTypes[i].propertyFlags,
                    dmp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "DEVICE_LOCAL " : "",
                    dmp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "HOST_VISIBLE " : "",
                    dmp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? "COHERENT " : "",
                    dmp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "CACHED" : "");
        }
    }
    *buf = b; *mem = m; *offset = off;
    return 0;
}

static void release_ptr(VkDeviceMemory mem, VkBuffer buf) {
    vkDestroyBuffer(g.dev, buf, NULL);
    vkFreeMemory(g.dev, mem, NULL);
}

// 转置: out[N][K] = in[K][N]^T (row-major), in 行步长 ldin, out 紧密 (行步长 N)
// 用于 TB=0 场景: 把行优先 B 转成列优先, 走 gemm 的 TB=1 快路径 (gfx803 实测)
// 返回 out 的 Vulkan buffer 与行步长; 转置 buffer 为全局池, 调用方用完即复用
static int transpose_buf(const void* in, size_t bytes_in,
                         uint32_t K, uint32_t N, uint32_t ldin,
                         VkBuffer* out_vk, uint32_t* ldout) {
    size_t need = (size_t)N * K * 4;
    if (g.tbuf == VK_NULL_HANDLE || need > g.tbuf_size) {
        if (g.tbuf != VK_NULL_HANDLE) {
            vkFreeMemory(g.dev, g.tmem, NULL);
            vkDestroyBuffer(g.dev, g.tbuf, NULL);
        }
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        if (vkCreateBuffer(g.dev, &bci, NULL, &g.tbuf) != VK_SUCCESS) return -1;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g.dev, g.tbuf, &mr);
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.size };
        VkPhysicalDeviceMemoryProperties mprops;
        vkGetPhysicalDeviceMemoryProperties(g.phys, &mprops);
        for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
            if (mr.memoryTypeBits & (1u << i) &&
                (mprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                mai.memoryTypeIndex = i; break;
            }
        }
        if (vkAllocateMemory(g.dev, &mai, NULL, &g.tmem) != VK_SUCCESS) return -1;
        if (vkBindBufferMemory(g.dev, g.tbuf, g.tmem, 0) != VK_SUCCESS) return -1;
        g.tbuf_size = need;
    }

    VkBuffer bIn; VkDeviceMemory mIn; VkDeviceSize oIn;
    if (import_ptr(in, bytes_in, &bIn, &mIn, &oIn) != 0) return -1;

    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {g.tbuf, 0, need} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    // Out 是 (N×K) 紧密, 行步长 = K (不是 N! 非方阵时 N≠K 会写错位)
    // gfx803/RADV: 单次 dispatch 过大 (≥16384 wg) 性能崩溃 (~7.5ms → 理论 <1ms),
    // 按 512 行/块拆多个 dispatch (实测 8×8MB 分块快 4.7 倍)
    struct { uint32_t R, C, ldin, ldout, roff; } pc = { K, N, ldin, K, 0 };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
    // 按 8MB/块动态分块 (实测最优 ~2048 wg/8MB; 固定 512 行对小 N 块太小)
    const uint32_t ROWS_PER_DISP = (8u * 1024 * 1024) / (N * 4u);  // 8MB / 行字节数
    uint32_t rpd = ROWS_PER_DISP < 32 ? 32 : (ROWS_PER_DISP > 4096 ? 4096 : ROWS_PER_DISP);
    for (uint32_t ro = 0; ro < K; ro += rpd) {
        pc.roff = ro;
        uint32_t rows = K - ro < rpd ? K - ro : rpd;
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (rows + 31) / 32, (N + 31) / 32, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "t submit");
    vkQueueWaitIdle(g.queue);

    release_ptr(mIn, bIn);
    *out_vk = g.tbuf;
    *ldout = K;   // Out (N×K) 紧密, 行步长 K
    return 0;
}

static int run_gemm(int variant, const void* A, VkBuffer bB_opt, const void* B, void* C,
                    size_t ba, size_t bb, size_t bc,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t lda, uint32_t ldb, uint32_t ldc,
                    float alpha, float beta) {
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    int bB_imported = 0;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (bB_opt != VK_NULL_HANDLE) {
        bB = bB_opt; mB = VK_NULL_HANDLE; oB = 0;  // 转置产物, 直接复用
    } else {
        if (import_ptr(B, bb, &bB, &mB, &oB) != 0) {
            release_ptr(mA, bA);
            return -1;
        }
        bB_imported = 1;
    }
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA);
        if (bB_imported) release_ptr(mB, bB);
        return -1;
    }
    (void)oA; (void)oB; (void)oC;

    VkDescriptorBufferInfo db[3] = {
        {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc; float alpha, beta; } pc = {
        M, N, K, (M + 63) / 64, (N + 63) / 64, (K + 31) / 32, lda, ldb, ldc, alpha, beta };

    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, pc.Mt, pc.Nt, 1);
    vkEndCommandBuffer(g.cmd);

    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "submit");
    vkQueueWaitIdle(g.queue);

    release_ptr(mA, bA);
    if (bB_imported) release_ptr(mB, bB);
    release_ptr(mC, bC);
    return 0;
}

vkblas_status_t vkblas_gemm_f32(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    float alpha,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    float beta,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    if (M == 0 || N == 0 || K == 0 || batch == 0) return VKBLAS_ERR_PARAM;
    pthread_mutex_lock(&g.lock);
    ensure_init();
    if (!g.ready) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_INIT; }

    // buffer 大小按 shader 实际访问范围 (精确覆盖, 转置/直接读不同):
    //   TA=0 直接读 A[m*lda+k]:  max = (M-1)*lda + K-1
    //   TA=1 转置读 A[k*lda+m]:  max = (K-1)*lda + M-1
    //   TB=0 直接读 B[k*ldb+n]:  max = (K-1)*ldb + N-1
    //   TB=1 转置读 B[n*ldb+k]:  max = (N-1)*ldb + K-1
    //   C 写 C[row*ldc+col]:     max = (M-1)*ldc + N-1
    size_t ba = (op_a == VKBLAS_OP_T) ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K;
    size_t bb = (op_b == VKBLAS_OP_T) ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N;
    size_t bc = (size_t)(M - 1) * ldc + N;
    ba *= 4; bb *= 4; bc *= 4;
    int variant = (int)op_a * 2 + (int)op_b;
    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    for (uint32_t i = 0; i < batch; i++) {
        if (op_b == VKBLAS_OP_N && g.tpipe != VK_NULL_HANDLE) {
            // B 是行优先 (K×N, 行步长 ldb) → 转置成 (N×K) 紧密, 走 TB=1 快路径
            // (gfx803/RADV 上行优先 B 直接读慢 ~9x)
            VkBuffer bBt;
            uint32_t ldb_t;
            if (transpose_buf(pb, bb, K, N, ldb, &bBt, &ldb_t) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
            size_t bb_t = (size_t)(N - 1) * ldb_t + K;  // TB=1 转置读范围
            bb_t *= 4;
            if (run_gemm(variant ^ 1, pa, bBt, NULL, pc, ba, bb_t, bc,
                         M, N, K, lda, ldb_t, ldc, alpha, beta) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
        } else {
            if (run_gemm(variant, pa, VK_NULL_HANDLE, pb, pc, ba, bb, bc,
                         M, N, K, lda, ldb, ldc, alpha, beta) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
        }
        pa += stride_a * 4; pb += stride_b * 4; pc += stride_c * 4;
    }
    pthread_mutex_unlock(&g.lock);
    return VKBLAS_OK;
}
