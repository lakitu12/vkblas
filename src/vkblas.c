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

    // bf16 回退: cvt pipeline (复用 tpl 布局: 2 storage buffer + 20B push)
    // 0 = bf16→fp32, 1 = bf16→fp32转置, 2 = fp32→bf16 (ldout 偶, 列对), 3 = fp32→bf16 (原子)
    VkPipeline cpipe[8];
    VkDescriptorSet cdset[8];
    // complex64 回退: cvt_cx (3-binding layout) + cx_combine (4-binding layout)
    VkDescriptorSetLayout cxl;      // 3 storage (In, OutR, OutI / InR, InI, Out)
    VkPipelineLayout cxpl;
    VkPipeline cxpipes[2];          // 0 = 交错→平面, 1 = 平面→交错
    VkDescriptorSet cxdsets[2];
    VkDescriptorSetLayout cml;      // 4 storage (Cr, Ci, Cold, Cout)
    VkPipelineLayout cmpl;
    VkPipeline cmpipe;
    VkDescriptorSet cmdset;
    // 内部 fp32 临时 buffer (grow-only)
    // [0]=Ar [1]=Ai [2]=Br [3]=Bi [4]=BrT [5]=BiT [6]=Cr [7]=Ci
    VkDeviceMemory imem[8];
    VkBuffer ibuf[8];
    size_t ibuf_size[8];

    // fp64 回退: d64 GEMM pipeline (复用 g.dsl 3-binding, push 56B)
    VkPipelineLayout d64pl;
    VkPipeline d64pipe[4];      // idx = op_a*2+op_b (nn,tn,nt,tt)
    VkPipeline d64tpipe;        // transpose_d64 (复用 tdsl/tpl 布局)
    // complex128 回退: cvt_cz (复用 cxl/cxpl 布局) + cx_combine_d64 (复用 cml, push 48B)
    VkPipeline czpipe;          // cvt_cz_planar (complex128 → double 平面)
    VkDescriptorSet czdset;
    VkPipelineLayout cm64pl;
    VkPipeline cm64pipe;
    VkDescriptorSet cmdset64;

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

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40 };  // GEMM 3+转置 2+cvt 4×2+fp16 cvt 3×2+cx 2×3+cmb 4+cz 3+cm64 4
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 15, .poolSizeCount = 1, .pPoolSizes = &ps };
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

    // ---- cvt pipeline (复用 tdsl/tpl: 2 storage buffer + 20B push) ----
    // 0 = bf16→fp32, 1 = bf16→fp32 转置输出 (B 快路径), 2 = fp32→bf16 列对, 3 = fp32→bf16 原子
    // 4 = fp16→fp32, 5 = (预留), 6 = fp32→fp16 列对, 7 = fp32→fp16 原子
    static const char* cnames[8] = { "cvt_b2f.spv", "cvt_b2f_tsp.spv", "cvt_f2b.spv", "cvt_f2b_atomic.spv",
                                     "cvt_h2f.spv", NULL, "cvt_f2h.spv", "cvt_f2h_atomic.spv" };
    for (int i = 0; i < 8; i++) {
        if (cnames[i] == NULL) continue;
        if (load_spv(cnames[i], &sm) != 0) continue;  // 缺 shader 则对应回退不可用
        VkComputePipelineCreateInfo ccp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.tpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &ccp, NULL, &g.cpipe[i]), "c pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
        VkDescriptorSetAllocateInfo cdai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.tdsl };
        vk_check(vkAllocateDescriptorSets(g.dev, &cdai, &g.cdset[i]), "c dset");
    }

    // ---- complex64: cvt_cx pipeline (3 storage buffer + 20B push) ----
    VkDescriptorSetLayoutBinding xbnds[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo xdslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = xbnds };
    vk_check(vkCreateDescriptorSetLayout(g.dev, &xdslci, NULL, &g.cxl), "cx dsl");
    VkPushConstantRange xpcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 20 };
    VkPipelineLayoutCreateInfo xplci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.cxl, .pushConstantRangeCount = 1, .pPushConstantRanges = &xpcr };
    vk_check(vkCreatePipelineLayout(g.dev, &xplci, NULL, &g.cxpl), "cx pl");
    static const char* xnames[2] = { "cvt_cx_planar.spv", "cvt_cx_inter.spv" };
    for (int i = 0; i < 2; i++) {
        if (load_spv(xnames[i], &sm) != 0) continue;
        VkComputePipelineCreateInfo xcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.cxpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &xcp, NULL, &g.cxpipes[i]), "cx pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
        VkDescriptorSetAllocateInfo xdsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.cxl };
        vk_check(vkAllocateDescriptorSets(g.dev, &xdsai, &g.cxdsets[i]), "cx dset");
    }

    // ---- complex64: cx_combine pipeline (4 storage buffer + 32B push) ----
    VkDescriptorSetLayoutBinding mbnds[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo mdslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4, .pBindings = mbnds };
    vk_check(vkCreateDescriptorSetLayout(g.dev, &mdslci, NULL, &g.cml), "cm dsl");
    VkPushConstantRange mpcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 32 };
    VkPipelineLayoutCreateInfo mplci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.cml, .pushConstantRangeCount = 1, .pPushConstantRanges = &mpcr };
    vk_check(vkCreatePipelineLayout(g.dev, &mplci, NULL, &g.cmpl), "cm pl");
    if (load_spv("cx_combine.spv", &sm) == 0) {
        VkComputePipelineCreateInfo mcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.cmpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &mcp, NULL, &g.cmpipe), "cm pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
        VkDescriptorSetAllocateInfo mdsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.cml };
        vk_check(vkAllocateDescriptorSets(g.dev, &mdsai, &g.cmdset), "cm dset");
    }

    // ---- complex128: cvt_cz pipeline (复用 cxl/cxpl: 3 storage + 20B push) ----
    if (load_spv("cvt_cz_planar.spv", &sm) == 0) {
        VkComputePipelineCreateInfo zcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.cxpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &zcp, NULL, &g.czpipe), "cz pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
        VkDescriptorSetAllocateInfo zdsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.cxl };
        vk_check(vkAllocateDescriptorSets(g.dev, &zdsai, &g.czdset), "cz dset");
    }
    // ---- complex128: cx_combine_d64 (复用 cml 4-binding; push 48B: 4 uint + 4 double) ----
    VkPushConstantRange m64pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 48 };
    VkPipelineLayoutCreateInfo m64plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.cml, .pushConstantRangeCount = 1, .pPushConstantRanges = &m64pcr };
    vk_check(vkCreatePipelineLayout(g.dev, &m64plci, NULL, &g.cm64pl), "cm64 pl");
    if (load_spv("cx_combine_d64.spv", &sm) == 0) {
        VkComputePipelineCreateInfo m64cp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.cm64pl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &m64cp, NULL, &g.cm64pipe), "cm64 pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
        VkDescriptorSetAllocateInfo m64dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.cml };
        vk_check(vkAllocateDescriptorSets(g.dev, &m64dsai, &g.cmdset64), "cm64 dset");
    }

    // ---- fp64: d64 GEMM pipeline (复用 g.dsl 3-binding; push 56B: 10 uint + 2 double) ----
    VkPushConstantRange d64pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 56 };
    VkPipelineLayoutCreateInfo d64plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &d64pcr };
    vk_check(vkCreatePipelineLayout(g.dev, &d64plci, NULL, &g.d64pl), "d64 pl");
    // 索引 = op_a*2+op_b → {nn, nt, tn, tt} (与 fp32 引擎一致!)
    static const char* d64names[4] = { "gemm_d64_nn.spv", "gemm_d64_nt.spv", "gemm_d64_tn.spv", "gemm_d64_tt.spv" };
    for (int i = 0; i < 4; i++) {
        if (load_spv(d64names[i], &sm) != 0) continue;
        VkComputePipelineCreateInfo dcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.d64pl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &dcp, NULL, &g.d64pipe[i]), "d64 pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }
    if (load_spv("transpose_d64.spv", &sm) == 0) {
        VkComputePipelineCreateInfo dtcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.tpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &dtcp, NULL, &g.d64tpipe), "d64 tpipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }

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
        .pNext = &embci, .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

// 转置核心: out[N][K] = in[K][N]^T (row-major), in 行步长 ldin, out 紧密 (行步长 N)
// bOut 由调用方提供 (可为 tbuf 池或内部 buffer)
static int transpose_into(VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t need,
                          uint32_t K, uint32_t N, uint32_t ldin, uint32_t* ldout) {
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, need} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    // Out 是 (N×K) 紧密, 行步长 = K (不是 N! 非方阵时 N≠K 会写错位)
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

    *ldout = K;   // Out (N×K) 紧密, 行步长 K
    return 0;
}

// 转置: 内部 buffer 版 (bf16/complex 回退用); 输出全局池 tbuf, 用完即复用
static int transpose_vk(VkBuffer bIn, size_t bytes_in,
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
    int rc = transpose_into(bIn, bytes_in, g.tbuf, need, K, N, ldin, ldout);
    if (rc == 0) *out_vk = g.tbuf;
    return rc;
}

// 转置: HIP 指针版 (fp32 GEMM TB=0 场景)
static int transpose_buf(const void* in, size_t bytes_in,
                         uint32_t K, uint32_t N, uint32_t ldin,
                         VkBuffer* out_vk, uint32_t* ldout) {
    VkBuffer bIn; VkDeviceMemory mIn; VkDeviceSize oIn;
    if (import_ptr(in, bytes_in, &bIn, &mIn, &oIn) != 0) return -1;
    int rc = transpose_vk(bIn, bytes_in, K, N, ldin, out_vk, ldout);
    release_ptr(mIn, bIn);
    return rc;
}

// 纯 VkBuffer 版 GEMM 提交 (buffer 已 import 或为内部 buffer, 由调用方管理生命周期)
static int run_gemm_vk(int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                       VkBuffer bC, size_t bc,
                       uint32_t M, uint32_t N, uint32_t K,
                       uint32_t lda, uint32_t ldb, uint32_t ldc,
                       float alpha, float beta) {
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

    int rc = run_gemm_vk(variant, bA, ba, bB, bb, bC, bc, M, N, K, lda, ldb, ldc, alpha, beta);

    release_ptr(mA, bA);
    if (bB_imported) release_ptr(mB, bB);
    release_ptr(mC, bC);
    return rc;
}

// ---------- bf16 回退支持 ----------
// 内部 fp32 临时 buffer (grow-only, device-local)
static int ensure_ibuf(int i, size_t need) {
    if (g.ibuf[i] != VK_NULL_HANDLE && g.ibuf_size[i] >= need) return 0;
    if (g.ibuf[i] != VK_NULL_HANDLE) {
        vkFreeMemory(g.dev, g.imem[i], NULL);
        vkDestroyBuffer(g.dev, g.ibuf[i], NULL);
        g.ibuf[i] = VK_NULL_HANDLE;
    }
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(g.dev, &bci, NULL, &g.ibuf[i]) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.dev, g.ibuf[i], &mr);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size };
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &mprops);
    for (uint32_t t = 0; t < mprops.memoryTypeCount; t++) {
        if (mr.memoryTypeBits & (1u << t) &&
            (mprops.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mai.memoryTypeIndex = t; break;
        }
    }
    if (vkAllocateMemory(g.dev, &mai, NULL, &g.imem[i]) != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(g.dev, g.ibuf[i], g.imem[i], 0) != VK_SUCCESS) return -1;
    g.ibuf_size[i] = need;
    return 0;
}

// cvt 执行: bIn → bOut
// pipe: 0=b2f 1=b2f_tsp(转置输出) 2=f2b(列对,要求 ldout 偶) 3=f2b(原子,任意 ldout)
// R×C 有效区域, ldin/ldout 元素步长; 分块: 纯搬运, 大 dispatch 会性能崩溃 → 8MB/块
static int cvt_run(int pipe, VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t bytes_out,
                   uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout) {
    if (g.cpipe[pipe] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, bytes_out} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.cdset[pipe], .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    struct { uint32_t R, C, ldin, ldout, roff; } pc = { R, C, ldin, ldout, 0 };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cpipe[pipe]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.cdset[pipe], 0, NULL);
    if (pipe == 3 || pipe == 7) {
        // 线性 uint 对模式 (f2b/f2h 奇数 ldout): total = (R-1)*ldout+C 半字,
        // 每线程 2 个连续半字写 1 uint (非原子, 跨行安全, 无需前置清零)
        uint32_t total = (R - 1u) * ldout + C;
        uint32_t npairs = (total + 1u) / 2u;
        uint32_t pairs_per = (8u * 1024 * 1024) / 8u;   // 8MB / 8B 每对
        if (pairs_per > (1u << 20)) pairs_per = (1u << 20);
        for (uint32_t ro = 0; ro < npairs; ro += pairs_per) {
            pc.roff = ro;
            uint32_t np = npairs - ro < pairs_per ? npairs - ro : pairs_per;
            vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(g.cmd, (np + 255) / 256, 1, 1);
        }
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cvt submit");
        vkQueueWaitIdle(g.queue);
        return 0;
    }
    // 每块最多 ~2M 元素 (8MB/4B), 按行切
    uint32_t rows_per = (8u * 1024 * 1024) / (C * 4u);
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    for (uint32_t ro = 0; ro < R; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = R - ro < rows_per ? R - ro : rows_per;
        // 每行线程数: b2f 2元素/线程 → C/2; f2b 列对 → ceil(C/2); f2b 原子 → C
        uint32_t per_row = (pipe <= 1) ? ((C + 1) / 2) : (pipe == 2 ? (C + 1) / 2 : C);
        uint32_t threads = rows * per_row;
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (threads + 255) / 256, 1, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cvt submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// bf16 计时 (VKBLAS_TRACE=1 时打印各阶段耗时)
static uint64_t now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
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

// bf16 GEMM 回退: A/B/C (bf16, 2B/元素) → 内部 fp32 → fp32 GEMM → 回写 bf16
// op_b==N 时 cvt 直接输出转置 (N×K 紧密) 走 TB=1 快路径 (省独立转置)
vkblas_status_t vkblas_gemm_bf16(
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
    if (!g.ready || g.cpipe[0] == VK_NULL_HANDLE) {
        pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_INIT;
    }

    // 有效区域与字节数 (元素单位)
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;          // A 行数
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;          // A 每行有效元素
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;            // A 元素数
    size_t ba2  = ba_e * 4;                               // A2 fp32 字节
    size_t bb_e, bb2;
    uint32_t Rb, Cb;
    int variant;
    if (op_b == VKBLAS_OP_N) {
        // B 转置在 fp32 化后单独做 (transpose.comp 快路径, cvt_tsp 直接写太慢)
        Rb = K; Cb = N;
        bb_e = (size_t)(K - 1) * ldb + N;
        bb2  = bb_e * 4;                                  // 先按原布局 fp32 化
        variant = (int)op_a * 2 + 1;                      // TB=1 (转置后)
    } else {
        Rb = N; Cb = K;
        bb_e = (size_t)(N - 1) * ldb + K;
        bb2  = bb_e * 4;
        variant = (int)op_a * 2 + (int)op_b;
    }
    size_t bc_e = (size_t)(M - 1) * ldc + N;
    size_t bc2  = bc_e * 4;

    if (ensure_ibuf(0, ba2) != 0 || ensure_ibuf(1, bb2) != 0 || ensure_ibuf(2, bc2) != 0) {
        pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT;
    }

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    int trace = getenv("VKBLAS_TRACE") != NULL;
    for (uint32_t i = 0; i < batch; i++) {
        // import bf16 输入
        uint64_t t0 = trace ? now_us() : 0;
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 2, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pb, bb_e * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pc, bc_e * 2, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        // A: bf16 → fp32 (布局不变); B: bf16 → fp32, 若 op_b==N 再转置 (快路径)
        int rc = 0;
        rc |= cvt_run(0, bA, ba_e * 2, g.ibuf[0], ba2, Ra, Ca, lda, lda);
        uint64_t t1 = trace ? now_us() : 0;
        rc |= cvt_run(0, bB, bb_e * 2, g.ibuf[1], bb2, Rb, Cb, ldb, ldb);
        uint64_t t2 = trace ? now_us() : 0;
        VkBuffer bBt = VK_NULL_HANDLE;
        uint32_t ldb_eff = ldb;
        if (op_b == VKBLAS_OP_N) {
            // ibuf[1] 是 (K×N, ldb) → 转置成 (N×K) 紧密, 走 TB=1
            if (transpose_vk(g.ibuf[1], bb2, K, N, ldb, &bBt, &ldb_eff) != 0) rc = -1;
        }
        uint64_t t3 = trace ? now_us() : 0;
        // C: beta!=0 时读入 fp32; beta==0 时 ibuf[2] 无需初始化 (GEMM 覆盖写)
        if (beta != 0.0f)
            rc |= cvt_run(0, bC, bc_e * 2, g.ibuf[2], bc2, M, N, ldc, ldc);
        uint64_t t3b = trace ? now_us() : 0;
        if (rc != 0) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT;
        }

        // fp32 GEMM (A2/B2 内部, C2 内部; B 可能走转置快路径)
        size_t bb2_used = bb2;
        if (bBt != VK_NULL_HANDLE) {
            bb2_used = (size_t)N * K * 4;   // 转置后 (N×K) 紧密
        }
        rc = run_gemm_vk(variant, g.ibuf[0], ba2, bBt != VK_NULL_HANDLE ? bBt : g.ibuf[1],
                         bb2_used, g.ibuf[2], bc2,
                         M, N, K, lda, ldb_eff, ldc, alpha, beta);
        uint64_t t4 = trace ? now_us() : 0;
        // 回写: fp32 → bf16 (ldc 偶数走列对快路径, 奇数走原子变体; 原子路径由 cvt_run 内部先清零)
        if (rc == 0)
            rc = cvt_run((ldc & 1) ? 3 : 2, g.ibuf[2], bc2, bC, bc_e * 2, M, N, ldc, ldc);
        uint64_t t5 = trace ? now_us() : 0;
        if (trace)
            fprintf(stderr, "[vk] bf16: import %lu cvtA %lu cvtB %lu bTsp %lu cvtC %lu gemm %lu f2b %lu us (total %lu)\n",
                    (unsigned long)(t1 - t0), (unsigned long)(t1 - t0), (unsigned long)(t2 - t1),
                    (unsigned long)(t3 - t2), (unsigned long)(t3b - t3),
                    (unsigned long)(t4 - t3b), (unsigned long)(t5 - t4),
                    (unsigned long)(t5 - t0));
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        pa += stride_a * 2; pb += stride_b * 2; pc += stride_c * 2;
    }
    pthread_mutex_unlock(&g.lock);
    return VKBLAS_OK;
}

// fp16 GEMM 回退: A/B/C (fp16, 2B/元素) → 内部 fp32 → fp32 GEMM → 回写 fp16
// 管道同 bf16 (cvt 索引 4/6/7), 位转换不同: unpackHalf2x16/packHalf2x16 (RNE)
// op_b==N 时 cvt 后单独转置 (transpose.comp 快路径) 走 TB=1
vkblas_status_t vkblas_gemm_f16(
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
    if (!g.ready || g.cpipe[4] == VK_NULL_HANDLE) {
        pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_INIT;
    }

    // 有效区域与字节数 (元素单位)
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;          // A 行数
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;          // A 每行有效元素
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;            // A 元素数
    size_t ba2  = ba_e * 4;                               // A2 fp32 字节
    size_t bb_e, bb2;
    uint32_t Rb, Cb;
    int variant;
    if (op_b == VKBLAS_OP_N) {
        // B 转置在 fp32 化后单独做 (transpose.comp 快路径)
        Rb = K; Cb = N;
        bb_e = (size_t)(K - 1) * ldb + N;
        bb2  = bb_e * 4;
        variant = (int)op_a * 2 + 1;                      // TB=1 (转置后)
    } else {
        Rb = N; Cb = K;
        bb_e = (size_t)(N - 1) * ldb + K;
        bb2  = bb_e * 4;
        variant = (int)op_a * 2 + (int)op_b;
    }
    size_t bc_e = (size_t)(M - 1) * ldc + N;
    size_t bc2  = bc_e * 4;

    if (ensure_ibuf(0, ba2) != 0 || ensure_ibuf(1, bb2) != 0 || ensure_ibuf(2, bc2) != 0) {
        pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT;
    }

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    int trace = getenv("VKBLAS_TRACE") != NULL;
    for (uint32_t i = 0; i < batch; i++) {
        // import fp16 输入
        uint64_t t0 = trace ? now_us() : 0;
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 2, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pb, bb_e * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pc, bc_e * 2, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        // A: fp16 → fp32 (布局不变); B: fp16 → fp32, 若 op_b==N 再转置 (快路径)
        int rc = 0;
        rc |= cvt_run(4, bA, ba_e * 2, g.ibuf[0], ba2, Ra, Ca, lda, lda);
        uint64_t t1 = trace ? now_us() : 0;
        rc |= cvt_run(4, bB, bb_e * 2, g.ibuf[1], bb2, Rb, Cb, ldb, ldb);
        uint64_t t2 = trace ? now_us() : 0;
        VkBuffer bBt = VK_NULL_HANDLE;
        uint32_t ldb_eff = ldb;
        if (op_b == VKBLAS_OP_N) {
            // ibuf[1] 是 (K×N, ldb) → 转置成 (N×K) 紧密, 走 TB=1
            if (transpose_vk(g.ibuf[1], bb2, K, N, ldb, &bBt, &ldb_eff) != 0) rc = -1;
        }
        uint64_t t3 = trace ? now_us() : 0;
        // C: beta!=0 时读入 fp32; beta==0 时 ibuf[2] 无需初始化 (GEMM 覆盖写)
        if (beta != 0.0f)
            rc |= cvt_run(4, bC, bc_e * 2, g.ibuf[2], bc2, M, N, ldc, ldc);
        uint64_t t3b = trace ? now_us() : 0;
        if (rc != 0) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT;
        }

        // fp32 GEMM (A2/B2 内部, C2 内部; B 可能走转置快路径)
        size_t bb2_used = bb2;
        if (bBt != VK_NULL_HANDLE) {
            bb2_used = (size_t)N * K * 4;   // 转置后 (N×K) 紧密
        }
        rc = run_gemm_vk(variant, g.ibuf[0], ba2, bBt != VK_NULL_HANDLE ? bBt : g.ibuf[1],
                         bb2_used, g.ibuf[2], bc2,
                         M, N, K, lda, ldb_eff, ldc, alpha, beta);
        uint64_t t4 = trace ? now_us() : 0;
        // 回写: fp32 → fp16 (ldc 偶数走列对快路径, 奇数走原子变体; 原子路径由 cvt_run 内部先清零)
        if (rc == 0)
            rc = cvt_run((ldc & 1) ? 7 : 6, g.ibuf[2], bc2, bC, bc_e * 2, M, N, ldc, ldc);
        uint64_t t5 = trace ? now_us() : 0;
        if (trace)
            fprintf(stderr, "[vk] f16: import %lu cvtA %lu cvtB %lu bTsp %lu cvtC %lu gemm %lu f2h %lu us (total %lu)\n",
                    (unsigned long)(t1 - t0), (unsigned long)(t1 - t0), (unsigned long)(t2 - t1),
                    (unsigned long)(t3 - t2), (unsigned long)(t3b - t3),
                    (unsigned long)(t4 - t3b), (unsigned long)(t5 - t4),
                    (unsigned long)(t5 - t0));
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        pa += stride_a * 2; pb += stride_b * 2; pc += stride_c * 2;
    }
    pthread_mutex_unlock(&g.lock);
    return VKBLAS_OK;
}

// ---------- complex64 回退 ----------
// 数学: C = alpha·(A·B) + beta·C_old, alpha/beta 为 complex
//   A·B = (ArBr - AiBi) + i(ArBi + AiBr) → 4 次 fp32 GEMM
//   T1 = ArBr - AiBi; T2 = ArBi + AiBr
//   C.r = ar·T1 - ai·T2 + br·C_old.r - bi·C_old.i
//   C.i = ar·T2 + ai·T1 + br·C_old.i + bi·C_old.r  (cx_combine shader)
// A/B/C 为 complex 交错存储 (实虚交替, 8B/元素), ld 为元素单位
// cvt_cx_planar: 交错 → 实/虚平面 (fp32, 布局不变); GEMM 后 cx_combine 组合回写
static int cx_run(int pipe, VkBuffer bIn, size_t bytes_in,
                  VkBuffer bOutR, size_t bytes_out, VkBuffer bOutI,
                  uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout) {
    if (g.cxpipes[pipe] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[3] = { {bIn, 0, bytes_in}, {bOutR, 0, bytes_out}, {bOutI, 0, bytes_out} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.cxdsets[pipe], .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t R, C, ldin, ldout, roff; } pc = { R, C, ldin, ldout, 0 };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cxpipes[pipe]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cxpl, 0, 1, &g.cxdsets[pipe], 0, NULL);
    uint32_t rows_per = (8u * 1024 * 1024) / (C * 8u);
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    for (uint32_t ro = 0; ro < R; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = R - ro < rows_per ? R - ro : rows_per;
        vkCmdPushConstants(g.cmd, g.cxpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (rows * C + 255) / 256, 1, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cx submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// combine: C_out (complex) = alpha·(T1 + i·T2) + beta·C_old (complex)
static int cx_combine(VkBuffer bT1, size_t bT1_sz, VkBuffer bT2, size_t bT2_sz,
                      VkBuffer bCold, size_t bCold_sz, VkBuffer bCout, size_t bCout_sz,
                      uint32_t M, uint32_t N, uint32_t ldc,
                      float ar, float ai, float br, float bi) {
    if (g.cmpipe == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[4] = { {bT1, 0, bT1_sz}, {bT2, 0, bT2_sz},
                                     {bCold, 0, bCold_sz}, {bCout, 0, bCout_sz} };
    VkWriteDescriptorSet wds[4];
    for (int i = 0; i < 4; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.cmdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 4, wds, 0, NULL);

    struct { uint32_t R, C, ldc, roff; float ar, ai, br, bi; } pc = { M, N, ldc, 0, ar, ai, br, bi };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cmpipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cmpl, 0, 1, &g.cmdset, 0, NULL);
    uint32_t rows_per = (8u * 1024 * 1024) / (N * 8u);
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    for (uint32_t ro = 0; ro < M; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = M - ro < rows_per ? M - ro : rows_per;
        vkCmdPushConstants(g.cmd, g.cmpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (rows * N + 255) / 256, 1, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cm submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// complex64 GEMM: 语义同 vkblas_gemm_f32 的 row-major 翻译, 但 A/B/C 为 complex64
// 返回: 0=OK (含 fallback 到真库), 1=引擎不可用 (调用方应转发真库)
int vkblas_gemm_c64(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    float alpha_r, float alpha_i,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    float beta_r, float beta_i,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    if (M == 0 || N == 0 || K == 0 || batch == 0) return 1;
    pthread_mutex_lock(&g.lock);
    ensure_init();
    if (!g.ready || g.cxpipes[0] == VK_NULL_HANDLE || g.cmpipe == VK_NULL_HANDLE) {
        pthread_mutex_unlock(&g.lock); return 1;
    }

    // 有效区域 (元素单位)
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;        // complex 元素数
    size_t ba_p = ba_e * 4;                           // 平面 fp32 字节
    size_t bb_e, bb_p;
    uint32_t Rb, Cb;
    int variant;
    if (op_b == VKBLAS_OP_N) {
        Rb = K; Cb = N;
        bb_e = (size_t)(K - 1) * ldb + N;
        bb_p = bb_e * 4;
        variant = (int)op_a * 2 + 1;                  // TB=1 (转置后)
    } else {
        Rb = N; Cb = K;
        bb_e = (size_t)(N - 1) * ldb + K;
        bb_p = bb_e * 4;
        variant = (int)op_a * 2 + (int)op_b;
    }
    size_t bc_e = (size_t)(M - 1) * ldc + N;
    size_t bc_p = bc_e * 4;
    // 转置后 (N×K) 紧密
    size_t bt_p = (op_b == VKBLAS_OP_N) ? (size_t)N * K * 4 : 0;

    // 内部 buffer: [0]=Ar [1]=Ai [2]=Br [3]=Bi [4]=BrT [5]=BiT [6]=T1(Cr) [7]=T2(Ci)
    if (ensure_ibuf(0, ba_p) != 0 || ensure_ibuf(1, ba_p) != 0 ||
        ensure_ibuf(2, bb_p) != 0 || ensure_ibuf(3, bb_p) != 0 ||
        (op_b == VKBLAS_OP_N && (ensure_ibuf(4, bt_p) != 0 || ensure_ibuf(5, bt_p) != 0)) ||
        ensure_ibuf(6, bc_p) != 0 || ensure_ibuf(7, bc_p) != 0) {
        pthread_mutex_unlock(&g.lock); return 1;
    }

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    for (uint32_t i = 0; i < batch; i++) {
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 8, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pb, bb_e * 8, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pc, bc_e * 8, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return 1; }

        int rc = 0;
        // A/B: 交错 → 平面 (Ar/Ai, Br/Bi)
        rc |= cx_run(0, bA, ba_e * 8, g.ibuf[0], ba_p, g.ibuf[1], Ra, Ca, lda, lda);
        rc |= cx_run(0, bB, bb_e * 8, g.ibuf[2], bb_p, g.ibuf[3], Rb, Cb, ldb, ldb);
        uint32_t ldb_eff = ldb;
        VkBuffer bBr = g.ibuf[2], bBi = g.ibuf[3];
        if (op_b == VKBLAS_OP_N) {
            // Br/Bi → BrT/BiT (N×K 紧密)
            rc |= transpose_into(g.ibuf[2], bb_p, g.ibuf[4], bt_p, K, N, ldb, &ldb_eff);
            rc |= transpose_into(g.ibuf[3], bb_p, g.ibuf[5], bt_p, K, N, ldb, &ldb_eff);
            bBr = g.ibuf[4]; bBi = g.ibuf[5];
        }
        if (rc != 0) { release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return 1; }

        // 4 次 fp32 GEMM (beta 项在 combine 处理, 这里全用 beta=0 或累加)
        // T1 = ArBr; T1 -= AiBi; T2 = ArBi; T2 += AiBr
        rc |= run_gemm_vk(variant, g.ibuf[0], ba_p, bBr, bt_p ? bt_p : bb_p,
                          g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0f, 0.0f);
        rc |= run_gemm_vk(variant, g.ibuf[1], ba_p, bBi, bt_p ? bt_p : bb_p,
                          g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, -1.0f, 1.0f);
        rc |= run_gemm_vk(variant, g.ibuf[0], ba_p, bBi, bt_p ? bt_p : bb_p,
                          g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0f, 0.0f);
        rc |= run_gemm_vk(variant, g.ibuf[1], ba_p, bBr, bt_p ? bt_p : bb_p,
                          g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0f, 1.0f);
        // combine: C = alpha·(T1+iT2) + beta·C_old (读 bC 旧值, 写回 bC)
        if (rc == 0)
            rc = cx_combine(g.ibuf[6], bc_p, g.ibuf[7], bc_p, bC, bc_e * 8, bC, bc_e * 8,
                            M, N, ldc, alpha_r, alpha_i, beta_r, beta_i);
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return 1; }

        pa += stride_a * 8; pb += stride_b * 8; pc += stride_c * 8;
    }
    pthread_mutex_unlock(&g.lock);
    return 0;
}

// ---------- fp64 回退 ----------
// d64 转置: In (K×N, ldin) → Out (N×K 紧密, ldout=K), double
static int transpose_d64_vk(VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t need,
                            uint32_t K, uint32_t N, uint32_t ldin, uint32_t* ldout) {
    if (g.d64tpipe == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, need} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);
    struct { uint32_t R, C, ldin, ldout, roff; } pc = { K, N, ldin, K, 0 };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.d64tpipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
    const uint32_t ROWS_PER_DISP = (8u * 1024 * 1024) / (N * 8u);  // 8MB / 行字节数 (double)
    uint32_t rpd = ROWS_PER_DISP < 32 ? 32 : (ROWS_PER_DISP > 4096 ? 4096 : ROWS_PER_DISP);
    for (uint32_t ro = 0; ro < K; ro += rpd) {
        pc.roff = ro;
        uint32_t rows = K - ro < rpd ? K - ro : rpd;
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        // d64 transpose shader: r 覆盖 16 行/块 (tid>>4), 不是 32!
        vkCmdDispatch(g.cmd, (rows + 15) / 16, (N + 31) / 32, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "d64t submit");
    vkQueueWaitIdle(g.queue);
    *ldout = K;
    return 0;
}

// fp64 GEMM 核心 dispatch: 对 Vulkan buffer 直接跑 (d64 pipeline, 3-binding, 56B push)
// 供 vkblas_gemm_f64 (外部指针) 与 vkblas_gemm_z64 (内部 buffer ×4) 复用
static int gemm_d64_dispatch(VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                             VkBuffer bC, size_t bc,
                             uint32_t M, uint32_t N, uint32_t K,
                             uint32_t lda, uint32_t ldb, uint32_t ldc,
                             double alpha, double beta, int variant) {
    if (g.d64pipe[variant] == VK_NULL_HANDLE) return -1;
    // push: 10 uint + 2 double = 56B; double 需 8B 对齐,
    // 9 uint=36B 后 double 在 std430 偏移 40, host 必须补 pad 到 40
    VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int j = 0; j < 3; j++) {
        wds[j] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)j, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[j] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc, pad; double alpha, beta; } d64pc = {
        M, N, K, (M + 31) / 32, (N + 31) / 32, (K + 31) / 32, lda, ldb, ldc, 0, alpha, beta };
    if (getenv("VKBLAS_TRACE"))
        fprintf(stderr, "[vk] d64pc: M=%u N=%u K=%u lda=%u ldb=%u ldc=%u alpha=%g beta=%g variant=%d\n",
                M, N, K, lda, ldb, ldc, alpha, beta, variant);
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.d64pipe[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.d64pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.d64pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(d64pc), &d64pc);
    vkCmdDispatch(g.cmd, d64pc.Mt, d64pc.Nt, 1);
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "d64 submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// fp64 GEMM: A/B/C 为 double (8B/元素); 内部纯 double shader
// 返回: 0=已处理, 非 0=引擎不可用 (调用方转发真库)
int vkblas_gemm_f64(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    double alpha,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    double beta,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    if (M == 0 || N == 0 || K == 0 || batch == 0) return 1;
    pthread_mutex_lock(&g.lock);
    ensure_init();
    if (!g.ready || g.d64pipe[0] == VK_NULL_HANDLE) {
        pthread_mutex_unlock(&g.lock); return 1;
    }

    size_t ba = (op_a == VKBLAS_OP_T) ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K;
    size_t bb = (op_b == VKBLAS_OP_T) ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N;
    size_t bc = (size_t)(M - 1) * ldc + N;
    ba *= 8; bb *= 8; bc *= 8;  // double 字节

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    for (uint32_t i = 0; i < batch; i++) {
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pb, bb, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pc, bc, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return 1; }

        int variant = (int)op_a * 2 + (int)op_b;
        VkBuffer bB_use = bB;
        size_t bb_use = bb;
        uint32_t ldb_use = ldb;
        if (op_b == VKBLAS_OP_N && g.d64tpipe != VK_NULL_HANDLE) {
            // B 行优先 (K×N, ldb) → 转置成 (N×K) 紧密, 走 TB=1 快路径
            size_t bt = (size_t)N * K * 8;
            // 必须先 ensure_ibuf 再 transpose (g.ibuf[0] 可能未分配!)
            if (ensure_ibuf(0, bt) != 0) {
                release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
                pthread_mutex_unlock(&g.lock); return 1;
            }
            int trc = transpose_d64_vk(bB, bb, g.ibuf[0], bt, K, N, ldb, &ldb_use);
            if (getenv("VKBLAS_TRACE"))
                fprintf(stderr, "[vk] d64 transpose: rc=%d ldb=%u→%u bt=%zu\n", trc, ldb, ldb_use, bt);
            if (trc != 0) {
                release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
                pthread_mutex_unlock(&g.lock); return 1;
            }
            bB_use = g.ibuf[0]; bb_use = bt;
            variant = (int)op_a * 2 + 1;  // TB=1
        }

        // d64 GEMM dispatch (核心逻辑在 gemm_d64_dispatch)
        int rc = gemm_d64_dispatch(bA, ba, bB_use, bb_use, bC, bc,
                                   M, N, K, lda, ldb_use, ldc, alpha, beta, variant);
        if (rc != 0) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return 1;
        }

        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        pa += stride_a * 8; pb += stride_b * 8; pc += stride_c * 8;
    }
    pthread_mutex_unlock(&g.lock);
    return 0;
}

// ---------- complex128 回退 ----------
// 数学同 complex64, 但 4 次 fp64 GEMM (复用 d64 引擎):
//   A·B = (ArBr - AiBi) + i(ArBi + AiBr)
//   T1 = ArBr - AiBi; T2 = ArBi + AiBr
//   C.r = ar·T1 - ai·T2 + br·C_old.r - bi·C_old.i
//   C.i = ar·T2 + ai·T1 + br·C_old.i + bi·C_old.r  (cx_combine_d64 shader)
// A/B/C 为 complex128 交错 (16B/元素), ld 为元素单位
// cvt_cz_planar: 交错 → 实/虚 double 平面 (布局不变); GEMM 后 cx_combine_d64 组合回写
static int cz_run(VkBuffer bIn, size_t bytes_in,
                  VkBuffer bOutR, size_t bytes_out, VkBuffer bOutI,
                  uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout) {
    if (g.czpipe == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[3] = { {bIn, 0, bytes_in}, {bOutR, 0, bytes_out}, {bOutI, 0, bytes_out} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.czdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t R, C, ldin, ldout, roff; } pc = { R, C, ldin, ldout, 0 };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.czpipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cxpl, 0, 1, &g.czdset, 0, NULL);
    uint32_t rows_per = (8u * 1024 * 1024) / (C * 8u);  // 输出 double 平面 8B/元素
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    for (uint32_t ro = 0; ro < R; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = R - ro < rows_per ? R - ro : rows_per;
        vkCmdPushConstants(g.cmd, g.cxpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (rows * C + 255) / 256, 1, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cz submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// combine: C_out (complex128) = alpha·(T1 + i·T2) + beta·C_old, 全 double
// push: 4 uint + 4 double = 48B (double 8B 对齐于 16B 偏移 ✓)
static int cx_combine_d64(VkBuffer bT1, size_t bT1_sz, VkBuffer bT2, size_t bT2_sz,
                          VkBuffer bCold, size_t bCold_sz, VkBuffer bCout, size_t bCout_sz,
                          uint32_t M, uint32_t N, uint32_t ldc,
                          double ar, double ai, double br, double bi) {
    if (g.cm64pipe == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[4] = { {bT1, 0, bT1_sz}, {bT2, 0, bT2_sz},
                                     {bCold, 0, bCold_sz}, {bCout, 0, bCout_sz} };
    VkWriteDescriptorSet wds[4];
    for (int i = 0; i < 4; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.cmdset64, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 4, wds, 0, NULL);

    struct { uint32_t R, C, ldc, roff; double ar, ai, br, bi; } pc = { M, N, ldc, 0, ar, ai, br, bi };
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &cbbi);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cm64pipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cm64pl, 0, 1, &g.cmdset64, 0, NULL);
    uint32_t rows_per = (8u * 1024 * 1024) / (N * 16u);  // C 元素 16B
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    for (uint32_t ro = 0; ro < M; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = M - ro < rows_per ? M - ro : rows_per;
        vkCmdPushConstants(g.cmd, g.cm64pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (rows * N + 255) / 256, 1, 1);
    }
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cm64 submit");
    vkQueueWaitIdle(g.queue);
    return 0;
}

// complex128 GEMM: 语义同 vkblas_gemm_c64, 但 A/B/C 为 complex128 (16B/元素),
// 4 次 fp64 GEMM (d64 引擎) + cx_combine_d64 组合
// 返回: 0=OK (含 fallback 到真库), 1=引擎不可用 (调用方应转发真库)
int vkblas_gemm_z64(
    vkblas_op_t op_a, vkblas_op_t op_b,
    uint32_t M, uint32_t N, uint32_t K,
    double alpha_r, double alpha_i,
    const void* A, uint32_t lda,
    const void* B, uint32_t ldb,
    double beta_r, double beta_i,
    void* C, uint32_t ldc,
    uint32_t batch,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    if (M == 0 || N == 0 || K == 0 || batch == 0) return 1;
    pthread_mutex_lock(&g.lock);
    ensure_init();
    if (!g.ready || g.czpipe == VK_NULL_HANDLE || g.cm64pipe == VK_NULL_HANDLE ||
        g.d64pipe[0] == VK_NULL_HANDLE) {
        pthread_mutex_unlock(&g.lock); return 1;
    }

    // 有效区域 (元素单位)
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;        // complex128 元素数
    size_t ba_p = ba_e * 8;                           // double 平面字节
    size_t bb_e, bb_p;
    uint32_t Rb, Cb;
    int variant;
    if (op_b == VKBLAS_OP_N) {
        Rb = K; Cb = N;
        bb_e = (size_t)(K - 1) * ldb + N;
        bb_p = bb_e * 8;
        variant = (int)op_a * 2 + 1;                  // TB=1 (转置后)
    } else {
        Rb = N; Cb = K;
        bb_e = (size_t)(N - 1) * ldb + K;
        bb_p = bb_e * 8;
        variant = (int)op_a * 2 + (int)op_b;
    }
    size_t bc_e = (size_t)(M - 1) * ldc + N;
    size_t bc_p = bc_e * 8;
    // 转置后 (N×K) 紧密 (double)
    size_t bt_p = (op_b == VKBLAS_OP_N) ? (size_t)N * K * 8 : 0;

    // 内部 buffer: [0]=Ar [1]=Ai [2]=Br [3]=Bi [4]=BrT [5]=BiT [6]=T1(Cr) [7]=T2(Ci)
    if (ensure_ibuf(0, ba_p) != 0 || ensure_ibuf(1, ba_p) != 0 ||
        ensure_ibuf(2, bb_p) != 0 || ensure_ibuf(3, bb_p) != 0 ||
        (op_b == VKBLAS_OP_N && (ensure_ibuf(4, bt_p) != 0 || ensure_ibuf(5, bt_p) != 0)) ||
        ensure_ibuf(6, bc_p) != 0 || ensure_ibuf(7, bc_p) != 0) {
        pthread_mutex_unlock(&g.lock); return 1;
    }

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    for (uint32_t i = 0; i < batch; i++) {
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 16, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pb, bb_e * 16, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return 1; }
        if (import_ptr(pc, bc_e * 16, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return 1; }

        int rc = 0;
        // A/B: 交错 → double 平面 (Ar/Ai, Br/Bi)
        rc |= cz_run(bA, ba_e * 16, g.ibuf[0], ba_p, g.ibuf[1], Ra, Ca, lda, lda);
        rc |= cz_run(bB, bb_e * 16, g.ibuf[2], bb_p, g.ibuf[3], Rb, Cb, ldb, ldb);
        uint32_t ldb_eff = ldb;
        VkBuffer bBr = g.ibuf[2], bBi = g.ibuf[3];
        if (op_b == VKBLAS_OP_N) {
            // Br/Bi → BrT/BiT (N×K 紧密)
            rc |= transpose_d64_vk(g.ibuf[2], bb_p, g.ibuf[4], bt_p, K, N, ldb, &ldb_eff);
            rc |= transpose_d64_vk(g.ibuf[3], bb_p, g.ibuf[5], bt_p, K, N, ldb, &ldb_eff);
            bBr = g.ibuf[4]; bBi = g.ibuf[5];
        }
        if (rc != 0) { release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return 1; }

        // 4 次 fp64 GEMM (beta 项在 combine 处理, 这里全用 beta=0 或累加)
        // T1 = ArBr; T1 -= AiBi; T2 = ArBi; T2 += AiBr
        rc |= gemm_d64_dispatch(g.ibuf[0], ba_p, bBr, bt_p ? bt_p : bb_p,
                                g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 0.0, variant);
        rc |= gemm_d64_dispatch(g.ibuf[1], ba_p, bBi, bt_p ? bt_p : bb_p,
                                g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, -1.0, 1.0, variant);
        rc |= gemm_d64_dispatch(g.ibuf[0], ba_p, bBi, bt_p ? bt_p : bb_p,
                                g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 0.0, variant);
        rc |= gemm_d64_dispatch(g.ibuf[1], ba_p, bBr, bt_p ? bt_p : bb_p,
                                g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 1.0, variant);
        // combine: C = alpha·(T1+iT2) + beta·C_old (读 bC 旧值, 写回 bC)
        if (rc == 0)
            rc = cx_combine_d64(g.ibuf[6], bc_p, g.ibuf[7], bc_p, bC, bc_e * 16, bC, bc_e * 16,
                                M, N, ldc, alpha_r, alpha_i, beta_r, beta_i);
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return 1; }

        pa += stride_a * 16; pb += stride_b * 16; pc += stride_c * 16;
    }
    pthread_mutex_unlock(&g.lock);
    return 0;
}
