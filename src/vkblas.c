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
#include <hip/hip_runtime_api.h>

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
// dma-buf 导入缓存见 import_ptr/ic_*: 条目按 HIP 块基址失效 (hipFree hook), 见下
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
VkPipeline mpipe[8];  // matvec: [0]=TB0 [1]=TB1 (M==1 单 pass), [2]=TB0 sk [3]=TB1 sk
    //              [4]=sk fp16 (TB0) [5]=sk bf16 (TB0) [6,7]=预留
    VkPipeline rhpipe[2];  // split_k_reduce_h: [0]=fp16, [1]=bf16
    VkPipeline pipe128[4]; // v7-128 tile (llama l_warptile 移植): 128×128, BK16, 64 acc/线程
    // f16/bf16 直通 GEMM (llama.cpp mul_mm 思路: 2B 半精度直进 LDS, 无 cvt 中间 buffer)
    // [0]=fp16, [1]=bf16; 每 dtype 4 变体 + 转置
    VkPipeline pipe_h[2][4];
    VkPipeline pipe128_h[2][4];
    VkPipeline tpipe_h[2];
    // split-k (llama.cpp 借鉴: K 分段并行 + reduce 归约); 仅 fp32
    VkPipeline pipe_sk[4];        // v6 tile split-k 4 变体
    VkPipeline pipe_sk128[4];     // v7-128 tile split-k 4 变体
    VkPipeline sk_reduce_pipe;
    VkDeviceMemory sk_mem;
    VkBuffer sk_buf;
    size_t sk_buf_size;
    // B 转置缓存 (VKBLAS_CACHE_TRANSPOSE=1, 推理场景权重不变): key=(ptr,N,K,ldb)
    // 供 op_b==T && M==1 的 matvec 复用 (PyTorch decode: TB=1 256 路行流 6.4ms → TB=0 0.55ms)
    struct tc_entry { const void* ptr; uint32_t N, K, ldb; size_t size; VkBuffer b; VkDeviceMemory m; const void* base; } tc_tab[8];
    int tc_cnt;
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
    VkDescriptorSet cdset[16];  // [0..7]=pipe 默认, [8..15]=融合路径实例槽 (A/B/C 独立)
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

static uint64_t now_us(void);  // 定义见 bf16 计时段 (VKBLAS_PROFILE/TRACE 用)

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
    const char* env = getenv("VKBLAS_SHADER_DIR");
    if (env && *env)
        snprintf(path, sizeof(path), "%s/%s", env, name);
    else
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

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 48 };
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
    // matvec (M==1 decode): 免 B 转置; [0]=B(K×N) 直读, [1]=B(N×K) 列读, [2/3]=split-k 版
    static const char* mnames[8] = { "matvec_n.spv", "matvec_t.spv",
                                     "matvec_sk_n.spv", "matvec_sk_t.spv",
                                     "matvec_sk_h16.spv", "matvec_sk_bf16.spv",
                                     NULL, NULL };
    for (int i = 0; i < 8; i++) {
        g.mpipe[i] = VK_NULL_HANDLE;
        if (mnames[i] == NULL) continue;
        if (load_spv(mnames[i], &sm) != 0) continue;
        VkComputePipelineCreateInfo mcp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.pl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &mcp, NULL, &g.mpipe[i]), "matvec pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }
    // v7-128 tile (实验性, 缺 shader 则不可用; VKBLAS_TILE128=1 时 fp32 GEMM 走此路径)
    static const char* names128[4] = { "gemm128_nn.spv", "gemm128_nt.spv", "gemm128_tn.spv", "gemm128_tt.spv" };
    for (int i = 0; i < 4; i++) {
        g.pipe128[i] = VK_NULL_HANDLE;
        if (load_spv(names128[i], &sm) != 0) continue;
        VkComputePipelineCreateInfo cp128 = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.pl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cp128, NULL, &g.pipe128[i]), "pipeline128");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 72 };  // GEMM 3+转置 2+cvt 8×2+实例 8×2+cx 2×3+cmb 4+cz 3+cm64 4
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 24, .poolSizeCount = 1, .pPoolSizes = &ps };
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
    VkPushConstantRange tpcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 24 };
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
    // 融合路径实例槽 [8..15]: 每 pipe 2 个额外 (A/B 同 pipe 需独立 dset)
    for (int i = 8; i < 16; i++) {
        VkDescriptorSetAllocateInfo cdai2 = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = g.dpool, .descriptorSetCount = 1, .pSetLayouts = &g.tdsl };
        vk_check(vkAllocateDescriptorSets(g.dev, &cdai2, &g.cdset[i]), "c dset2");
    }

    // ---- f16/bf16 直通 GEMM pipeline (复用 g.dsl/g.pl: 3 storage + 44B push) ----
    // dtype: 0 = fp16 (h16), 1 = bf16 (b16); 缺 shader 则对应 dtype 回退旧 cvt 管道
    static const char* hnames[2][4] = {
        { "gemm_h16_nn.spv", "gemm_h16_nt.spv", "gemm_h16_tn.spv", "gemm_h16_tt.spv" },
        { "gemm_b16_nn.spv", "gemm_b16_nt.spv", "gemm_b16_tn.spv", "gemm_b16_tt.spv" } };
    static const char* h128names[2][4] = {
        { "gemm128_h16_nn.spv", "gemm128_h16_nt.spv", "gemm128_h16_tn.spv", "gemm128_h16_tt.spv" },
        { "gemm128_b16_nn.spv", "gemm128_b16_nt.spv", "gemm128_b16_tn.spv", "gemm128_b16_tt.spv" } };
    static const char* htnames[2] = { "transpose_h16.spv", "transpose_b16.spv" };
    for (int d = 0; d < 2; d++) {
        for (int i = 0; i < 4; i++) {
            g.pipe_h[d][i] = VK_NULL_HANDLE;
            g.pipe128_h[d][i] = VK_NULL_HANDLE;
            if (load_spv(hnames[d][i], &sm) == 0) {
                VkComputePipelineCreateInfo hp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                               VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                    .layout = g.pl };
                vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &hp, NULL, &g.pipe_h[d][i]), "h pipe");
                vkDestroyShaderModule(g.dev, sm, NULL);
            }
            if (load_spv(h128names[d][i], &sm) == 0) {
                VkComputePipelineCreateInfo hp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                               VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                    .layout = g.pl };
                vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &hp, NULL, &g.pipe128_h[d][i]), "h128 pipe");
                vkDestroyShaderModule(g.dev, sm, NULL);
            }
        }
        g.tpipe_h[d] = VK_NULL_HANDLE;
        if (load_spv(htnames[d], &sm) == 0) {
            VkComputePipelineCreateInfo tp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                           VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                .layout = g.tpl };
            vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &tp, NULL, &g.tpipe_h[d]), "ht pipe");
            vkDestroyShaderModule(g.dev, sm, NULL);
        }
    }

    // ---- split-k (llama.cpp 借鉴: K 分段并行 + reduce; 仅 fp32, 复用 g.dsl/g.pl) ----
    static const char* sknames[4] = { "gemm_sk_nn.spv", "gemm_sk_nt.spv", "gemm_sk_tn.spv", "gemm_sk_tt.spv" };
    static const char* sk128names[4] = { "gemm_sk128_nn.spv", "gemm_sk128_nt.spv", "gemm_sk128_tn.spv", "gemm_sk128_tt.spv" };
    for (int i = 0; i < 4; i++) {
        g.pipe_sk[i] = VK_NULL_HANDLE;
        g.pipe_sk128[i] = VK_NULL_HANDLE;
        if (load_spv(sknames[i], &sm) == 0) {
            VkComputePipelineCreateInfo sp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                           VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                .layout = g.pl };
            vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &sp, NULL, &g.pipe_sk[i]), "sk pipe");
            vkDestroyShaderModule(g.dev, sm, NULL);
        }
        if (load_spv(sk128names[i], &sm) == 0) {
            VkComputePipelineCreateInfo sp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                           VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                .layout = g.pl };
            vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &sp, NULL, &g.pipe_sk128[i]), "sk128 pipe");
            vkDestroyShaderModule(g.dev, sm, NULL);
        }
    }
    g.sk_reduce_pipe = VK_NULL_HANDLE;
    static const char* rhnames[2] = { "split_k_reduce_h16.spv", "split_k_reduce_bf16.spv" };
    for (int d = 0; d < 2; d++) {
        g.rhpipe[d] = VK_NULL_HANDLE;
        if (load_spv(rhnames[d], &sm) != 0) continue;
        VkComputePipelineCreateInfo rhp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.tpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &rhp, NULL, &g.rhpipe[d]), "rh pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
    }
    if (load_spv("split_k_reduce.spv", &sm) == 0) {
        VkComputePipelineCreateInfo rp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                       VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            .layout = g.tpl };
        vk_check(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &rp, NULL, &g.sk_reduce_pipe), "sk reduce pipe");
        vkDestroyShaderModule(g.dev, sm, NULL);
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
// vkblas 导出 (供 hipblas 层 hipFree/hipHostFree/hipFreeManaged hook 调用): 失效并释放
// 所有底层 HIP 块基址 == base 的缓存条目 (含 base==NULL → 全清, 测试用)
void vkblas_cache_invalidate_base(const void* base);
// hipblas 层 free hook 的自证实门控: ==1 表示本进程 free 都经过我们 (可安全缓存)
extern int vkblas_hook_active;

// ===== dma-buf import 缓存 (小矩阵热路径: 免去每次 GEMM 的 export/create/alloc/bind) =====
// 正确性模型 (上一版注释的"不能缓存"顾虑已解决):
//   hipFree 后虚拟地址可能被复用给新分配 — 单凭指针无法区分"同一 VA 不同代"。
//   因此缓存条目记录底层 HIP 块基址 (hipPointerGetAttributes().devicePointer),
//   hipblas 层 hook hipFree(ptr) 时按块基址失效全部条目: 块被 free → 其所有 dma-buf
//   引用立即释放, 复用后的新块必然重新 import。hook 漏网的 (HIP 内部 free) 由
//   [守卫] base=NULL 的指针不缓存 兜底。
//   命中成本 = 一次线性查表 (<1μs); miss 成本 = hipPointerGetAttributes (≈1μs) + 原导入。
#define IC_CAP 256
struct ic_entry {
    void* ptr;              // 缓存 key (GEMM 传入的用户指针)
    const void* base;       // 底层 HIP 块基址 (free hook 的失效键; 无法取得 → 不缓存)
    size_t size;            // 创建时的 buffer 字节数
    VkBuffer b;
    VkDeviceMemory m;
    VkDeviceSize off;       // dma-buf 内偏移
};
static struct ic_entry ic_tab[IC_CAP];
static uint32_t ic_cnt = 0;
static uint64_t ic_hits = 0, ic_misses = 0;

// 返回非 NULL 且可用于失效匹配的 HIP 块基址; 无法 vouch 时返回 NULL
static const void* ic_block_base(const void* ptr) {
    hipPointerAttribute_t attr;
    if (hipPointerGetAttributes(&attr, (hipDeviceptr_t)ptr) != hipSuccess) return NULL;
    const void* base = attr.type == hipMemoryTypeHost ? attr.hostPointer : attr.devicePointer;
    return base;
}

// swap-remove 第 i 条并释放其 Vulkan 对象 (调用方须持锁)
static void ic_drop(size_t i) {
    vkDestroyBuffer(g.dev, ic_tab[i].b, NULL);
    vkFreeMemory(g.dev, ic_tab[i].m, NULL);
    ic_tab[i] = ic_tab[--ic_cnt];
}

// 释放所有底层块 == base 的条目 (base==NULL → 全清)
void vkblas_cache_invalidate_base(const void* base) {
    if (!g.ready) return;
    pthread_mutex_lock(&g.lock);
    for (size_t i = 0; i < ic_cnt; ) {
        if (base == NULL || ic_tab[i].base == base) ic_drop(i);
        else i++;
    }
    // 转置缓存同失效 (指针复用 → 陈旧转置危险)
    for (int i = 0; i < g.tc_cnt; ) {
        if (base == NULL || g.tc_tab[i].base == base) {
            vkDestroyBuffer(g.dev, g.tc_tab[i].b, NULL);
            vkFreeMemory(g.dev, g.tc_tab[i].m, NULL);
            g.tc_tab[i] = g.tc_tab[--g.tc_cnt];
        } else i++;
    }
    pthread_mutex_unlock(&g.lock);
}

// 注册新条目 (满则 FIFO 逐出最旧的)
static void ic_add(const void* ptr, const void* base, size_t size,
                   VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    if (ic_cnt >= IC_CAP) ic_drop(0);   // 简单 FIFO
    struct ic_entry* e = &ic_tab[ic_cnt++];
    e->ptr = (void*)ptr; e->base = base; e->size = size;
    e->b = b; e->m = m; e->off = off;
}

static int import_ptr(const void* ptr, size_t size, VkBuffer* buf,
                      VkDeviceMemory* mem, VkDeviceSize* offset) {
    if (!g.ready) return -1;
    int profile = getenv("VKBLAS_PROFILE") != NULL;
    double t0 = profile ? now_us() : 0;

    if (ptr != NULL && vkblas_hook_active) {
        // ---- 缓存快路径: 同一 ptr 且 size 满足 → 直接复用 ----
        // 双保险: 命中前校验底层 HIP 块基址仍一致 (hook 漏网场景兜底)
        for (uint32_t i = 0; i < ic_cnt; i++) {
            if (ic_tab[i].ptr == ptr) {
                if (size <= ic_tab[i].size && ic_block_base(ptr) == ic_tab[i].base) {
                    *buf = ic_tab[i].b; *mem = ic_tab[i].m; *offset = ic_tab[i].off;
                    ic_hits++;
                    if (getenv("VKBLAS_TRACE"))
                        fprintf(stderr, "[vkblas] import cache HIT  ptr=%p size=%zu (cached=%zu, n=%u)\n",
                                ptr, size, ic_tab[i].size, ic_cnt);
                    return 0;
                }
                ic_drop(i);   // size 不够或块已变 → 作废重导
                break;
            }
        }
    }

    // 可缓存的先取块基址 (hook 未证实或无法 vouch → 不缓存, 每次全量导出)
    const void* cbase = (ptr != NULL && vkblas_hook_active) ? ic_block_base(ptr) : NULL;
    int cacheable = cbase != NULL;

    int fd = -1;
    uint64_t off = 0;
    hsa_status_t st = g.hsa_export(ptr, size, &fd, &off);
    if (st != 0) {
        if (getenv("VKBLAS_TRACE"))
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
    if (getenv("VKBLAS_TRACE")) {
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
    }
    if (cacheable) {
        ic_add(ptr, cbase, size, b, m, off);
        ic_misses++;
        if (getenv("VKBLAS_TRACE"))
            fprintf(stderr, "[vkblas] import cache MISS ptr=%p size=%zu (n=%u)\n", ptr, size, ic_cnt);
    }
    if (profile)
        fprintf(stderr, "[vkblas] prof: import ptr=%p size=%zu took %.2fms%s\n",
                ptr, size, (now_us() - t0) / 1e3, cacheable ? " (cached)" : " (uncacheable)");
    *buf = b; *mem = m; *offset = off;
    return 0;
}

static void release_ptr(VkDeviceMemory mem, VkBuffer buf) {
    // 缓存条目由缓存持有生命周期 (hipFree hook / 容量逐出时统一释放), 这里只在
    // 非缓存对象 (临时 buffer 等) 上真释放
    for (uint32_t i = 0; i < ic_cnt; i++) {
        if (ic_tab[i].b == buf && ic_tab[i].m == mem) return;
    }
    vkDestroyBuffer(g.dev, buf, NULL);
    vkFreeMemory(g.dev, mem, NULL);
}

// v7-128 tile 适用性 (所有 GEMM 路径统一入口):
//   128×128 tile 在 M,N≥256 时快 5-76%; 小 shape (单/少 wg) 用 v6 (64×64) 保持并行度;
//   NN 实测定界 (gfx803, 2026-08-23):
//     方阵/近方阵: v7 多数胜 (1.05-1.07x), 大方阵 v6 略快 (0.97-0.99x)
//     M<N (高瘦): v7 胜或持平 (0.90-1.07x), batch 大时 (bs512) v6 更优 (0.90x)
//     M>N 窄 N (Nt≤4) + Mt≤32 + tiles≥128: v6 胜 (0.81x) — 仅 (4096,512) 边界
//     其余 M>N 窄 N: v7 大胜 (1.13-3.29x)
//   环境变量 VKBLAS_TILE128=0/1 强制覆盖 (调试/对比)
static int pick_tile128(uint32_t M, uint32_t N) {
    const char* env = getenv("VKBLAS_TILE128");
    if (env != NULL) {
        if (env[0] == '0' && env[1] == '\0') return 0;
        if (env[0] == '1' && env[1] == '\0') return 1;
    }
    if (M < 256 || N < 256) return 0;
    uint32_t Mt = (M + 127) / 128, Nt = (N + 127) / 128;
    // M>N 窄 N: v7 workgroup 数少 → 占用率不足; 仅在 (4096,512) 边界 v6 胜
    if (M > N && Nt <= 4 && Mt * Nt >= 128 && Mt <= 32) return 0;
    return 1;
}

// Record helpers used when several compute passes share one submission.
static void cmd_begin(void) {
    vk_check(vkResetCommandBuffer(g.cmd, 0), "reset command buffer");
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vk_check(vkBeginCommandBuffer(g.cmd, &cbbi), "begin command buffer");
}

static void cmd_barrier(VkBuffer buffer, VkDeviceSize size) {
    VkBufferMemoryBarrier bmb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = 0,
        .size = size,
    };
    vkCmdPipelineBarrier(g.cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &bmb, 0, NULL);
}

static void cmd_end_submit(const char* what) {
    int profile = getenv("VKBLAS_PROFILE") != NULL;
    uint64_t t0 = profile ? now_us() : 0;
    vk_check(vkEndCommandBuffer(g.cmd), what);
    uint64_t t1 = profile ? now_us() : 0;
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
    vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), what);
    uint64_t t2 = profile ? now_us() : 0;
    vk_check(vkQueueWaitIdle(g.queue), what);
    uint64_t t3 = profile ? now_us() : 0;
    if (profile)
        fprintf(stderr, "[vkblas] prof: %s end=%.2fus submit=%.2fus wait=%.2fms\n",
                what, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3);
}

// 转置核心: out[N][K] = in[K][N]^T (row-major), in 行步长 ldin, out 紧密 (行步长 N)
// bOut 由调用方提供 (可为 tbuf 池或内部 buffer)
static int transpose_into(VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t need,
                          uint32_t K, uint32_t N, uint32_t ldin, uint32_t* ldout,
                          uint32_t batch, int64_t stride_in, int64_t stride_out,
                          int submit) {
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, need} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    // Out 是 (N×K) 紧密, 行步长 = K (不是 N! 非方阵时 N≠K 会写错位)
    // batch_base: RADV/gfx803 z>16 后 workgroup 调度串行, 拆批 z≤16
    struct { uint32_t R, C, ldin, ldout, roff, batch, batch_base, stride_in, stride_out; } pc = {
        K, N, ldin, K, 0, 0, 0, (uint32_t)stride_in, (uint32_t)stride_out };
    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
    // 按 8MB/块动态分块 (实测最优 ~2048 wg/8MB; 固定 512 行对小 N 块太小)
    // VKBLAS_TRANSPOSE_BLK env 覆盖 行数/块 (实验)
    uint32_t rpd;
    const char* tbenv = getenv("VKBLAS_TRANSPOSE_BLK");
    if (tbenv && *tbenv) {
        rpd = (uint32_t)strtoul(tbenv, NULL, 10);
    } else {
        const uint32_t ROWS_PER_DISP = (8u * 1024 * 1024) / (N * 4u);
        rpd = ROWS_PER_DISP < 32 ? 32 : (ROWS_PER_DISP > 4096 ? 4096 : ROWS_PER_DISP);
    }
    for (uint32_t bo = 0; bo < batch; bo += 16) {
        pc.batch_base = bo;
        pc.batch = batch - bo < 16 ? batch - bo : 16;
        for (uint32_t ro = 0; ro < K; ro += rpd) {
            pc.roff = ro;
            uint32_t rows = K - ro < rpd ? K - ro : rpd;
            vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(g.cmd, (rows + 31) / 32, (N + 31) / 32, pc.batch);
        }
    }
    if (submit) cmd_end_submit("transpose submit");

    *ldout = K;   // Out (N×K) 紧密, 行步长 K
    return 0;
}

// 转置: 内部 buffer 版 (bf16/complex 回退用); 输出全局池 tbuf, 用完即复用
static int transpose_vk(VkBuffer bIn, size_t bytes_in,
                        uint32_t K, uint32_t N, uint32_t ldin,
                        VkBuffer* out_vk, uint32_t* ldout,
                        uint32_t batch, int64_t stride_in, int64_t stride_out,
                        int submit) {
    size_t need = (size_t)batch * N * K * 4;
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
    int rc = transpose_into(bIn, bytes_in, g.tbuf, need, K, N, ldin, ldout, batch, stride_in, stride_out, submit);
    if (rc == 0) *out_vk = g.tbuf;
    return rc;
}

// 转置: HIP 指针版 (fp32 GEMM TB=0 场景); batch>1 时一次转置全部 batch (输出紧密串联)
static int transpose_buf(const void* in, size_t bytes_in,
                         uint32_t K, uint32_t N, uint32_t ldin,
                         VkBuffer* out_vk, uint32_t* ldout,
                         uint32_t batch, int64_t stride_in, int64_t stride_out) {
    size_t bytes_all = bytes_in + (batch > 1 && stride_in > 0 ? (size_t)stride_in * (batch - 1) * 4 : 0);
    VkBuffer bIn; VkDeviceMemory mIn; VkDeviceSize oIn;
    if (import_ptr(in, bytes_all, &bIn, &mIn, &oIn) != 0) {
        if (getenv("VKBLAS_TRACE")) fprintf(stderr, "[vk] merged: transpose import B failed ptr=%p size=%zu\n", in, bytes_all);
        return -1;
    }
    int rc = transpose_vk(bIn, bytes_all, K, N, ldin, out_vk, ldout, batch, stride_in, stride_out, 1);
    release_ptr(mIn, bIn);
    return rc;
}

// ---- f16/bf16 直通 (llama.cpp mul_mm 思路) ----
// 转置核心 (2B 元素): out[N][K] = in[K][N]^T, 输出行步长 pad 偶 (消除跨行共享 uint)
// dtype: 0 = fp16, 1 = bf16
static int transpose_h_into(VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t need,
                            uint32_t K, uint32_t N, uint32_t ldin,
                            uint32_t* ldout, int dtype,
                            uint32_t batch, int64_t stride_in, int64_t stride_out,
                            int submit) {
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, need} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    // Out 是 (N×K), 行步长 pad 偶 (K 奇时 +1, 保证行首 uint 对齐)
    uint32_t ldout_pad = K + (K & 1u);
    // batch_base: RADV/gfx803 z>16 后 workgroup 调度串行, 拆批 z≤16
    struct { uint32_t R, C, ldin, ldout, roff, batch, batch_base, stride_in, stride_out; } pc = {
        K, N, ldin, ldout_pad, 0, 0, 0, (uint32_t)stride_in, (uint32_t)stride_out };
    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpipe_h[dtype]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
    // 8MB/块分块 (2B 元素: 行字节 = N*2)
    const uint32_t ROWS_PER_DISP = (8u * 1024 * 1024) / (N * 2u);
    uint32_t rpd = ROWS_PER_DISP < 32 ? 32 : (ROWS_PER_DISP > 4096 ? 4096 : ROWS_PER_DISP);
    for (uint32_t bo = 0; bo < batch; bo += 16) {
        pc.batch_base = bo;
        pc.batch = batch - bo < 16 ? batch - bo : 16;
        for (uint32_t ro = 0; ro < K; ro += rpd) {
            pc.roff = ro;
            uint32_t rows = K - ro < rpd ? K - ro : rpd;
            vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(g.cmd, (rows + 31) / 32, (N + 15) / 16, pc.batch);
        }
    }
    if (submit) cmd_end_submit("half transpose submit");

    *ldout = ldout_pad;
    return 0;
}

// 转置 (2B): 内部 buffer 池版
static int transpose_h_vk(VkBuffer bIn, size_t bytes_in,
                          uint32_t K, uint32_t N, uint32_t ldin,
                          VkBuffer* out_vk, uint32_t* ldout, int dtype,
                          uint32_t batch, int64_t stride_in, int64_t stride_out,
                          int submit) {
    size_t need = (size_t)batch * N * (K + (K & 1u)) * 2;
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
    int rc = transpose_h_into(bIn, bytes_in, g.tbuf, need, K, N, ldin, ldout, dtype,
                              batch, stride_in, stride_out, submit);
    if (rc == 0) *out_vk = g.tbuf;
    return rc;
}

// 直通 GEMM 提交 (v6 tile, 2B 元素; 复用 g.dsl/g.pl, 44B push)
static int run_gemm_h(int dtype, int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                      VkBuffer bC, size_t bc,
                      uint32_t M, uint32_t N, uint32_t K,
                      uint32_t lda, uint32_t ldb, uint32_t ldc,
                      uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                      float alpha, float beta, int submit) {
    if (g.pipe_h[dtype][variant] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[3] = {
        {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc;
             uint32_t batch_stride_a, batch_stride_b, batch_stride_c; float alpha, beta; } pc = {
        M, N, K, (M + 63) / 64, (N + 63) / 64, (K + 31) / 32, lda, ldb, ldc,
        (uint32_t)stride_a, (uint32_t)stride_b, (uint32_t)stride_c, alpha, beta };

    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_h[dtype][variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, pc.Mt * batch, pc.Nt, 1);
    if (submit) cmd_end_submit("half GEMM submit");
    return 0;
}

// 直通 GEMM 提交 (v7-128 tile, 2B 元素)
static int run_gemm_h128(int dtype, int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                         VkBuffer bC, size_t bc,
                         uint32_t M, uint32_t N, uint32_t K,
                         uint32_t lda, uint32_t ldb, uint32_t ldc,
                         uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                         float alpha, float beta, int submit) {
    if (g.pipe128_h[dtype][variant] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[3] = {
        {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc;
             uint32_t batch_stride_a, batch_stride_b, batch_stride_c; float alpha, beta; } pc = {
        M, N, K, (M + 127) / 128, (N + 127) / 128, (K + 15) / 16, lda, ldb, ldc,
        (uint32_t)stride_a, (uint32_t)stride_b, (uint32_t)stride_c, alpha, beta };

    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe128_h[dtype][variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, pc.Mt * batch, pc.Nt, 1);
    if (submit) cmd_end_submit("half GEMM128 submit");
    return 0;
}

// Forward declarations: the fused helper is kept next to the half path, while
// the fp32 record-only submitters are defined below.
static int run_gemm_vk(int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                       VkBuffer bC, size_t bc, uint32_t M, uint32_t N, uint32_t K,
                       uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t batch,
                       int64_t stride_a, int64_t stride_b, int64_t stride_c,
                       float alpha, float beta, int submit);
static int run_gemm_vk128(int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                          VkBuffer bC, size_t bc, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t batch,
                          int64_t stride_a, int64_t stride_b, int64_t stride_c,
                          float alpha, float beta, int submit);

// Record one transpose and one half GEMM into the same command buffer.
static int run_fused_half_transpose_gemm(int dtype, int use128, int variant,
                                         VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                                         VkBuffer bC, size_t bc,
                                         uint32_t M, uint32_t N, uint32_t K,
                                         uint32_t lda, uint32_t ldb, uint32_t ldc,
                                         uint32_t batch, int64_t stride_a, int64_t stride_b,
                                         int64_t stride_c, float alpha, float beta) {
    VkBuffer bBt = VK_NULL_HANDLE;
    uint32_t ldb_t = ldb;
    cmd_begin();
    if (transpose_h_vk(bB, bb, K, N, ldb, &bBt, &ldb_t, dtype,
                       batch, stride_b, (int64_t)N * (K + (K & 1u)), 0) != 0) {
        vkEndCommandBuffer(g.cmd);
        return -1;
    }
    size_t bb_t = (size_t)batch * N * ldb_t * 2;
    cmd_barrier(bBt, bb_t);
    int rc;
    if (use128)
        rc = run_gemm_h128(dtype, variant, bA, ba, bBt, bb_t, bC, bc,
                           M, N, K, lda, ldb_t, ldc, batch, stride_a,
                           (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
    else
        rc = run_gemm_h(dtype, variant, bA, ba, bBt, bb_t, bC, bc,
                        M, N, K, lda, ldb_t, ldc, batch, stride_a,
                        (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
    if (rc != 0) {
        vkEndCommandBuffer(g.cmd);
        return rc;
    }
    cmd_end_submit("fused half transpose+GEMM");
    return 0;
}

// Record one transpose and one fp32 GEMM into the same command buffer.
static int run_fused_f32_transpose_gemm(int use128, int variant,
                                        VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                                        VkBuffer bC, size_t bc,
                                        uint32_t M, uint32_t N, uint32_t K,
                                        uint32_t lda, uint32_t ldb, uint32_t ldc,
                                        uint32_t batch, int64_t stride_a, int64_t stride_b,
                                        int64_t stride_c, float alpha, float beta) {
    VkBuffer bBt = VK_NULL_HANDLE;
    uint32_t ldb_t = ldb;
    cmd_begin();
    if (transpose_vk(bB, bb, K, N, ldb, &bBt, &ldb_t, batch,
                     stride_b, (int64_t)N * K, 0) != 0) {
        vkEndCommandBuffer(g.cmd);
        return -1;
    }
    size_t bb_t = (size_t)batch * N * ldb_t * 4;
    cmd_barrier(bBt, bb_t);
    int rc;
    if (use128)
        rc = run_gemm_vk128(variant ^ 1, bA, ba, bBt, bb_t, bC, bc,
                            M, N, K, lda, ldb_t, ldc, batch, stride_a,
                            (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
    else
        rc = run_gemm_vk(variant ^ 1, bA, ba, bBt, bb_t, bC, bc,
                         M, N, K, lda, ldb_t, ldc, batch, stride_a,
                         (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
    if (rc != 0) {
        vkEndCommandBuffer(g.cmd);
        return rc;
    }
    cmd_end_submit("fused fp32 transpose+GEMM");
    return 0;
}

// 直通 batch 合并 (f16/bf16): 一次 import 全 batch + 一次转置 + 一次 dispatch (z=batch)
// 要求 stride_c 偶 (写 C 对不变式: e = b*stride_c + row*ldc + cc 恒偶); 跨分配 → FALLBACK
static vkblas_status_t gemm_h_direct_merged(int dtype, vkblas_op_t op_a, vkblas_op_t op_b,
                                            uint32_t M, uint32_t N, uint32_t K,
                                            float alpha,
                                            const void* A, uint32_t lda,
                                            const void* B, uint32_t ldb,
                                            float beta,
                                            void* C, uint32_t ldc,
                                            uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;
    size_t bb_e, bc_e = (size_t)(M - 1) * ldc + N;
    int variant;
    if (op_b == VKBLAS_OP_N) { bb_e = (size_t)(K - 1) * ldb + N; variant = (int)op_a * 2 + 1; }
    else                     { bb_e = (size_t)(N - 1) * ldb + K; variant = (int)op_a * 2 + (int)op_b; }
    size_t ba_all = ba_e + (stride_a > 0 ? (size_t)stride_a * (batch - 1) : 0);   // 元素
    size_t bb_all = bb_e + (stride_b > 0 ? (size_t)stride_b * (batch - 1) : 0);
    size_t bc_all = bc_e + (stride_c > 0 ? (size_t)stride_c * (batch - 1) : 0);
    VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
    if (import_ptr(A, ba_all * 2, &bA, &mA, &oA) != 0) return VKBLAS_ERR_FALLBACK;
    if (import_ptr(B, bb_all * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); return VKBLAS_ERR_FALLBACK; }
    if (import_ptr(C, bc_all * 2, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA); release_ptr(mB, bB); return VKBLAS_ERR_FALLBACK;
    }
    int rc = 0;
    if (op_b == VKBLAS_OP_N) {
        int use128 = pick_tile128(M, N);
        rc = run_fused_half_transpose_gemm(dtype, use128, variant,
                                           bA, ba_all * 2, bB, bb_all * 2, bC, bc_all * 2,
                                           M, N, K, lda, ldb, ldc, batch,
                                           stride_a, stride_b, stride_c, alpha, beta);
    } else if (pick_tile128(M, N)) {
        rc = run_gemm_h128(dtype, variant, bA, ba_all * 2, bB, bb_all * 2, bC, bc_all * 2,
                           M, N, K, lda, ldb, ldc,
                           batch, stride_a, stride_b, stride_c, alpha, beta, 1);
    } else {
        rc = run_gemm_h(dtype, variant, bA, ba_all * 2, bB, bb_all * 2, bC, bc_all * 2,
                        M, N, K, lda, ldb, ldc,
                        batch, stride_a, stride_b, stride_c, alpha, beta, 1);
    }
    release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
    return rc != 0 ? VKBLAS_ERR_FALLBACK : VKBLAS_OK;
}

// 直通 GEMM 主路径 (f16/bf16, llama.cpp mul_mm 思路): 2B 半精度 A/B/C 直接进 shader
// host 保证 ldc 偶 && N 偶; op_b==N 时 B 先 transpose_h (输出行步长 pad 偶)
// dtype: 0 = fp16, 1 = bf16; 调用方持锁
static vkblas_status_t gemm_h_direct(int dtype, vkblas_op_t op_a, vkblas_op_t op_b,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     float alpha,
                                     const void* A, uint32_t lda,
                                     const void* B, uint32_t ldb,
                                     float beta,
                                     void* C, uint32_t ldc,
                                     uint32_t batch,
                                     int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    uint32_t Ra = (op_a == VKBLAS_OP_T) ? K : M;
    uint32_t Ca = (op_a == VKBLAS_OP_T) ? M : K;
    size_t ba_e = (size_t)(Ra - 1) * lda + Ca;
    size_t bb_e;
    int variant;
    if (op_b == VKBLAS_OP_N) {
        bb_e = (size_t)(K - 1) * ldb + N;
        variant = (int)op_a * 2 + 1;                      // 转置后 TB=1
    } else {
        bb_e = (size_t)(N - 1) * ldb + K;
        variant = (int)op_a * 2 + (int)op_b;
    }
    size_t bc_e = (size_t)(M - 1) * ldc + N;

    const char* pa = (const char*)A, *pb = (const char*)B, *pc = (const char*)C;
    int trace = getenv("VKBLAS_TRACE") != NULL;

    // ---- batch>1 合并快路径: 一次 import + 一次 dispatch (z=batch) ----
    // 条件: stride 全 ≥0 且 stride_c 偶 (直通写 C 对不变式: e 恒偶)
    // RADV/gfx803 实测: 单次 dispatch 的 batch 维 >16 后 workgroup 调度退化为串行 (与轴无关)
    // → 拆批 ≤16 (每批独立指针偏移, stride 不变)
    if (batch > 1 && stride_a >= 0 && stride_b >= 0 && stride_c >= 0 && (stride_c & 1) == 0) {
        uint32_t rem = batch;
        const char* pa2 = pa, *pb2 = pb, *pc2 = pc;
        int rc = 0, fallback = 0;
        while (rem > 0) {
            uint32_t nb = rem > 16 ? 16 : rem;
            rc = gemm_h_direct_merged(dtype, op_a, op_b, M, N, K,
                                      alpha, pa2, lda, pb2, ldb, beta, pc2, ldc,
                                      nb, stride_a, stride_b, stride_c);
            if (rc == VKBLAS_OK) {
                pa2 += (int64_t)nb * stride_a * 2;
                pb2 += (int64_t)nb * stride_b * 2;
                pc2 += (int64_t)nb * stride_c * 2;
                rem -= nb;
            } else {
                fallback = 1;
                break;
            }
        }
        if (!fallback) return VKBLAS_OK;
        if (rc != VKBLAS_ERR_FALLBACK) return rc;
        // FALLBACK → 落回逐 batch 循环
    }

    for (uint32_t i = 0; i < batch; i++) {
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 2, &bA, &mA, &oA) != 0) return VKBLAS_ERR_IMPORT;
        if (import_ptr(pb, bb_e * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pc, bc_e * 2, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); return VKBLAS_ERR_IMPORT; }

        int rc = 0;
        if (op_b == VKBLAS_OP_N) {
            int use128 = pick_tile128(M, N);
            rc = run_fused_half_transpose_gemm(dtype, use128, variant,
                                               bA, ba_e * 2, bB, bb_e * 2, bC, bc_e * 2,
                                               M, N, K, lda, ldb, ldc, 1,
                                               0, 0, 0, alpha, beta);
        } else if (pick_tile128(M, N)) {
            rc = run_gemm_h128(dtype, variant, bA, ba_e * 2, bB, bb_e * 2, bC, bc_e * 2,
                               M, N, K, lda, ldb, ldc, 1, 0, 0, 0, alpha, beta, 1);
        } else {
            rc = run_gemm_h(dtype, variant, bA, ba_e * 2, bB, bb_e * 2, bC, bc_e * 2,
                            M, N, K, lda, ldb, ldc, 1, 0, 0, 0, alpha, beta, 1);
        }
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (trace)
            fprintf(stderr, "[vk] %s direct: batch %u done (rc=%d)\n", dtype ? "bf16" : "f16", i, rc);
        if (rc != 0) return VKBLAS_ERR_IMPORT;
        pa += stride_a * 2; pb += stride_b * 2; pc += stride_c * 2;
    }
    return VKBLAS_OK;
}

// 纯 VkBuffer 版 GEMM 提交 (buffer 已 import 或为内部 buffer, 由调用方管理生命周期)
static int run_gemm_vk(int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                       VkBuffer bC, size_t bc,
                       uint32_t M, uint32_t N, uint32_t K,
                       uint32_t lda, uint32_t ldb, uint32_t ldc,
                       uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                       float alpha, float beta, int submit) {
    VkDescriptorBufferInfo db[3] = {
        {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc;
             uint32_t batch_stride_a, batch_stride_b, batch_stride_c; float alpha, beta; } pc = {
        M, N, K, (M + 63) / 64, (N + 63) / 64, (K + 31) / 32, lda, ldb, ldc,
        (uint32_t)stride_a, (uint32_t)stride_b, (uint32_t)stride_c, alpha, beta };

    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, pc.Mt * batch, pc.Nt, 1);
    if (submit) cmd_end_submit("GEMM submit");
    return 0;
}

// v7-128 tile GEMM 提交 (128×128, BK16; llama.cpp l_warptile 移植, VKBLAS_TILE128=1 启用)
static int run_gemm_vk128(int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                          VkBuffer bC, size_t bc,
                          uint32_t M, uint32_t N, uint32_t K,
                          uint32_t lda, uint32_t ldb, uint32_t ldc,
                          uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                          float alpha, float beta, int submit) {
    if (g.pipe128[variant] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[3] = {
        {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc;
             uint32_t batch_stride_a, batch_stride_b, batch_stride_c; float alpha, beta; } pc = {
        M, N, K, (M + 127) / 128, (N + 127) / 128, (K + 15) / 16, lda, ldb, ldc,
        (uint32_t)stride_a, (uint32_t)stride_b, (uint32_t)stride_c, alpha, beta };

    if (submit) cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe128[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, pc.Mt * batch, pc.Nt, 1);
    if (submit) cmd_end_submit("GEMM128 submit");
    return 0;
}

// split-k GEMM 提交 (llama.cpp 借鉴): K 切 split_k 段并行 (dispatch x = Mt×split_k),
// 各段写 Sk[ik*M*N + m*N+n] (紧密 fp32), 再 reduce 归约到 C (alpha/beta 在 reduce 处理)
// use128: 1 = v7-128 tile (Mt = ceil(M/128)), 0 = v6 (Mt = ceil(M/64))
static int run_gemm_sk(int use128, int variant, VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                       VkBuffer bC, size_t bc,
                       uint32_t M, uint32_t N, uint32_t K,
                       uint32_t lda, uint32_t ldb, uint32_t ldc,
                       float alpha, float beta, uint32_t split_k) {
    VkPipeline pipe = use128 ? g.pipe_sk128[variant] : g.pipe_sk[variant];
    if (pipe == VK_NULL_HANDLE || g.sk_reduce_pipe == VK_NULL_HANDLE) return -1;

    // split_k is the number of K partitions. The shader expects the K span
    // per partition, not the partition count.
    uint32_t split_count = split_k;
    uint32_t k_tile = use128 ? 16 : 32;
    uint32_t k_split = (K + split_count - 1) / split_count;
    k_split = ((k_split + k_tile - 1) / k_tile) * k_tile;

    // Sk buffer: [split_count][M][N] 紧密 fp32 (grow-only, device-local)
    size_t need = (size_t)split_count * M * N * 4;
    if (g.sk_buf == VK_NULL_HANDLE || need > g.sk_buf_size) {
        if (g.sk_buf != VK_NULL_HANDLE) {
            vkFreeMemory(g.dev, g.sk_mem, NULL);
            vkDestroyBuffer(g.dev, g.sk_buf, NULL);
        }
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        if (vkCreateBuffer(g.dev, &bci, NULL, &g.sk_buf) != VK_SUCCESS) return -1;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g.dev, g.sk_buf, &mr);
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
        if (vkAllocateMemory(g.dev, &mai, NULL, &g.sk_mem) != VK_SUCCESS) return -1;
        if (vkBindBufferMemory(g.dev, g.sk_buf, g.sk_mem, 0) != VK_SUCCESS) return -1;
        g.sk_buf_size = need;
    }

    uint32_t Mt = use128 ? (M + 127) / 128 : (M + 63) / 64;
    uint32_t Nt = use128 ? (N + 127) / 128 : (N + 63) / 64;
    uint32_t Kt = use128 ? (K + 15) / 16 : (K + 31) / 32;

    // ---- pass 1: 分段 GEMM → Sk ----
    {
        VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {g.sk_buf, 0, need} };
        VkWriteDescriptorSet wds[3];
        for (int i = 0; i < 3; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

        struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc; float alpha, beta; uint32_t k_split; } pc = {
            M, N, K, Mt, Nt, Kt, lda, ldb, ldc, alpha, beta, k_split };

        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, Mt * split_count, Nt, 1);

        VkBufferMemoryBarrier sk_barrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = g.sk_buf,
            .offset = 0,
            .size = need,
        };
        vkCmdPipelineBarrier(g.cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, NULL, 1, &sk_barrier, 0, NULL);
    }

    // ---- pass 2: reduce Sk → C (alpha/beta) ----
    {
        VkDescriptorBufferInfo db[2] = { {g.sk_buf, 0, need}, {bC, 0, bc} };
        VkWriteDescriptorSet wds[2];
        for (int i = 0; i < 2; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

        struct { uint32_t ne, k_num, N, ldc; float alpha, beta; } pc2 = {
            M * N, split_count, N, ldc, alpha, beta };

        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.sk_reduce_pipe);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
        uint32_t ne_align = (M * N + 255) / 256;
        vkCmdDispatch(g.cmd, ne_align, 1, 1);
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "sk submit+reduce");
        vkQueueWaitIdle(g.queue);
    }
    return 0;
}

// matvec (M==1 decode): C(1×N) += A(1×K)·B, 免 B 全量转置
// variant: 0 = B(K×N) 直读, 1 = B(N×K) 列读; 调用方持锁
static int run_matvec(int variant, const void* A, size_t ba,
                      const void* B, size_t bb, void* C, size_t bc,
                      uint32_t K, uint32_t N, uint32_t ldb, uint32_t ldc,
                      float alpha, float beta) {
    if (g.mpipe[variant] == VK_NULL_HANDLE) return -1;
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (import_ptr(B, bb, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); return -1; }
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA); release_ptr(mB, bB); return -1;
    }
    (void)oA; (void)oB; (void)oC;

    VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t K, N, ldb, ldc; float alpha, beta; } pc = {
        K, N, ldb, ldc, alpha, beta };

    cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.mpipe[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t mvcols = 4;
    const char* mcenv = getenv("VKBLAS_MV_COLS");
    if (mcenv && *mcenv) mvcols = (uint32_t)strtoul(mcenv, NULL, 10);
    if (mvcols < 1) mvcols = 1;
    if (mvcols > 8) mvcols = 8;
    uint32_t wg = variant == 0 ? (N + 256 * mvcols - 1) / (256 * mvcols)
                               : (N + 255) / 256;
    vkCmdDispatch(g.cmd, wg, 1, 1);
    cmd_end_submit("matvec submit");

    release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
    return 0;
}

// matvec (M==1, TB=0) VkBuffer B 版: 供 B 转置缓存 (op_b==T 场景) 复用
static int run_matvec_vk(int variant, const void* A, size_t ba,
                         VkBuffer bB, size_t bb, void* C, size_t bc,
                         uint32_t K, uint32_t N, uint32_t ldb, uint32_t ldc,
                         float alpha, float beta) {
    if (g.mpipe[variant] == VK_NULL_HANDLE) return -1;
    VkBuffer bA, bC;
    VkDeviceMemory mA, mC;
    VkDeviceSize oA, oC;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); return -1; }
    (void)oA; (void)oC;

    VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {bC, 0, bc} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

    struct { uint32_t K, N, ldb, ldc; float alpha, beta; } pc = { K, N, ldb, ldc, alpha, beta };
    cmd_begin();
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.mpipe[0]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t wg = (N + 256 * 4 - 1) / (256 * 4);  // MV_COLS=4
    vkCmdDispatch(g.cmd, wg, 1, 1);
    cmd_end_submit("matvec cached-B submit");

    release_ptr(mA, bA); release_ptr(mC, bC);
    return 0;
}

// matvec split-k VkBuffer B 版: 同 run_matvec_sk 但 B 直接 VkBuffer (转置缓存)
static int run_matvec_sk_vk(int variant, const void* A, size_t ba,
                            VkBuffer bB, size_t bb, void* C, size_t bc,
                            uint32_t K, uint32_t N, uint32_t ldb, uint32_t ldc,
                            float alpha, float beta) {
    if (g.mpipe[2 + variant] == VK_NULL_HANDLE || g.sk_reduce_pipe == VK_NULL_HANDLE) return -1;
    (void)bb;
    VkBuffer bA, bC;
    VkDeviceMemory mA, mC;
    VkDeviceSize oA, oC;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); return -1; }
    (void)oA; (void)oC;

    uint32_t Nt = (N + 255) / 256;
    uint32_t seg = 144 / Nt + 1;
    if (seg < 2) seg = 2;
    if (seg > 64) seg = 64;
    size_t need = (size_t)seg * N * 4;
    if (g.sk_buf == VK_NULL_HANDLE || need > g.sk_buf_size) {
        if (g.sk_buf != VK_NULL_HANDLE) {
            vkFreeMemory(g.dev, g.sk_mem, NULL);
            vkDestroyBuffer(g.dev, g.sk_buf, NULL);
        }
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        if (vkCreateBuffer(g.dev, &bci, NULL, &g.sk_buf) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mC, bC);
            return -1;
        }
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g.dev, g.sk_buf, &mr);
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
        if (vkAllocateMemory(g.dev, &mai, NULL, &g.sk_mem) != VK_SUCCESS ||
            vkBindBufferMemory(g.dev, g.sk_buf, g.sk_mem, 0) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mC, bC);
            return -1;
        }
        g.sk_buf_size = need;
    }
    uint32_t k_split = (K + seg - 1) / seg;
    cmd_begin();
    {
        VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, (size_t)K * N * 4}, {g.sk_buf, 0, need} };
        VkWriteDescriptorSet wds[3];
        for (int i = 0; i < 3; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);
        struct { uint32_t K, N, ldb, k_split, pad; } pc = { K, N, ldb, k_split, 0 };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.mpipe[2 + variant]);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, Nt, 1, seg);
        VkBufferMemoryBarrier bmb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = g.sk_buf, .offset = 0, .size = need };
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bmb, 0, NULL);
    }
    {
        VkDescriptorBufferInfo db[2] = { {g.sk_buf, 0, need}, {bC, 0, bc} };
        VkWriteDescriptorSet wds[2];
        for (int i = 0; i < 2; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);
        struct { uint32_t ne, k_num, N, ldc; float alpha, beta; } pc2 = {
            N, seg, N, ldc, alpha, beta };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.sk_reduce_pipe);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
        vkCmdDispatch(g.cmd, (N + 255) / 256, 1, 1);
    }
    cmd_end_submit("matvec cached split-k submit");
    release_ptr(mA, bA); release_ptr(mC, bC);
    return 0;
}

// B 转置缓存构建/查找 (op_b==T && M==1): (N×K, ldb) → (K×N 紧密) 
// 返回 0 且 *out 有效; -1 失败 (不影响调用方走原 TB=1 路径)
static int tc_get(const void* ptr, uint32_t N, uint32_t K, uint32_t ldb,
                  VkBuffer* out, size_t* out_size, int* built) {
    *built = 0;
    for (int i = 0; i < g.tc_cnt; i++) {
        if (g.tc_tab[i].ptr == ptr && g.tc_tab[i].N == N &&
            g.tc_tab[i].K == K && g.tc_tab[i].ldb == ldb) {
            *out = g.tc_tab[i].b;
            *out_size = g.tc_tab[i].size;
            return 0;
        }
    }
    if (g.tc_cnt >= 8) {
        // 满 → 清最旧
        vkDestroyBuffer(g.dev, g.tc_tab[0].b, NULL);
        vkFreeMemory(g.dev, g.tc_tab[0].m, NULL);
        g.tc_tab[0] = g.tc_tab[--g.tc_cnt];
    }
    // import B 并转置
    size_t bb = (size_t)(N - 1) * ldb + K;
    VkBuffer bB; VkDeviceMemory mB; VkDeviceSize oB;
    if (import_ptr(ptr, bb * 4, &bB, &mB, &oB) != 0) return -1;
    const void* cbase = ic_block_base(ptr);
    size_t need = (size_t)N * K * 4;
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VkBuffer bTt; VkDeviceMemory mTt;
    if (vkCreateBuffer(g.dev, &bci, NULL, &bTt) != VK_SUCCESS) { release_ptr(mB, bB); return -1; }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.dev, bTt, &mr);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size };
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &mprops);
    for (uint32_t t = 0; t < mprops.memoryTypeCount; t++) {
        if (mr.memoryTypeBits & (1u << t) &&
            (mprops.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mai.memoryTypeIndex = t; break;
        }
    }
    if (vkAllocateMemory(g.dev, &mai, NULL, &mTt) != VK_SUCCESS ||
        vkBindBufferMemory(g.dev, bTt, mTt, 0) != VK_SUCCESS) {
        vkDestroyBuffer(g.dev, bTt, NULL);
        release_ptr(mB, bB);
        return -1;
    }
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    cmd_begin();
    uint32_t ldout = 0;
    // In (N×K, ldb) → Out (K×N 紧密): transpose_into 的 R/C 是 In 的行列
    transpose_into(bB, bb * 4, bTt, need, N, K, ldb, &ldout, 1, 0, 0, 0);
    cmd_end_submit("cached transpose submit");
    release_ptr(mB, bB);
    // 存缓存 (base 用于失效)
    struct tc_entry* e = &g.tc_tab[g.tc_cnt++];
    e->ptr = ptr; e->N = N; e->K = K; e->ldb = ldb; e->size = need;
    e->b = bTt; e->m = mTt; e->base = cbase;
    *out = bTt; *out_size = need;
    *built = 1;
    return 0;
}

// matvec split-k (M==1 decode, wg 不足时): pass1 各 K 段写 Sk[seg][N], pass2 reduce→C
// 并行度目标 ≥144 wg (36 CU×4); variant: 0=B(K×N), 1=B(N×K)
static int run_matvec_sk(int variant, const void* A, size_t ba,
                         const void* B, size_t bb, void* C, size_t bc,
                         uint32_t K, uint32_t N, uint32_t ldb, uint32_t ldc,
                         float alpha, float beta) {
    if (g.mpipe[2 + variant] == VK_NULL_HANDLE || g.sk_reduce_pipe == VK_NULL_HANDLE) return -1;
    if (g.mpipe[2 + variant] == VK_NULL_HANDLE || g.mpipe[2 + variant] == VK_NULL_HANDLE) return -1;
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (import_ptr(B, bb, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); return -1; }
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA); release_ptr(mB, bB); return -1;
    }
    (void)oA; (void)oB; (void)oC;

    uint32_t Nt = (N + 255) / 256;
    uint32_t seg = 144 / Nt + 1;
    if (seg < 2) seg = 2;
    if (seg > 64) seg = 64;

    // Sk buffer: [seg][N] 紧密 fp32
    size_t need = (size_t)seg * N * 4;
    if (g.sk_buf == VK_NULL_HANDLE || need > g.sk_buf_size) {
        if (g.sk_buf != VK_NULL_HANDLE) {
            vkFreeMemory(g.dev, g.sk_mem, NULL);
            vkDestroyBuffer(g.dev, g.sk_buf, NULL);
        }
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        if (vkCreateBuffer(g.dev, &bci, NULL, &g.sk_buf) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            return -1;
        }
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g.dev, g.sk_buf, &mr);
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
        if (vkAllocateMemory(g.dev, &mai, NULL, &g.sk_mem) != VK_SUCCESS ||
            vkBindBufferMemory(g.dev, g.sk_buf, g.sk_mem, 0) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            return -1;
        }
        g.sk_buf_size = need;
    }

    uint32_t k_split = (K + seg - 1) / seg;
    cmd_begin();

    // ---- pass1: matvec_sk → Sk[seg][N] ----
    {
        VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {g.sk_buf, 0, need} };
        VkWriteDescriptorSet wds[3];
        for (int i = 0; i < 3; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);
        struct { uint32_t K, N, ldb, k_split, seg; } pc = { K, N, ldb, k_split, 0 };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.mpipe[2 + variant]);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, Nt, 1, seg);
        VkBufferMemoryBarrier bmb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = g.sk_buf, .offset = 0, .size = need };
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bmb, 0, NULL);
    }
    // ---- pass2: reduce Sk → C (alpha/beta) ----
    {
        VkDescriptorBufferInfo db[2] = { {g.sk_buf, 0, need}, {bC, 0, bc} };
        VkWriteDescriptorSet wds[2];
        for (int i = 0; i < 2; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);
        struct { uint32_t ne, k_num, N, ldc; float alpha, beta; } pc2 = {
            N, seg, N, ldc, alpha, beta };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.sk_reduce_pipe);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
        vkCmdDispatch(g.cmd, (N + 255) / 256, 1, 1);
    }
    cmd_end_submit("matvec split-k submit");

    release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
    return 0;
}

// matvec split-k (M==1 decode, 2B 元素 fp16/bf16): 同 run_matvec_sk, 但 A/B/C 是 2B
// dtype: 0=fp16, 1=bf16; 要求 ldb 偶 && N 偶 && ldc 偶 (uint 对齐); 调用方持锁
static int run_matvec_sk_h(int dtype, const void* A, size_t ba,
                           const void* B, size_t bb, void* C, size_t bc,
                           uint32_t K, uint32_t N, uint32_t ldb, uint32_t ldc,
                           float alpha, float beta) {
    VkPipeline p1 = g.mpipe[4 + dtype];
    VkPipeline p2 = g.rhpipe[dtype];
    if (p1 == VK_NULL_HANDLE || p2 == VK_NULL_HANDLE) return -1;
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    if (import_ptr(A, ba, &bA, &mA, &oA) != 0) return -1;
    if (import_ptr(B, bb, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); return -1; }
    if (import_ptr(C, bc, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA); release_ptr(mB, bB); return -1;
    }
    (void)oA; (void)oB; (void)oC;

    uint32_t Nt = (N + 511) / 512;   // half: 每线程 2 列, 256 线程 → 512 列/wg
    uint32_t seg = 144 / Nt + 1;
    if (seg < 2) seg = 2;
    if (seg > 64) seg = 64;

    size_t need = (size_t)seg * N * 4;
    if (g.sk_buf == VK_NULL_HANDLE || need > g.sk_buf_size) {
        if (g.sk_buf != VK_NULL_HANDLE) {
            vkFreeMemory(g.dev, g.sk_mem, NULL);
            vkDestroyBuffer(g.dev, g.sk_buf, NULL);
        }
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = need, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        if (vkCreateBuffer(g.dev, &bci, NULL, &g.sk_buf) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            return -1;
        }
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g.dev, g.sk_buf, &mr);
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
        if (vkAllocateMemory(g.dev, &mai, NULL, &g.sk_mem) != VK_SUCCESS ||
            vkBindBufferMemory(g.dev, g.sk_buf, g.sk_mem, 0) != VK_SUCCESS) {
            release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            return -1;
        }
        g.sk_buf_size = need;
    }

    uint32_t k_split = (K + seg - 1) / seg;
    cmd_begin();
    {
        VkDescriptorBufferInfo db[3] = { {bA, 0, ba}, {bB, 0, bb}, {g.sk_buf, 0, need} };
        VkWriteDescriptorSet wds[3];
        for (int i = 0; i < 3; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);
        struct { uint32_t K, N, ldb, k_split, pad; } pc = { K, N, ldb, k_split, 0 };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p1);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, Nt, 1, seg);
        VkBufferMemoryBarrier bmb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = g.sk_buf, .offset = 0, .size = need };
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bmb, 0, NULL);
    }
    {
        VkDescriptorBufferInfo db[2] = { {g.sk_buf, 0, need}, {bC, 0, bc} };
        VkWriteDescriptorSet wds[2];
        for (int i = 0; i < 2; i++) {
            wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = g.tdset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
        }
        vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);
        struct { uint32_t ne, k_num, N, ldc; float alpha, beta; } pc2 = {
            N, seg, N, ldc, alpha, beta };
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p2);
        vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.tdset, 0, NULL);
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
        vkCmdDispatch(g.cmd, (N / 2 + 255) / 256, 1, 1);
    }
    cmd_end_submit("matvec half split-k submit");

    release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
    return 0;
}

static int run_gemm(int variant, const void* A, VkBuffer bB_opt, const void* B, void* C,
                    size_t ba, size_t bb, size_t bc,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t lda, uint32_t ldb, uint32_t ldc,
                    uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                    float alpha, float beta) {
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    int bB_imported = 0;
    // batch>1 合并: import 覆盖全 batch (stride>0 才扩展; stride=0 = 广播, 单份即可)
    // 注意: 调用方传 ba/bb/bc 为单 batch 范围, 这里按 batch 扩展;
    //       bB_opt (转置产物) 时 bb 已是全 batch 范围, 不再扩展
    size_t ba_all = ba + (batch > 1 && stride_a > 0 ? (size_t)stride_a * (batch - 1) * 4 : 0);
    size_t bb_all = (bB_opt != VK_NULL_HANDLE) ? bb
                  : bb + (batch > 1 && stride_b > 0 ? (size_t)stride_b * (batch - 1) * 4 : 0);
    size_t bc_all = bc + (batch > 1 && stride_c > 0 ? (size_t)stride_c * (batch - 1) * 4 : 0);
    if (import_ptr(A, ba_all, &bA, &mA, &oA) != 0) return -1;
    if (bB_opt != VK_NULL_HANDLE) {
        bB = bB_opt; mB = VK_NULL_HANDLE; oB = 0;  // 转置产物, 直接复用
    } else {
        if (import_ptr(B, bb_all, &bB, &mB, &oB) != 0) {
            release_ptr(mA, bA);
            return -1;
        }
        bB_imported = 1;
    }
    if (import_ptr(C, bc_all, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA);
        if (bB_imported) release_ptr(mB, bB);
        return -1;
    }
    (void)oA; (void)oB; (void)oC;

    // VKBLAS_TILE128=1: fp32 GEMM 走 v7-128 tile (llama l_warptile 移植, 实验对比)
    // 128×128 tile 在 M,N≥256 时快 5-36%; 小 shape (单/少 wg) 用 v6 (64×64) 保持并行度
    int use128 = pick_tile128(M, N);

    // split-k (llama.cpp 借鉴): K>=2048 时补足 wave 占用; gfx803=36 CU:
    // tiles<=18: split_count=min(36/tiles,8); tiles 19..24: split_count=3;
    // 仅 fp32, 直通 f16/bf16 不做
    int rc = -1;
    uint32_t Mt = use128 ? (M + 127) / 128 : (M + 63) / 64;
    uint32_t Nt = use128 ? (N + 127) / 128 : (N + 63) / 64;
    if (batch == 1 && K >= 2048 && Mt * Nt <= 24 &&
        (use128 ? g.pipe_sk128[0] : g.pipe_sk[0]) != VK_NULL_HANDLE &&
        getenv("VKBLAS_NOSPLITK") == NULL) {
        uint32_t tiles = Mt * Nt;
        uint32_t split_k = tiles <= 18 ? 36 / tiles : 3;
        if (split_k > 8) split_k = 8;
        if (split_k > 1) {
            // 保证每段非空: k_split = ceil(K/split_k) → k_split*(split_k-1) < K 恒真
            rc = run_gemm_sk(use128, variant, bA, ba, bB, bb, bC, bc,
                             M, N, K, lda, ldb, ldc, alpha, beta, split_k);
        }
    }
    if (rc != 0) {
        if (use128)
            rc = run_gemm_vk128(variant, bA, ba_all, bB, bb_all, bC, bc_all,
                                M, N, K, lda, ldb, ldc, batch, stride_a, stride_b, stride_c, alpha, beta, 1);
        else
            rc = run_gemm_vk(variant, bA, ba_all, bB, bb_all, bC, bc_all,
                             M, N, K, lda, ldb, ldc, batch, stride_a, stride_b, stride_c, alpha, beta, 1);
    }

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
// submit==0: 记录到已打开的 command buffer (调用方负责 begin/end/submit)
static int cvt_run(int pipe, VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t bytes_out,
                   uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout, int submit,
                   int dslot) {
    if (g.cpipe[pipe] == VK_NULL_HANDLE) return -1;
    VkDescriptorBufferInfo db[2] = { {bIn, 0, bytes_in}, {bOut, 0, bytes_out} };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g.cdset[dslot], .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(g.dev, 2, wds, 0, NULL);

    struct { uint32_t R, C, ldin, ldout, roff; } pc = { R, C, ldin, ldout, 0 };
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.cpipe[pipe]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.tpl, 0, 1, &g.cdset[dslot], 0, NULL);
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
        if (submit) {
            vkEndCommandBuffer(g.cmd);
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
            vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cvt submit");
            vkQueueWaitIdle(g.queue);
        }
        return 0;
    }
    // 每块最多 ~2M 元素 (8MB/4B), 按行切
    uint32_t rows_per = (8u * 1024 * 1024) / (C * 4u);
    if (rows_per < 32) rows_per = 32;
    if (rows_per > 4096) rows_per = 4096;
    // 每行线程数 (每线程处理的元素数):
    //   pipe 0/1 (b2f, b2f_tsp) = 2 元素对 → (C+1)/2
    //   pipe 2 (f2b 列对)        = 2 列对   → (C+1)/2
    //   pipe 4/5 (h2f)           = 2 元素对 → (C+1)/2
    //   pipe 6 (f2h 列对)        = 2 列对   → (C+1)/2
    //   pipe 3/7 (f2b/f2h 线性 uint 对) 在独立分支处理
    uint32_t per_row = (C + 1) / 2;
    for (uint32_t ro = 0; ro < R; ro += rows_per) {
        pc.roff = ro;
        uint32_t rows = R - ro < rows_per ? R - ro : rows_per;
        uint32_t threads = rows * per_row;
        vkCmdPushConstants(g.cmd, g.tpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(g.cmd, (threads + 255) / 256, 1, 1);
    }
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cvt submit");
        vkQueueWaitIdle(g.queue);
    }
    return 0;
}

// bf16 计时 (VKBLAS_TRACE=1 时打印各阶段耗时)
static uint64_t now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// ---- batch 合并快路径 (llama.cpp mul_mm 思路: z 轴一次 dispatch 全 batch) ----
// 一次 import 全 batch (跨 hipMalloc 分配时 hsa 导出失败 → FALLBACK 回退逐 batch 循环)
// 返回: VKBLAS_OK 成功 / VKBLAS_ERR_FALLBACK 不可合并 (调用方回退循环) / 其他错误
static int gemm_f32_merged(int variant, vkblas_op_t op_b,
                           const char* pa, const char* pb, const char* pc,
                           size_t ba, size_t bb, size_t bc,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb, uint32_t ldc,
                           uint32_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                           float alpha, float beta) {
    // Import all three allocations once, then record transpose (if needed) and GEMM
    // into one command buffer. This is the batch analogue of the single-call
    // transpose+GEMM fusion and avoids the old duplicate imports in run_gemm().
    size_t ba_all = ba + (stride_a > 0 ? (size_t)stride_a * (batch - 1) * 4 : 0);
    size_t bb_all = bb + (stride_b > 0 ? (size_t)stride_b * (batch - 1) * 4 : 0);
    size_t bc_all = bc + (stride_c > 0 ? (size_t)stride_c * (batch - 1) * 4 : 0);
    VkBuffer bA, bB, bC;
    VkDeviceMemory mA, mB, mC;
    VkDeviceSize oA, oB, oC;
    if (import_ptr(pa, ba_all, &bA, &mA, &oA) != 0) {
        if (getenv("VKBLAS_TRACE")) fprintf(stderr, "[vk] merged: import A failed (fallback loop) ptr=%p size=%zu\n", pa, ba_all);
        return VKBLAS_ERR_FALLBACK;
    }
    if (import_ptr(pb, bb_all, &bB, &mB, &oB) != 0) {
        release_ptr(mA, bA);
        if (getenv("VKBLAS_TRACE")) fprintf(stderr, "[vk] merged: import B failed (fallback loop) ptr=%p size=%zu\n", pb, bb_all);
        return VKBLAS_ERR_FALLBACK;
    }
    if (import_ptr(pc, bc_all, &bC, &mC, &oC) != 0) {
        release_ptr(mA, bA); release_ptr(mB, bB);
        if (getenv("VKBLAS_TRACE")) fprintf(stderr, "[vk] merged: import C failed (fallback loop) ptr=%p size=%zu\n", pc, bc_all);
        return VKBLAS_ERR_FALLBACK;
    }
    (void)oA; (void)oB; (void)oC;

    int rc = 0;
    cmd_begin();
    int profile = getenv("VKBLAS_PROFILE") != NULL;
    uint64_t pt0 = profile ? now_us() : 0;
    if (op_b == VKBLAS_OP_N && g.tpipe != VK_NULL_HANDLE) {
        VkBuffer bBt = VK_NULL_HANDLE;
        uint32_t ldb_t = ldb;
        if (transpose_vk(bB, bb_all, K, N, ldb, &bBt, &ldb_t,
                         batch, stride_b, (int64_t)N * K, 0) != 0) {
            rc = VKBLAS_ERR_FALLBACK;
        } else {
            size_t bb_t = (size_t)batch * N * ldb_t * 4;
            cmd_barrier(bBt, bb_t);
            if (profile)
                fprintf(stderr, "[vkblas] prof: transpose rec took %.3fms\n", (now_us() - pt0) / 1e3);
            uint64_t pt1 = profile ? now_us() : 0;
            int use128 = pick_tile128(M, N);
            if (use128)
                rc = run_gemm_vk128(variant ^ 1, bA, ba_all, bBt, bb_t, bC, bc_all,
                                    M, N, K, lda, ldb_t, ldc, batch, stride_a,
                                    (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
            else
                rc = run_gemm_vk(variant ^ 1, bA, ba_all, bBt, bb_t, bC, bc_all,
                                 M, N, K, lda, ldb_t, ldc, batch, stride_a,
                                 (int64_t)N * ldb_t, stride_c, alpha, beta, 0);
        }
    } else {
        rc = run_gemm_vk(variant, bA, ba_all, bB, bb_all, bC, bc_all,
                         M, N, K, lda, ldb, ldc, batch, stride_a, stride_b,
                         stride_c, alpha, beta, 0);
    }
    if (rc == 0)
        cmd_end_submit("fused fp32 batch transpose+GEMM");
    else
        vkEndCommandBuffer(g.cmd);

    release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
    return rc == 0 ? VKBLAS_OK : VKBLAS_ERR_FALLBACK;
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

    // ---- M==1 decode 专用 matvec (免 B 全量转置; hipblas 翻译: M_vk=n) ----
    // 条件: op_a=N (A 单行直读), batch==1 (单 token), matvec shader 可用
    if (M == 1 && op_a == VKBLAS_OP_N && batch == 1 &&
        (g.mpipe[op_b] != VK_NULL_HANDLE || g.mpipe[2 + op_b] != VK_NULL_HANDLE)) {
        size_t ba_mv = (size_t)K * 4;  // A (1×K) 紧密
        size_t bb_mv = (op_b == VKBLAS_OP_T) ? (size_t)(N - 1) * ldb + K
                                             : (size_t)(K - 1) * ldb + N;
        size_t bc_mv = (size_t)N * 4;  // C (1×N) 紧密
        int mrc;
        // op_b==T && VKBLAS_CACHE_TRANSPOSE: 转置缓存 (推理权重不变) → TB=0 快路径
        if (op_b == VKBLAS_OP_T && getenv("VKBLAS_CACHE_TRANSPOSE") != NULL &&
            g.mpipe[0] != VK_NULL_HANDLE) {
            VkBuffer bBt; size_t bt_sz; int built;
            if (tc_get(B, N, K, ldb, &bBt, &bt_sz, &built) == 0)
                mrc = run_matvec_sk_vk(0, A, ba_mv, bBt, bt_sz, C, bc_mv,
                                       K, N, N, ldc, alpha, beta);
            else
                mrc = -1;  // 缓存构建失败 → 落回 TB=1 路径
        } else if (g.mpipe[op_b] != VK_NULL_HANDLE) {
            // 单 pass: wg 数 = ceil(N / (256*MV_COLS)); 不足 36 (3×12 CU) 时换 split-k
            const char* mcenv = getenv("VKBLAS_MV_COLS");
            long mvcols = mcenv && *mcenv ? strtol(mcenv, NULL, 10) : 4;
            if (mvcols < 1) mvcols = 1;
            if (mvcols > 8) mvcols = 8;
            uint32_t wg_mv = (N + 256 * (uint32_t)mvcols - 1) / (256 * (uint32_t)mvcols);
            if (wg_mv < 36 && g.mpipe[2 + op_b] != VK_NULL_HANDLE &&
                g.sk_reduce_pipe != VK_NULL_HANDLE)
                mrc = run_matvec_sk((int)op_b, A, ba_mv, B, bb_mv * 4, C, bc_mv,
                                    K, N, ldb, ldc, alpha, beta);
            else
                mrc = run_matvec((int)op_b, A, ba_mv, B, bb_mv * 4, C, bc_mv,
                                 K, N, ldb, ldc, alpha, beta);
        } else {
            mrc = run_matvec_sk((int)op_b, A, ba_mv, B, bb_mv * 4, C, bc_mv,
                                K, N, ldb, ldc, alpha, beta);
        }
        if (mrc == 0) {
            pthread_mutex_unlock(&g.lock);
            return VKBLAS_OK;
        }
        // import 失败 → 落回通用路径
    }

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

    // ---- batch>1 合并快路径: 一次 import + 一次 dispatch (z=batch), 省 (batch-1) 次固定开销 ----
    // RADV/gfx803 实测: 单次 dispatch 的 batch 维 >16 后 workgroup 调度退化为串行 (与轴无关)
    // → 拆批 ≤16 (每批独立指针偏移, stride 不变, shader 零改动)
    if (batch > 1 && stride_a >= 0 && stride_b >= 0 && stride_c >= 0) {
        uint32_t rem = batch;
        const char* pa2 = pa, *pb2 = pb, *pc2 = pc;
        int rc = 0, fallback = 0;
        while (rem > 0) {
            uint32_t nb = rem > 16 ? 16 : rem;
            rc = gemm_f32_merged(variant, op_b, pa2, pb2, pc2, ba, bb, bc,
                                 M, N, K, lda, ldb, ldc,
                                 nb, stride_a, stride_b, stride_c, alpha, beta);
            if (rc == VKBLAS_OK) {
                pa2 += (int64_t)nb * stride_a * 4;
                pb2 += (int64_t)nb * stride_b * 4;
                pc2 += (int64_t)nb * stride_c * 4;
                rem -= nb;
            } else {
                fallback = 1;
                break;
            }
        }
        if (!fallback) { pthread_mutex_unlock(&g.lock); return VKBLAS_OK; }
        if (rc != VKBLAS_ERR_FALLBACK) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        // FALLBACK → 落回逐 batch 循环
    }

    for (uint32_t i = 0; i < batch; i++) {
        int use128 = pick_tile128(M, N);
        uint32_t fMt = use128 ? (M + 127) / 128 : (M + 63) / 64;
        uint32_t fNt = use128 ? (N + 127) / 128 : (N + 63) / 64;
        int split_candidate = batch == 1 && K >= 2048 && fMt * fNt <= 24 &&
                              (use128 ? g.pipe_sk128[0] : g.pipe_sk[0]) != VK_NULL_HANDLE &&
                              getenv("VKBLAS_NOSPLITK") == NULL;
        if (op_b == VKBLAS_OP_N && g.tpipe != VK_NULL_HANDLE && !split_candidate) {
            VkBuffer bA2, bB2, bC2;
            VkDeviceMemory mA2, mB2, mC2;
            VkDeviceSize oA2, oB2, oC2;
            if (import_ptr(pa, ba, &bA2, &mA2, &oA2) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
            if (import_ptr(pb, bb, &bB2, &mB2, &oB2) != 0) {
                release_ptr(mA2, bA2);
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
            if (import_ptr(pc, bc, &bC2, &mC2, &oC2) != 0) {
                release_ptr(mA2, bA2); release_ptr(mB2, bB2);
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
            int frc = run_fused_f32_transpose_gemm(use128, variant, bA2, ba, bB2, bb,
                                                   bC2, bc, M, N, K, lda, ldb, ldc,
                                                   1, 0, 0, 0, alpha, beta);
            release_ptr(mA2, bA2); release_ptr(mB2, bB2); release_ptr(mC2, bC2);
            if (frc != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
        } else if (op_b == VKBLAS_OP_N && g.tpipe != VK_NULL_HANDLE) {
            // Keep split-k on the existing path; it owns the command buffer and reduce pass.
            VkBuffer bBt;
            uint32_t ldb_t;
            if (transpose_buf(pb, bb, K, N, ldb, &bBt, &ldb_t, 1, 0, 0) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
            size_t bb_t = (size_t)(N - 1) * ldb_t + K;
            bb_t *= 4;
            if (run_gemm(variant ^ 1, pa, bBt, NULL, pc, ba, bb_t, bc,
                         M, N, K, lda, ldb_t, ldc, 1, 0, 0, 0, alpha, beta) != 0) {
                pthread_mutex_unlock(&g.lock);
                return VKBLAS_ERR_IMPORT;
            }
        } else {
            if (run_gemm(variant, pa, VK_NULL_HANDLE, pb, pc, ba, bb, bc,
                         M, N, K, lda, ldb, ldc, 1, 0, 0, 0, alpha, beta) != 0) {
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

    // ---- M==1 decode 专用 matvec (免 B 转置; bf16 2B 元素) ----
    if (M == 1 && op_a == VKBLAS_OP_N && op_b == VKBLAS_OP_N && batch == 1 &&
        (ldc & 1u) == 0 && (ldb & 1u) == 0 && (N & 1u) == 0 &&
        g.mpipe[5] != VK_NULL_HANDLE) {
        size_t ba_h = (size_t)K * 2;
        size_t bb_h = ((size_t)(K - 1) * ldb + N) * 2;
        size_t bc_h = (size_t)N * 2;
        int hrc = run_matvec_sk_h(1, A, ba_h, B, bb_h, C, bc_h, K, N, ldb, ldc, alpha, beta);
        if (hrc == 0) {
            pthread_mutex_unlock(&g.lock);
            return VKBLAS_OK;
        }
        // import/资源失败 → 落回直通/回退路径
    }

    // ---- 直通快路径 (llama.cpp mul_mm 思路: 2B 半精度直进 LDS, 无 cvt 中间 buffer) ----
    // 条件: ldc 偶 && N 偶 (否则行尾/行首跨 uint 竞争); bf16 直通 pipeline 可用
    if ((ldc & 1u) == 0 && (N & 1u) == 0 && g.pipe_h[1][0] != VK_NULL_HANDLE) {
        int rc = gemm_h_direct(1, op_a, op_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                               batch, stride_a, stride_b, stride_c);
        pthread_mutex_unlock(&g.lock);
        return rc;
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
        // import bf16 输入 (3 个独立 dma-buf import, 仍需分次; 内部流水单提交)
        uint64_t t0 = trace ? now_us() : 0;
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 2, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pb, bb_e * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pc, bc_e * 2, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        // 内部流水单提交: cvt(bf16→fp32) → transpose(optional) → (optional cvt C) → GEMM → f2b
        int rc = 0;
        cmd_begin();
        rc |= cvt_run(0, bA, ba_e * 2, g.ibuf[0], ba2, Ra, Ca, lda, lda, 0, 0);
        uint64_t t1 = trace ? now_us() : 0;
        rc |= cvt_run(0, bB, bb_e * 2, g.ibuf[1], bb2, Rb, Cb, ldb, ldb, 0, 8);
        uint64_t t2 = trace ? now_us() : 0;
        if (rc == 0) cmd_barrier(g.ibuf[0], ba2), cmd_barrier(g.ibuf[1], bb2);
        VkBuffer bBt = VK_NULL_HANDLE;
        uint32_t ldb_eff = ldb;
        size_t bb2_used = bb2;
        if (rc == 0 && op_b == VKBLAS_OP_N) {
            if (transpose_vk(g.ibuf[1], bb2, K, N, ldb, &bBt, &ldb_eff, 1, 0, 0, 0) != 0) rc = -1;
            else { cmd_barrier(bBt, (size_t)N * K * 4); bb2_used = (size_t)N * K * 4; }
        }
        uint64_t t3 = trace ? now_us() : 0;
        if (rc == 0 && beta != 0.0f) {
            rc |= cvt_run(0, bC, bc_e * 2, g.ibuf[2], bc2, M, N, ldc, ldc, 0, 9);
            if (rc == 0) cmd_barrier(g.ibuf[2], bc2);
        }
        uint64_t t3b = trace ? now_us() : 0;
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        VkBuffer bB2 = bBt != VK_NULL_HANDLE ? bBt : g.ibuf[1];
        rc = run_gemm_vk(variant, g.ibuf[0], ba2, bB2, bb2_used, g.ibuf[2], bc2,
                         M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, alpha, beta, 0);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        cmd_barrier(g.ibuf[2], bc2);
        uint64_t t4 = trace ? now_us() : 0;
        rc = cvt_run((ldc & 1) ? 3 : 2, g.ibuf[2], bc2, bC, bc_e * 2, M, N, ldc, ldc, 0, (ldc & 1) ? 3 : 2);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        cmd_end_submit("fused bf16 fallback");
        uint64_t t5 = trace ? now_us() : 0;
        if (trace)
            fprintf(stderr, "[vk] bf16 fused: import %lu cvtAB %lu bTsp %lu cvtC %lu gemm %lu f2b %lu us (total %lu)\n",
                    (unsigned long)(t1 - t0), (unsigned long)(t2 - t1),
                    (unsigned long)(t3 - t2), (unsigned long)(t3b - t3),
                    (unsigned long)(t4 - t3b), (unsigned long)(t5 - t4),
                    (unsigned long)(t5 - t0));
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
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

    // ---- M==1 decode 专用 matvec (f16 2B 元素) ----
    if (M == 1 && op_a == VKBLAS_OP_N && op_b == VKBLAS_OP_N && batch == 1 &&
        (ldc & 1u) == 0 && (ldb & 1u) == 0 && (N & 1u) == 0 &&
        g.mpipe[4] != VK_NULL_HANDLE) {
        size_t ba_h = (size_t)K * 2;
        size_t bb_h = ((size_t)(K - 1) * ldb + N) * 2;
        size_t bc_h = (size_t)N * 2;
        int hrc = run_matvec_sk_h(0, A, ba_h, B, bb_h, C, bc_h, K, N, ldb, ldc, alpha, beta);
        if (hrc == 0) {
            pthread_mutex_unlock(&g.lock);
            return VKBLAS_OK;
        }
    }

    // ---- 直通快路径 (fp16): 条件同 bf16 (ldc 偶 && N 偶) ----
    if ((ldc & 1u) == 0 && (N & 1u) == 0 && g.pipe_h[0][0] != VK_NULL_HANDLE) {
        int rc = gemm_h_direct(0, op_a, op_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                               batch, stride_a, stride_b, stride_c);
        pthread_mutex_unlock(&g.lock);
        return rc;
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
        uint64_t t0 = trace ? now_us() : 0;
        VkBuffer bA, bB, bC; VkDeviceMemory mA, mB, mC; VkDeviceSize oA, oB, oC;
        if (import_ptr(pa, ba_e * 2, &bA, &mA, &oA) != 0) { pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pb, bb_e * 2, &bB, &mB, &oB) != 0) { release_ptr(mA, bA); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        if (import_ptr(pc, bc_e * 2, &bC, &mC, &oC) != 0) { release_ptr(mA, bA); release_ptr(mB, bB); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        int rc = 0;
        cmd_begin();
        rc |= cvt_run(4, bA, ba_e * 2, g.ibuf[0], ba2, Ra, Ca, lda, lda, 0, 4);
        uint64_t t1 = trace ? now_us() : 0;
        rc |= cvt_run(4, bB, bb_e * 2, g.ibuf[1], bb2, Rb, Cb, ldb, ldb, 0, 12);
        uint64_t t2 = trace ? now_us() : 0;
        if (rc == 0) cmd_barrier(g.ibuf[0], ba2), cmd_barrier(g.ibuf[1], bb2);
        VkBuffer bBt = VK_NULL_HANDLE;
        uint32_t ldb_eff = ldb;
        size_t bb2_used = bb2;
        if (rc == 0 && op_b == VKBLAS_OP_N) {
            if (transpose_vk(g.ibuf[1], bb2, K, N, ldb, &bBt, &ldb_eff, 1, 0, 0, 0) != 0) rc = -1;
            else { cmd_barrier(bBt, (size_t)N * K * 4); bb2_used = (size_t)N * K * 4; }
        }
        uint64_t t3 = trace ? now_us() : 0;
        if (rc == 0 && beta != 0.0f) {
            rc |= cvt_run(4, bC, bc_e * 2, g.ibuf[2], bc2, M, N, ldc, ldc, 0, 13);
            if (rc == 0) cmd_barrier(g.ibuf[2], bc2);
        }
        uint64_t t3b = trace ? now_us() : 0;
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }

        VkBuffer bB2 = bBt != VK_NULL_HANDLE ? bBt : g.ibuf[1];
        rc = run_gemm_vk(variant, g.ibuf[0], ba2, bB2, bb2_used, g.ibuf[2], bc2,
                         M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, alpha, beta, 0);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        cmd_barrier(g.ibuf[2], bc2);
        uint64_t t4 = trace ? now_us() : 0;

        rc = cvt_run((ldc & 1) ? 7 : 6, g.ibuf[2], bc2, bC, bc_e * 2, M, N, ldc, ldc, 0, (ldc & 1) ? 7 : 6);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return VKBLAS_ERR_IMPORT; }
        cmd_end_submit("fused f16 fallback");
        uint64_t t5 = trace ? now_us() : 0;
        if (trace)
            fprintf(stderr, "[vk] f16 fused: import %lu cvtAB %lu bTsp %lu cvtC %lu gemm %lu f2h %lu us (total %lu)\n",
                    (unsigned long)(t1 - t0), (unsigned long)(t2 - t1),
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
// submit==0: 记录到已打开的 command buffer (调用方负责 begin/end/submit)
static int cx_run(int pipe, VkBuffer bIn, size_t bytes_in,
                  VkBuffer bOutR, size_t bytes_out, VkBuffer bOutI,
                  uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
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
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cx submit");
        vkQueueWaitIdle(g.queue);
    }
    return 0;
}

// combine: C_out (complex) = alpha·(T1 + i·T2) + beta·C_old (complex)
// submit==0: 记录到已打开的 command buffer
static int cx_combine(VkBuffer bT1, size_t bT1_sz, VkBuffer bT2, size_t bT2_sz,
                      VkBuffer bCold, size_t bCold_sz, VkBuffer bCout, size_t bCout_sz,
                      uint32_t M, uint32_t N, uint32_t ldc,
                      float ar, float ai, float br, float bi, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
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
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cm submit");
        vkQueueWaitIdle(g.queue);
    }
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
        rc |= cx_run(0, bA, ba_e * 8, g.ibuf[0], ba_p, g.ibuf[1], Ra, Ca, lda, lda, 1);
        rc |= cx_run(0, bB, bb_e * 8, g.ibuf[2], bb_p, g.ibuf[3], Rb, Cb, ldb, ldb, 1);
        uint32_t ldb_eff = ldb;
        VkBuffer bBr = g.ibuf[2], bBi = g.ibuf[3];
        if (op_b == VKBLAS_OP_N) {
            // Br/Bi → BrT/BiT (N×K 紧密)
            rc |= transpose_into(g.ibuf[2], bb_p, g.ibuf[4], bt_p, K, N, ldb, &ldb_eff, 1, 0, 0, 1);
            rc |= transpose_into(g.ibuf[3], bb_p, g.ibuf[5], bt_p, K, N, ldb, &ldb_eff, 1, 0, 0, 1);
            bBr = g.ibuf[4]; bBi = g.ibuf[5];
        }
        if (rc != 0) { release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
            pthread_mutex_unlock(&g.lock); return 1; }

        // 4 次 fp32 GEMM (beta 项在 combine 处理, 这里全用 beta=0 或累加)
        // T1 = ArBr; T1 -= AiBi; T2 = ArBi; T2 += AiBr
        rc |= run_gemm_vk(variant, g.ibuf[0], ba_p, bBr, bt_p ? bt_p : bb_p,
                          g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, 1.0f, 0.0f, 1);
        rc |= run_gemm_vk(variant, g.ibuf[1], ba_p, bBi, bt_p ? bt_p : bb_p,
                          g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, -1.0f, 1.0f, 1);
        rc |= run_gemm_vk(variant, g.ibuf[0], ba_p, bBi, bt_p ? bt_p : bb_p,
                          g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, 1.0f, 0.0f, 1);
        rc |= run_gemm_vk(variant, g.ibuf[1], ba_p, bBr, bt_p ? bt_p : bb_p,
                          g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1, 0, 0, 0, 1.0f, 1.0f, 1);
        // combine: C = alpha·(T1+iT2) + beta·C_old (读 bC 旧值, 写回 bC)
        if (rc == 0)
            rc = cx_combine(g.ibuf[6], bc_p, g.ibuf[7], bc_p, bC, bc_e * 8, bC, bc_e * 8,
                            M, N, ldc, alpha_r, alpha_i, beta_r, beta_i, 1);
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return 1; }

        pa += stride_a * 8; pb += stride_b * 8; pc += stride_c * 8;
    }
    pthread_mutex_unlock(&g.lock);
    return 0;
}

// ---------- fp64 回退 ----------
// d64 转置: In (K×N, ldin) → Out (N×K 紧密, ldout=K), double
// submit==0: 记录到已打开的 command buffer
static int transpose_d64_vk(VkBuffer bIn, size_t bytes_in, VkBuffer bOut, size_t need,
                            uint32_t K, uint32_t N, uint32_t ldin, uint32_t* ldout, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
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
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "d64t submit");
        vkQueueWaitIdle(g.queue);
    }
    *ldout = K;
    return 0;
}

// fp64 GEMM 核心 dispatch: 对 Vulkan buffer 直接跑 (d64 pipeline, 3-binding, 56B push)
// 供 vkblas_gemm_f64 (外部指针) 与 vkblas_gemm_z64 (内部 buffer ×4) 复用
// submit==0: 记录到已打开的 command buffer
static int gemm_d64_dispatch(VkBuffer bA, size_t ba, VkBuffer bB, size_t bb,
                             VkBuffer bC, size_t bc,
                             uint32_t M, uint32_t N, uint32_t K,
                             uint32_t lda, uint32_t ldb, uint32_t ldc,
                             double alpha, double beta, int variant, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.d64pipe[variant]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.d64pl, 0, 1, &g.dset, 0, NULL);
    vkCmdPushConstants(g.cmd, g.d64pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(d64pc), &d64pc);
    vkCmdDispatch(g.cmd, d64pc.Mt, d64pc.Nt, 1);
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "d64 submit");
        vkQueueWaitIdle(g.queue);
    }
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
            int trc = transpose_d64_vk(bB, bb, g.ibuf[0], bt, K, N, ldb, &ldb_use, 1);
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
                                   M, N, K, lda, ldb_use, ldc, alpha, beta, variant, 1);
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
                  uint32_t R, uint32_t C, uint32_t ldin, uint32_t ldout, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
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
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cz submit");
        vkQueueWaitIdle(g.queue);
    }
    return 0;
}

// combine: C_out (complex128) = alpha·(T1 + i·T2) + beta·C_old, 全 double
// push: 4 uint + 4 double = 48B (double 8B 对齐于 16B 偏移 ✓)
static int cx_combine_d64(VkBuffer bT1, size_t bT1_sz, VkBuffer bT2, size_t bT2_sz,
                          VkBuffer bCold, size_t bCold_sz, VkBuffer bCout, size_t bCout_sz,
                          uint32_t M, uint32_t N, uint32_t ldc,
                          double ar, double ai, double br, double bi, int submit) {
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
    if (submit) {
        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(g.cmd, &cbbi);
    }
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
    if (submit) {
        vkEndCommandBuffer(g.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &g.cmd };
        vk_check(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE), "cm64 submit");
        vkQueueWaitIdle(g.queue);
    }
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
        cmd_begin();
        rc |= cz_run(bA, ba_e * 16, g.ibuf[0], ba_p, g.ibuf[1], Ra, Ca, lda, lda, 0);
        rc |= cz_run(bB, bb_e * 16, g.ibuf[2], bb_p, g.ibuf[3], Rb, Cb, ldb, ldb, 0);
        if (rc == 0) { cmd_barrier(g.ibuf[0], ba_p); cmd_barrier(g.ibuf[1], ba_p); cmd_barrier(g.ibuf[2], bb_p); cmd_barrier(g.ibuf[3], bb_p); }
        uint32_t ldb_eff = ldb;
        VkBuffer bBr = g.ibuf[2], bBi = g.ibuf[3];
        if (rc == 0 && op_b == VKBLAS_OP_N) {
            uint32_t ld_tmp = ldb;
            if (transpose_d64_vk(g.ibuf[2], bb_p, g.ibuf[4], bt_p, K, N, ldb, &ld_tmp, 0) != 0) rc = -1;
            else if (transpose_d64_vk(g.ibuf[3], bb_p, g.ibuf[5], bt_p, K, N, ldb, &ld_tmp, 0) != 0) rc = -1;
            else { cmd_barrier(g.ibuf[4], bt_p); cmd_barrier(g.ibuf[5], bt_p); ldb_eff = ld_tmp; bBr = g.ibuf[4]; bBi = g.ibuf[5]; }
        }
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return 1; }

        rc |= gemm_d64_dispatch(g.ibuf[0], ba_p, bBr, bt_p ? bt_p : bb_p,
                                g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 0.0, variant, 0);
        if (rc == 0) cmd_barrier(g.ibuf[6], bc_p);
        rc |= gemm_d64_dispatch(g.ibuf[1], ba_p, bBi, bt_p ? bt_p : bb_p,
                                g.ibuf[6], bc_p, M, N, K, lda, ldb_eff, ldc, -1.0, 1.0, variant, 0);
        if (rc == 0) cmd_barrier(g.ibuf[6], bc_p);
        rc |= gemm_d64_dispatch(g.ibuf[0], ba_p, bBi, bt_p ? bt_p : bb_p,
                                g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 0.0, variant, 0);
        if (rc == 0) cmd_barrier(g.ibuf[7], bc_p);
        rc |= gemm_d64_dispatch(g.ibuf[1], ba_p, bBr, bt_p ? bt_p : bb_p,
                                g.ibuf[7], bc_p, M, N, K, lda, ldb_eff, ldc, 1.0, 1.0, variant, 0);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return 1; }
        cmd_barrier(g.ibuf[6], bc_p); cmd_barrier(g.ibuf[7], bc_p);
        rc = cx_combine_d64(g.ibuf[6], bc_p, g.ibuf[7], bc_p, bC, bc_e * 16, bC, bc_e * 16,
                                M, N, ldc, alpha_r, alpha_i, beta_r, beta_i, 0);
        if (rc != 0) { vkEndCommandBuffer(g.cmd); release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC); pthread_mutex_unlock(&g.lock); return 1; }
        cmd_end_submit("fused z64");
        release_ptr(mA, bA); release_ptr(mB, bB); release_ptr(mC, bC);
        if (rc != 0) { pthread_mutex_unlock(&g.lock); return 1; }

        pa += stride_a * 16; pb += stride_b * 16; pc += stride_c * 16;
    }
    pthread_mutex_unlock(&g.lock);
    return 0;
}
