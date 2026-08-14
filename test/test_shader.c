// test_shader.c — 直接验证 shader 正确性 (不经过 dma-buf/hipblas)
// A/M 填充已知数据, 读回 C 与 CPU 期望对比
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkCommandPool cpool;
static VkCommandBuffer cmd;
static VkDescriptorPool dpool;
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkPipeline pipeline;
static VkDescriptorSet dset;

static void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) { fprintf(stderr, "FAIL %s: %d\n", what, r); exit(1); }
}

static void init_vk(void) {
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    check(vkCreateInstance(&ici, NULL, &inst), "instance");
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice* devs = malloc(n * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &n, devs);
    phys = devs[0];
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
    VkQueueFamilyProperties* qps = malloc(n * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qps);
    for (uint32_t i = 0; i < n; i++)
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &dqci };
    check(vkCreateDevice(phys, &dci, NULL, &dev), "device");
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam };
    check(vkCreateCommandPool(dev, &cpci, NULL, &cpool), "pool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    check(vkAllocateCommandBuffers(dev, &cbai, &cmd), "cmdbuf");
    VkDescriptorSetLayoutBinding bnds[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnds };
    check(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl), "dsl");
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 44 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    check(vkCreatePipelineLayout(dev, &plci, NULL, &pl), "pl");
    FILE* f = fopen("src/shaders/gemm_nn.spv", "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(sz); fread(code, 1, sz, f); fclose(f);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)sz, .pCode = code };
    VkShaderModule sm;
    check(vkCreateShaderModule(dev, &smci, NULL, &sm), "sm");
    VkComputePipelineCreateInfo cpci2 = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                   VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
        .layout = pl };
    check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci2, NULL, &pipeline), "pipe");
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    check(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool), "dpool");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    check(vkAllocateDescriptorSets(dev, &dsai, &dset), "dset");
}

static VkBuffer make_buf(VkDeviceSize size, VkDeviceMemory* mem) {
    VkBuffer b;
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    check(vkCreateBuffer(dev, &bci, NULL, &b), "buffer");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b, &mr);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size };
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(phys, &mprops);
    for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
        if (mr.memoryTypeBits & (1u << i) &&
            (mprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            mai.memoryTypeIndex = i; break;
        }
    }
    check(vkAllocateMemory(dev, &mai, NULL, mem), "alloc");
    check(vkBindBufferMemory(dev, b, *mem, 0), "bind");
    return b;
}

// 运行 shader 并返回 C
static void run_gemm(uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc,
                     const float* hA, const float* hB, float* hC_out) {
    VkDeviceMemory mA, mB, mC;
    VkBuffer bA = make_buf(lda * K * 4, &mA), bB = make_buf(ldb * N * 4, &mB), bC = make_buf(ldc * N * 4, &mC);
    void* p;
    vkMapMemory(dev, mA, 0, VK_WHOLE_SIZE, 0, &p); memcpy(p, hA, lda * K * 4); vkUnmapMemory(dev, mA);
    vkMapMemory(dev, mB, 0, VK_WHOLE_SIZE, 0, &p); memcpy(p, hB, ldb * N * 4); vkUnmapMemory(dev, mB);
    vkMapMemory(dev, mC, 0, VK_WHOLE_SIZE, 0, &p); memset(p, 0, ldc * N * 4); vkUnmapMemory(dev, mC);

    VkDescriptorBufferInfo db[3] = { {bA, 0, VK_WHOLE_SIZE}, {bB, 0, VK_WHOLE_SIZE}, {bC, 0, VK_WHOLE_SIZE} };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = dset, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db[i] };
    }
    vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);
    struct { uint32_t M, N, K, Mt, Nt, Kt, lda, ldb, ldc; float alpha, beta; } pc = {
        M, N, K, (M + 31) / 32, (N + 31) / 32, (K + 31) / 32, lda, ldb, ldc, 1.0f, 0.0f };
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, pc.Mt, pc.Nt, 1);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "submit");
    vkQueueWaitIdle(queue);
    vkMapMemory(dev, mC, 0, VK_WHOLE_SIZE, 0, &p);
    memcpy(hC_out, p, ldc * N * 4);
    vkUnmapMemory(dev, mC);
    vkFreeMemory(dev, mA, NULL); vkFreeMemory(dev, mB, NULL); vkFreeMemory(dev, mC, NULL);
    vkDestroyBuffer(dev, bA, NULL); vkDestroyBuffer(dev, bB, NULL); vkDestroyBuffer(dev, bC, NULL);
}

int main(void) {
    init_vk();
    // 小规模: M=4, N=6, K=5, 数据带 padding (lda=8, ldb=9, ldc=10)
    uint32_t M = 4, N = 6, K = 5, lda = 8, ldb = 9, ldc = 10;
    float hA[lda * K], hB[ldb * N], hC[ldc * N];
    memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
    for (int i = 0; i < (int)K; i++)
        for (int j = 0; j < (int)M; j++) hA[j * lda + i] = (float)((j + 1) * 10 + i);
    for (int i = 0; i < (int)N; i++)
        for (int j = 0; j < (int)K; j++) hB[j * ldb + i] = (float)((j + 1) * 100 + i);

    run_gemm(M, N, K, lda, ldb, ldc, hA, hB, hC);

    // 期望: C[m][n] = sum_k A[m][k]*B[k][n] (row-major 语义)
    double maxd = 0;
    int bad = 0;
    for (int m = 0; m < (int)M; m++) {
        for (int n = 0; n < (int)N; n++) {
            double s = 0;
            for (int k = 0; k < (int)K; k++) s += hA[m * lda + k] * hB[k * ldb + n];
            double d = fabs(hC[m * ldc + n] - s);
            if (d > maxd) maxd = d;
            if (d > 1e-3) {
                bad++;
                if (bad <= 5) fprintf(stderr, "C[%d][%d] = %.2f, expect %.2f\n", m, n, hC[m * ldc + n], s);
            }
        }
    }
    printf("small: maxdiff=%.2e, bad=%d/%d\n", maxd, bad, M * N);

    // NN 257 形状: M'=129, N'=257, K'=64 (模拟 hipblas NN 翻译后的 shader 参数)
    uint32_t M2 = 129, N2 = 257, K2 = 64, lda2 = 129, ldb2 = 257, ldc2 = 257;
    float *hA2 = calloc((size_t)lda2 * K2, 4), *hB2 = calloc((size_t)ldb2 * N2, 4);
    float *hC2 = calloc((size_t)ldc2 * N2, 4);
    srand(7);
    for (int i = 0; i < (int)lda2 * (int)K2; i++) hA2[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (int i = 0; i < (int)ldb2 * (int)N2; i++) hB2[i] = (float)(rand() % 200 - 100) / 10.0f;
    run_gemm(M2, N2, K2, lda2, ldb2, ldc2, hA2, hB2, hC2);
    bad = 0; maxd = 0;
    for (int m = 0; m < (int)M2; m++) {
        for (int n = 0; n < (int)N2; n++) {
            double s = 0;
            for (int k = 0; k < (int)K2; k++) s += hA2[m * lda2 + k] * hB2[k * ldb2 + n];
            double d = fabs(hC2[m * ldc2 + n] - s);
            if (d > maxd) maxd = d;
            if (d > 1e-3) bad++;
        }
    }
    printf("NN257: maxdiff=%.2e, bad=%d/%d\n", maxd, bad, M2 * N2);
    return (bad == 0) ? 0 : 1;
}
