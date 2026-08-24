// isa_dump.c — 从 RADV pipeline 提取 GFX ISA 二进制 (VK_KHR_pipeline_executable_properties)
// 用法: isa_dump <spv> <out.bin>   → 反汇编: llvm-objdump -d -m amdgcn --mcpu=gfx803 out.bin
// 输出 stats: VGPR/SGPR 数 (RADV LiveRegisterCount)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <spv> <out.bin>\n", argv[0]); return 1; }
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    // 检查扩展
    uint32_t ne; vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties* exts = malloc(sizeof(*exts) * ne);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, exts);
    int have = 0;
    for (uint32_t i = 0; i < ne; i++) if (!strcmp(exts[i].extensionName, "VK_KHR_pipeline_executable_properties")) have = 1;
    if (!have) { fprintf(stderr, "no VK_KHR_pipeline_executable_properties\n"); return 1; }
    uint32_t nq; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties* qps = malloc(sizeof(*qps) * nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qps);
    int q = -1;
    for (uint32_t i = 0; i < nq; i++) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { q = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = (uint32_t)q, .queueCount = 1, .pQueuePriorities = &prio };
    const char* extn[] = { "VK_KHR_pipeline_executable_properties" };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &dqci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = extn };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 48 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(sz); fread(code, 1, sz, f); fclose(f);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)sz, .pCode = code };
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL }, .layout = pl };
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    PFN_vkGetPipelineExecutablePropertiesKHR gpep = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gpes = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR gpeir = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    uint32_t n = 0;
    VkPipelineInfoKHR pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe };
    fprintf(stderr, "[gpep1]\n");
    gpep(dev, &pi, &n, NULL);
    fprintf(stderr, "[gpep2 n=%u]\n", n);
    VkPipelineExecutablePropertiesKHR* props = calloc(n, sizeof(*props));
    for (uint32_t i = 0; i < n; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    gpep(dev, &pi, &n, props);
    fprintf(stderr, "[gpep-done]\n");
    for (uint32_t i = 0; i < n; i++) printf("exec %u: %s\n", i, props[i].name);
    for (uint32_t i = 0; i < n; i++) {
        VkPipelineExecutableInfoKHR ei = { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipe, .executableIndex = i };
        (void)gpes; // RADV stats name 有段错误 bug, 跳过
        uint32_t ni = 0; fprintf(stderr, "[gpeir1]\n"); gpeir(dev, &ei, &ni, NULL);
        fprintf(stderr, "[gpeir2 ni=%u]\n", ni);
        VkPipelineExecutableInternalRepresentationKHR* reps = calloc(ni, sizeof(*reps));
        for (uint32_t j = 0; j < ni; j++) reps[j].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        gpeir(dev, &ei, &ni, reps);
        for (uint32_t j = 0; j < ni; j++) {
            printf("  rep %u: %s (%zu bytes)\n", j, reps[j].name, (size_t)reps[j].dataSize);
            if (reps[j].dataSize > 0) {
                void* data = malloc(reps[j].dataSize);
                reps[j].pData = data;
                gpeir(dev, &ei, &ni, reps);
                if (!strcmp(reps[j].name, "amdgcn")) {
                    FILE* o = fopen(argv[2], "wb");
                    if (o) { fwrite(data, 1, reps[j].dataSize, o); fclose(o); printf("  -> wrote %s\n", argv[2]); }
                }
            }
        }
    }
    return 0;
}