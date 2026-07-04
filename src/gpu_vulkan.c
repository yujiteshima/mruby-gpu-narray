/* gpu_vulkan.c -- Vulkan initialization and generic compute dispatch.
 *
 * Ported from yujiteshima/mruby-gpu (src/gpu_vulkan.c). The dispatch and
 * initialization logic is unchanged; only the pipeline/layout tables differ
 * (numeric-compute pipelines instead of the image/ML ones).
 */
#include "gpu_internal.h"

GpuCtx g_ctx = {0};

/* Which descriptor-set layout each pipeline uses. */
static const LayoutId pipe_to_layout[PIPE_COUNT] = {
  LAYOUT_3BUF, /* ADD   */
  LAYOUT_3BUF, /* SUB   */
  LAYOUT_3BUF, /* MUL   */
  LAYOUT_3BUF, /* DIV   */
  LAYOUT_2BUF, /* SCALE */
  LAYOUT_2BUF, /* ADDS  */
  LAYOUT_2BUF, /* SUM   */
};

/* SPIR-V file basenames (loaded from <shader_dir>/<name>.spv). */
static const char *pipe_names[PIPE_COUNT] = {
  "add", "sub", "mul", "div", "scale", "adds", "sum"
};

const char *gpu_pipe_name(PipeId pipe_id) {
  if (pipe_id < 0 || pipe_id >= PIPE_COUNT) return "?";
  return pipe_names[pipe_id];
}

/* Portability (MoltenVK on macOS) support. Absent on the Raspberry Pi's native
 * V3D driver, so everything below is gated on runtime extension detection and
 * has no effect there. Provide fallback macros for older SDK headers. */
#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif
#define GPU_PORTABILITY_SUBSET_EXT "VK_KHR_portability_subset"

static int has_instance_ext(const char *name) {
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(NULL, &count, NULL);
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  vkEnumerateInstanceExtensionProperties(NULL, &count, props);
  int found = 0;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
  }
  free(props);
  return found;
}

static int has_device_ext(VkPhysicalDevice dev, const char *name) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(dev, NULL, &count, NULL);
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  vkEnumerateDeviceExtensionProperties(dev, NULL, &count, props);
  int found = 0;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
  }
  free(props);
  return found;
}

/* ---- Helpers ---- */
static uint8_t *load_spv(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *size = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(*size);
  if (buf) {
    if (fread(buf, 1, *size, f) != *size) { free(buf); buf = NULL; }
  }
  fclose(f);
  return buf;
}

/* ---- Generic Compute Dispatch ---- */
void dispatch_compute(
    PipeId pipe_id,
    VkBuffer *buffers, VkDeviceSize *sizes, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  LayoutId lid = pipe_to_layout[pipe_id];

  /* Allocate descriptor set */
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layouts[lid]
  };
  VkDescriptorSet desc_set;
  vkAllocateDescriptorSets(g_ctx.device, &dsai, &desc_set);

  /* Update descriptor set */
  VkDescriptorBufferInfo buf_infos[3];
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < num_buffers; i++) {
    buf_infos[i] = (VkDescriptorBufferInfo){buffers[i], 0, sizes[i]};
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buf_infos[i]
    };
  }
  vkUpdateDescriptorSets(g_ctx.device, num_buffers, writes, 0, NULL);

  /* Command buffer */
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_ctx.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(g_ctx.device, &cbai, &cmd);

  VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  vkBeginCommandBuffer(cmd, &begin);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.pipelines[pipe_id]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layouts[lid], 0, 1, &desc_set, 0, NULL);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, g_ctx.pipe_layouts[lid], VK_SHADER_STAGE_COMPUTE_BIT,
      0, push_size, push_data);
  }
  vkCmdDispatch(cmd, group_x, group_y, group_z);
  vkEndCommandBuffer(cmd);

  /* Submit and wait */
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(g_ctx.device, &fi, NULL, &fence);

  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd
  };
  vkQueueSubmit(g_ctx.queue, 1, &si, fence);
  vkWaitForFences(g_ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(g_ctx.device, fence, NULL);
  vkFreeCommandBuffers(g_ctx.device, g_ctx.cmd_pool, 1, &cmd);
  vkFreeDescriptorSets(g_ctx.device, g_ctx.desc_pool, 1, &desc_set);
}

/* ---- Init ---- */
void gpu_init(const char *shader_dir) {
  if (g_ctx.initialized) return;

  /* Instance */
  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "mruby-gpu-narray",
    .apiVersion = VK_API_VERSION_1_1
  };
  /* Opt into portability enumeration if the loader exposes it (macOS/MoltenVK).
   * On the Pi this extension is absent, so we take the plain path. */
  const char *inst_exts[1];
  uint32_t inst_ext_count = 0;
  VkInstanceCreateFlags inst_flags = 0;
  if (has_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    inst_exts[inst_ext_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    inst_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .flags = inst_flags,
    .enabledExtensionCount = inst_ext_count,
    .ppEnabledExtensionNames = inst_ext_count ? inst_exts : NULL
  };
  vkCreateInstance(&inst_info, NULL, &g_ctx.instance);

  /* Physical Device: prefer a real GPU over a software rasterizer.
   * The Pi exposes both V3D (hardware) and llvmpipe (CPU); pick the first
   * non-CPU device so compute actually runs on the GPU. Falls back to the
   * first device if every device is CPU-type. */
  uint32_t dev_count = 0;
  vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, NULL);
  if (dev_count == 0) {
    fprintf(stderr, "mruby-gpu-narray: no Vulkan physical device found\n");
    return;
  }
  VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * dev_count);
  vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, devs);
  g_ctx.physical_device = devs[0];
  for (uint32_t i = 0; i < dev_count; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
      g_ctx.physical_device = devs[i];
      break;
    }
  }
  free(devs);

  /* Queue Family (compute) */
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = malloc(sizeof(VkQueueFamilyProperties) * qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, qf_props);
  g_ctx.queue_family = 0;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      g_ctx.queue_family = i;
      break;
    }
  }
  free(qf_props);

  /* Device + Queue */
  float priority = 1.0f;
  VkDeviceQueueCreateInfo q_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_ctx.queue_family,
    .queueCount = 1,
    .pQueuePriorities = &priority
  };
  /* If the device advertises VK_KHR_portability_subset it MUST be enabled
   * (Vulkan spec). Present on MoltenVK, absent on the Pi's V3D. */
  const char *dev_exts[1];
  uint32_t dev_ext_count = 0;
  if (has_device_ext(g_ctx.physical_device, GPU_PORTABILITY_SUBSET_EXT)) {
    dev_exts[dev_ext_count++] = GPU_PORTABILITY_SUBSET_EXT;
  }
  VkDeviceCreateInfo dev_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &q_info,
    .enabledExtensionCount = dev_ext_count,
    .ppEnabledExtensionNames = dev_ext_count ? dev_exts : NULL
  };
  vkCreateDevice(g_ctx.physical_device, &dev_info, NULL, &g_ctx.device);
  vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family, 0, &g_ctx.queue);

  /* Command Pool */
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_ctx.queue_family
  };
  vkCreateCommandPool(g_ctx.device, &pool_info, NULL, &g_ctx.cmd_pool);

  /* Descriptor Set Layouts: LAYOUT_3BUF has 3 storage buffers, LAYOUT_2BUF has 2 */
  int buf_counts[LAYOUT_COUNT] = {3, 2};
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < buf_counts[l]; i++) {
      bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      };
    }
    VkDescriptorSetLayoutCreateInfo dl_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = buf_counts[l],
      .pBindings = bindings
    };
    vkCreateDescriptorSetLayout(g_ctx.device, &dl_info, NULL, &g_ctx.desc_layouts[l]);
  }

  /* Pipeline Layouts (one per descriptor layout, shared push constant range).
   * 8 bytes is enough for the largest push block: { uint n; float scalar; }. */
  VkPushConstantRange push_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = 16
  };
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkPipelineLayoutCreateInfo pl_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &g_ctx.desc_layouts[l],
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_range
    };
    vkCreatePipelineLayout(g_ctx.device, &pl_info, NULL, &g_ctx.pipe_layouts[l]);
  }

  /* Load shaders and create pipelines */
  for (int p = 0; p < PIPE_COUNT; p++) {
    char spv_path[512];
    snprintf(spv_path, sizeof(spv_path), "%s/%s.spv", shader_dir, pipe_names[p]);

    size_t spv_size;
    uint8_t *spv_code = load_spv(spv_path, &spv_size);
    if (!spv_code) {
      fprintf(stderr, "mruby-gpu-narray: could not load %s (run `make -C shader`)\n", spv_path);
      g_ctx.pipelines[p] = VK_NULL_HANDLE;
      continue;
    }

    VkShaderModuleCreateInfo sm_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = (uint32_t *)spv_code
    };
    VkShaderModule shader;
    vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader);
    free(spv_code);

    LayoutId lid = pipe_to_layout[p];
    VkComputePipelineCreateInfo cp_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main"
      },
      .layout = g_ctx.pipe_layouts[lid]
    };
    vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL, &g_ctx.pipelines[p]);
    vkDestroyShaderModule(g_ctx.device, shader, NULL);
  }

  /* Descriptor Pool */
  VkDescriptorPoolSize pool_size = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 768
  };
  VkDescriptorPoolCreateInfo dp_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .maxSets = 256,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  vkCreateDescriptorPool(g_ctx.device, &dp_info, NULL, &g_ctx.desc_pool);

  g_ctx.initialized = 1;
}
