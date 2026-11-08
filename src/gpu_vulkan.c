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
  LAYOUT_2BUF, /* FFT_BITREV */
  LAYOUT_2BUF, /* CMAG       */
  LAYOUT_1BUF, /* FFT_STAGE  */
};

/* SPIR-V file basenames (loaded from <shader_dir>/<name>.spv). */
static const char *pipe_names[PIPE_COUNT] = {
  "add", "sub", "mul", "div", "scale", "adds", "sum",
  "fft_bitrev", "cmag", "fft_stage"
};

const char *gpu_pipe_name(PipeId pipe_id) {
  if (pipe_id < 0 || pipe_id >= PIPE_COUNT) return "?";
  return pipe_names[pipe_id];
}

/* ---- Error handling ----
 *
 * The Vulkan SDK ships vk_enum_string_helper.h for this, but relying on it
 * would add an SDK dependency to a gem whose only link dependency is the
 * loader, so the handful of codes that can actually reach us live here. */
const char *gpu_result_name(VkResult r) {
  switch (r) {
    case VK_SUCCESS:                      return "success";
    case VK_NOT_READY:                    return "not ready";
    case VK_TIMEOUT:                      return "timeout";
    case VK_INCOMPLETE:                   return "incomplete";
    case VK_ERROR_OUT_OF_HOST_MEMORY:     return "out of host memory";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:   return "out of device memory";
    case VK_ERROR_INITIALIZATION_FAILED:  return "initialization failed";
    case VK_ERROR_DEVICE_LOST:            return "device lost";
    case VK_ERROR_MEMORY_MAP_FAILED:      return "memory map failed";
    case VK_ERROR_LAYER_NOT_PRESENT:      return "layer not present";
    case VK_ERROR_EXTENSION_NOT_PRESENT:  return "extension not present";
    case VK_ERROR_FEATURE_NOT_PRESENT:    return "feature not present";
    case VK_ERROR_INCOMPATIBLE_DRIVER:    return "incompatible driver";
    case VK_ERROR_TOO_MANY_OBJECTS:       return "too many objects";
    case VK_ERROR_FRAGMENTED_POOL:        return "descriptor pool fragmented";
    case VK_ERROR_UNKNOWN:                return "unknown error";
    default:                              return "unrecognised VkResult";
  }
}

void gpu_check(mrb_state *mrb, VkResult r, const char *call) {
  if (r == VK_SUCCESS) return;
  mrb_raisef(mrb, E_RUNTIME_ERROR, "%s failed: %s (VkResult %d)",
             call, gpu_result_name(r), (int)r);
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

/* An extension that cannot be enumerated is treated as absent: the caller only
 * uses these to opt into optional behaviour, so failing closed is correct and
 * lets the Pi's plain path keep working. */
static int has_instance_ext(const char *name) {
  uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(NULL, &count, NULL) != VK_SUCCESS) return 0;
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  if (!props) return 0;
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
  if (vkEnumerateDeviceExtensionProperties(dev, NULL, &count, NULL) != VK_SUCCESS) return 0;
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  if (!props) return 0;
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

/* ---- Deferred submission ----
 *
 * A dispatch is no longer a round trip. It is recorded into one long-lived
 * command buffer, followed by a memory barrier so the next dispatch sees its
 * writes, and nothing is submitted until the host actually needs a result:
 * a host read or write of a buffer the batch touches (map_buffer), a
 * descriptor pool that has run dry, an explicit GPU.sync, or shutdown. Then
 * gpu_flush ends the command buffer, submits it once, waits once, and resets
 * the pool for the next batch.
 *
 * `a * 2 + 1 - 3 + 4` therefore costs four dispatches and one wait instead of
 * four waits; an FFT's log2(n) passes cost one wait instead of log2(n).
 * GPU.sync_mode = :eager flushes after every dispatch -- the pre-batching
 * behaviour, kept so the two can be measured against each other.
 *
 * A buffer remembers the epoch (batch number) it was last bound in. That is
 * how map_buffer knows whether a flush is due, and how the GC finalizer knows
 * a buffer must outlive its Ruby owner until the batch has run. */

static void batch_begin(mrb_state *mrb) {
  if (g_ctx.batch_recording) return;
  VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  /* The pool was created with RESET_COMMAND_BUFFER_BIT, so beginning a
   * command buffer that has already been submitted resets it implicitly. */
  VK_CHECK(mrb, vkBeginCommandBuffer(g_ctx.batch_cmd, &begin));
  g_ctx.batch_recording = 1;
}

/* One descriptor set from the pool. A full pool just means the batch is
 * flushed a little early: the sets it held are released by the reset. */
static VkDescriptorSet alloc_desc_set(mrb_state *mrb, LayoutId lid) {
  if (g_ctx.batch_sets >= GPU_MAX_DESC_SETS) gpu_flush(mrb);
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layouts[lid]
  };
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkResult r = vkAllocateDescriptorSets(g_ctx.device, &dsai, &set);
  if (r == VK_ERROR_FRAGMENTED_POOL
#ifdef VK_ERROR_OUT_OF_POOL_MEMORY
      || r == VK_ERROR_OUT_OF_POOL_MEMORY
#endif
  ) {
    /* Our count and the driver's disagree; a flush resets the pool either way. */
    gpu_flush(mrb);
    r = vkAllocateDescriptorSets(g_ctx.device, &dsai, &set);
  }
  gpu_check(mrb, r, "vkAllocateDescriptorSets");
  g_ctx.batch_sets++;
  return set;
}

void dispatch_pipeline(
    mrb_state *mrb, VkPipeline pipeline, LayoutId lid,
    GpuBuffer **bufs, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  /* Exceeding maxComputeWorkGroupCount is invalid usage, and what happens is
   * up to the driver: lavapipe runs the dispatch anyway and returns the right
   * answer, so the limit is easy to miss in testing, while a driver bounded by
   * a hardware register can clamp or fault instead. Reject it here rather than
   * let the answer depend on which GPU the code lands on. */
  if (group_x > g_ctx.max_workgroups) {
    /* Widened to mrb_int first: the element count overflows uint32_t on
     * devices that report a workgroup limit in the billions. */
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "array too large for one dispatch: %i workgroups needed, device allows %i "
      "(about %i elements)",
      (mrb_int)group_x, (mrb_int)g_ctx.max_workgroups,
      (mrb_int)g_ctx.max_workgroups * 256);
  }
  if (pipeline == VK_NULL_HANDLE) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "dispatch of a pipeline that was never created");
  }
  if (num_buffers < 1 || num_buffers > 3) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "dispatch takes 1..3 buffers, got %d", num_buffers);
  }

  VkDescriptorSet desc_set = alloc_desc_set(mrb, lid);

  VkDescriptorBufferInfo buf_infos[3];
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < num_buffers; i++) {
    buf_infos[i] = (VkDescriptorBufferInfo){bufs[i]->buffer, 0, bufs[i]->bytes};
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

  batch_begin(mrb);
  VkCommandBuffer cmd = g_ctx.batch_cmd;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layouts[lid], 0, 1, &desc_set, 0, NULL);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, g_ctx.pipe_layouts[lid], VK_SHADER_STAGE_COMPUTE_BIT,
      0, push_size, push_data);
  }
  vkCmdDispatch(cmd, group_x, group_y, group_z);

  /* Make this dispatch's writes visible to whatever is recorded next. One
   * global barrier per dispatch is coarser than tracking buffers one by one,
   * but it costs nothing next to the submit + fence wait it replaces. */
  VkMemoryBarrier barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
  };
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &barrier, 0, NULL, 0, NULL);

  for (int i = 0; i < num_buffers; i++) bufs[i]->epoch = g_ctx.epoch;
  g_ctx.batch_dispatches++;

  if (g_ctx.eager) gpu_flush(mrb);
}

void dispatch_compute(
    mrb_state *mrb, PipeId pipe_id,
    GpuBuffer **bufs, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  dispatch_pipeline(mrb, g_ctx.pipelines[pipe_id], pipe_to_layout[pipe_id],
                    bufs, num_buffers, push_data, push_size,
                    group_x, group_y, group_z);
}

/* Free what the GC could not: buffers whose owners died while a batch still
 * referenced them. Only called once no work is pending. */
static void bury_graveyard(mrb_state *mrb) {
  for (size_t i = 0; i < g_ctx.graveyard_len; i++) {
    GpuBuffer *b = g_ctx.graveyard[i];
    vkDestroyBuffer(g_ctx.device, b->buffer, NULL);
    vkFreeMemory(g_ctx.device, b->memory, NULL);
    mrb_free(mrb, b);
  }
  g_ctx.graveyard_len = 0;
}

void gpu_flush(mrb_state *mrb) {
  if (!g_ctx.initialized) return;
  VkResult r = VK_SUCCESS;
  const char *step = NULL;

  if (g_ctx.batch_recording) {
    step = "vkEndCommandBuffer";
    r = vkEndCommandBuffer(g_ctx.batch_cmd);
    if (r == VK_SUCCESS) {
      VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &g_ctx.batch_cmd
      };
      step = "vkQueueSubmit";
      r = vkQueueSubmit(g_ctx.queue, 1, &si, g_ctx.batch_fence);
    }
    if (r == VK_SUCCESS) {
      /* Without this check a lost device returns immediately and the caller
       * reads whatever is in the buffer as a valid result. */
      step = "vkWaitForFences";
      r = vkWaitForFences(g_ctx.device, 1, &g_ctx.batch_fence, VK_TRUE, UINT64_MAX);
      if (r == VK_SUCCESS) vkResetFences(g_ctx.device, 1, &g_ctx.batch_fence);
    }
    /* Whatever happened, this command buffer is spent; the next batch_begin
     * resets it. */
    g_ctx.batch_recording = 0;
  }

  /* No work is pending from here on (either it completed or it never left
   * the host), so the sets and the graveyard can go. */
  g_ctx.batch_dispatches = 0;
  if (g_ctx.batch_sets > 0) {
    vkResetDescriptorPool(g_ctx.device, g_ctx.desc_pool, 0);
    g_ctx.batch_sets = 0;
  }
  g_ctx.epoch++;
  bury_graveyard(mrb);

  if (step) gpu_check(mrb, r, step);
}

void gpu_sync_buffer(mrb_state *mrb, GpuBuffer *buf) {
  if (g_ctx.batch_recording && buf->epoch == g_ctx.epoch) gpu_flush(mrb);
}

/* ---- Init ---- */
void gpu_init(mrb_state *mrb, const char *shader_dir) {
  if (g_ctx.initialized) return;
  /* A half-built context cannot be reused, and retrying would leak everything
   * the failed attempt created -- once per operation, since initialization is
   * lazy. One attempt per process; fix the environment and restart. */
  if (g_ctx.init_failed) {
    mrb_raise(mrb, E_RUNTIME_ERROR,
      "GPU initialization already failed in this process and is not retried");
  }
  g_ctx.init_failed = 1;   /* cleared only on the success path at the end */

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
  VK_CHECK(mrb, vkCreateInstance(&inst_info, NULL, &g_ctx.instance));

  /* Physical Device: prefer a real GPU over a software rasterizer.
   * The Pi exposes both V3D (hardware) and llvmpipe (CPU); pick the first
   * non-CPU device so compute actually runs on the GPU. Falls back to the
   * first device if every device is CPU-type. */
  uint32_t dev_count = 0;
  VK_CHECK(mrb, vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, NULL));
  if (dev_count == 0) {
    mrb_raise(mrb, E_RUNTIME_ERROR,
      "no Vulkan physical device found (is a Vulkan driver installed?)");
  }
  VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * dev_count);
  if (!devs) mrb_raise(mrb, E_RUNTIME_ERROR, "out of memory enumerating devices");
  VkResult dev_r = vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, devs);
  if (dev_r != VK_SUCCESS) {
    free(devs);
    gpu_check(mrb, dev_r, "vkEnumeratePhysicalDevices");
  }
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

  /* Remember the dispatch limit, so an oversized array is reported rather
   * than silently truncated (see dispatch_compute). */
  VkPhysicalDeviceProperties dev_props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &dev_props);
  g_ctx.max_workgroups = dev_props.limits.maxComputeWorkGroupCount[0];

  /* Queue Family (compute) */
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = malloc(sizeof(VkQueueFamilyProperties) * qf_count);
  if (!qf_props) mrb_raise(mrb, E_RUNTIME_ERROR, "out of memory enumerating queue families");
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, qf_props);
  int have_compute = 0;
  g_ctx.queue_family = 0;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      g_ctx.queue_family = i;
      have_compute = 1;
      break;
    }
  }
  free(qf_props);
  if (!have_compute) {
    mrb_raisef(mrb, E_RUNTIME_ERROR,
      "device '%s' has no compute-capable queue family", dev_props.deviceName);
  }

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
  VK_CHECK(mrb, vkCreateDevice(g_ctx.physical_device, &dev_info, NULL, &g_ctx.device));
  vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family, 0, &g_ctx.queue);

  /* Command Pool */
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_ctx.queue_family
  };
  VK_CHECK(mrb, vkCreateCommandPool(g_ctx.device, &pool_info, NULL, &g_ctx.cmd_pool));

  /* The one command buffer every dispatch is recorded into, and the fence
   * each flush waits on. Both live as long as the context. */
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_ctx.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };
  VK_CHECK(mrb, vkAllocateCommandBuffers(g_ctx.device, &cbai, &g_ctx.batch_cmd));
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(mrb, vkCreateFence(g_ctx.device, &fi, NULL, &g_ctx.batch_fence));
  g_ctx.batch_recording = 0;
  g_ctx.batch_dispatches = 0;
  g_ctx.batch_sets = 0;
  g_ctx.epoch = 1;   /* a fresh buffer carries epoch 0, which never matches */

  /* Descriptor Set Layouts: 3, 2 and 1 storage buffers respectively */
  int buf_counts[LAYOUT_COUNT] = {3, 2, 1};
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
    VK_CHECK(mrb, vkCreateDescriptorSetLayout(g_ctx.device, &dl_info, NULL,
                                              &g_ctx.desc_layouts[l]));
  }

  /* Pipeline Layouts (one per descriptor layout, shared push constant range).
   * Every push block so far is 8 bytes ({ uint n; float scalar; },
   * { uint n; uint h; }, ...); 16 leaves headroom for the next one. */
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
    VK_CHECK(mrb, vkCreatePipelineLayout(g_ctx.device, &pl_info, NULL,
                                         &g_ctx.pipe_layouts[l]));
  }

  /* Load shaders and create pipelines */
  for (int p = 0; p < PIPE_COUNT; p++) {
    char spv_path[512];
    snprintf(spv_path, sizeof(spv_path), "%s/%s.spv", shader_dir, pipe_names[p]);

    size_t spv_size;
    uint8_t *spv_code = load_spv(spv_path, &spv_size);
    if (!spv_code) {
      fprintf(stderr, "mruby-gpu-narray: could not load %s (run `make -C shader`)\n", spv_path);
      g_ctx.pipelines[p]  = VK_NULL_HANDLE;
      g_ctx.pipe_state[p] = PIPE_MISSING_SPV;
      continue;
    }

    /* A missing shader is not fatal -- only the operations that need it fail,
     * and ensure_pipeline reports why at that point. The distinction between
     * "no file" and "the driver rejected it" is recorded here because both
     * leave the handle VK_NULL_HANDLE but call for opposite fixes. */
    VkShaderModuleCreateInfo sm_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = (uint32_t *)spv_code
    };
    VkShaderModule shader;
    VkResult r = vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader);
    free(spv_code);
    if (r != VK_SUCCESS) {
      g_ctx.pipelines[p]  = VK_NULL_HANDLE;
      g_ctx.pipe_state[p] = PIPE_CREATE_FAILED;
      g_ctx.pipe_error[p] = r;
      continue;
    }

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
    r = vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL,
                                 &g_ctx.pipelines[p]);
    vkDestroyShaderModule(g_ctx.device, shader, NULL);
    if (r == VK_SUCCESS) {
      g_ctx.pipe_state[p] = PIPE_READY;
    } else {
      g_ctx.pipelines[p]  = VK_NULL_HANDLE;
      g_ctx.pipe_state[p] = PIPE_CREATE_FAILED;
      g_ctx.pipe_error[p] = r;
    }
  }

  /* Descriptor Pool */
  VkDescriptorPoolSize pool_size = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 3 * GPU_MAX_DESC_SETS
  };
  VkDescriptorPoolCreateInfo dp_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .maxSets = GPU_MAX_DESC_SETS,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  VK_CHECK(mrb, vkCreateDescriptorPool(g_ctx.device, &dp_info, NULL, &g_ctx.desc_pool));

  g_ctx.initialized = 1;
  g_ctx.init_failed = 0;
}
