/* gpu_internal.h -- shared declarations for the Vulkan compute core.
 *
 * Ported from yujiteshima/mruby-gpu (src/gpu_internal.h). Trimmed to the
 * numeric-compute subset: no camera / face / display / matmul pipelines.
 */
#ifndef GPU_NARRAY_INTERNAL_H
#define GPU_NARRAY_INTERNAL_H

#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Pipeline / Layout enums ----
 *
 * 3BUF pipelines take (a, b, c): element-wise binary ops.
 * 2BUF pipelines take (a, b):    scalar ops and reduction.
 */
typedef enum {
  PIPE_ADD = 0, PIPE_SUB, PIPE_MUL, PIPE_DIV,  /* 3BUF: c = a (op) b   */
  PIPE_SCALE,                                  /* 2BUF: b = a * scalar */
  PIPE_ADDS,                                   /* 2BUF: b = a + scalar */
  PIPE_SUM,                                    /* 2BUF: partial sums   */
  PIPE_COUNT
} PipeId;

typedef enum { LAYOUT_3BUF = 0, LAYOUT_2BUF, LAYOUT_COUNT } LayoutId;

/* ---- GPU Context (singleton) ---- */
typedef struct {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorSetLayout desc_layouts[LAYOUT_COUNT];
  VkPipelineLayout pipe_layouts[LAYOUT_COUNT];
  VkPipeline pipelines[PIPE_COUNT];
  VkDescriptorPool desc_pool;
  int initialized;
} GpuCtx;

extern GpuCtx g_ctx;

/* ---- GPU Buffer (FP32, 1-D) ---- */
typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  uint32_t n;          /* element count */
  VkDeviceSize bytes;  /* n * sizeof(float) */
} GpuBuffer;

extern const struct mrb_data_type gpu_buffer_type;

/* ---- gpu_vulkan.c ---- */
void gpu_init(const char *shader_dir);
const char *gpu_pipe_name(PipeId pipe_id);
void dispatch_compute(PipeId pipe_id,
                      VkBuffer *buffers, VkDeviceSize *sizes, int num_buffers,
                      const void *push_data, uint32_t push_size,
                      uint32_t group_x, uint32_t group_y, uint32_t group_z);

/* ---- gpu_buffer.c ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n);
void       destroy_buffer(mrb_state *mrb, GpuBuffer *buf);  /* for un-wrapped scratch buffers */
mrb_value  wrap_buffer(mrb_state *mrb, struct RClass *klass, GpuBuffer *buf);
float     *map_buffer(GpuBuffer *buf);
void       unmap_buffer(GpuBuffer *buf);

#endif
