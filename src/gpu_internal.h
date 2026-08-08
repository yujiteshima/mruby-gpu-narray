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
 * 2BUF pipelines take (a, b):    scalar ops, reduction, and the FFT's
 *                                real->complex and complex->real passes.
 * 1BUF pipelines take (a):       the in-place FFT butterfly pass.
 */
typedef enum {
  PIPE_ADD = 0, PIPE_SUB, PIPE_MUL, PIPE_DIV,  /* 3BUF: c = a (op) b   */
  PIPE_SCALE,                                  /* 2BUF: b = a * scalar */
  PIPE_ADDS,                                   /* 2BUF: b = a + scalar */
  PIPE_SUM,                                    /* 2BUF: partial sums   */
  PIPE_FFT_BITREV,                             /* 2BUF: real -> bit-reversed complex */
  PIPE_CMAG,                                   /* 2BUF: complex -> magnitude / power */
  PIPE_FFT_STAGE,                              /* 1BUF: in-place butterfly pass */
  PIPE_COUNT
} PipeId;

typedef enum { LAYOUT_3BUF = 0, LAYOUT_2BUF, LAYOUT_1BUF, LAYOUT_COUNT } LayoutId;

/* Why a pipeline handle is missing. Without this, "the .spv file is absent"
 * and "the driver rejected the shader" both look like VK_NULL_HANDLE, and the
 * second one gets reported as the first -- telling the user to run a build
 * step that has already run. */
typedef enum {
  PIPE_READY = 0,     /* created and usable                         */
  PIPE_MISSING_SPV,   /* <name>.spv could not be read               */
  PIPE_CREATE_FAILED  /* the driver rejected it; see pipe_error[]    */
} PipeState;

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
  PipeState pipe_state[PIPE_COUNT];
  VkResult pipe_error[PIPE_COUNT];
  VkDescriptorPool desc_pool;
  uint32_t max_workgroups;  /* maxComputeWorkGroupCount[0] */
  int initialized;
  int init_failed;          /* a previous gpu_init raised; do not retry */
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

/* ---- Error handling ----
 *
 * Every Vulkan call that returns a VkResult goes through VK_CHECK, which turns
 * a failure into an mruby exception. Unchecked, a failed call leaves an
 * unusable handle behind and the next call either segfaults the VM or -- worse
 * -- returns whatever happened to be in the buffer as if it were a result.
 *
 * gpu_check raises, so it does not return on failure: anything the caller
 * allocated must be released *before* the call. */
void gpu_check(mrb_state *mrb, VkResult r, const char *call);
const char *gpu_result_name(VkResult r);
#define VK_CHECK(mrb, expr) gpu_check((mrb), (expr), #expr)

/* ---- gpu_vulkan.c ---- */
void gpu_init(mrb_state *mrb, const char *shader_dir);
const char *gpu_pipe_name(PipeId pipe_id);
void dispatch_compute(mrb_state *mrb, PipeId pipe_id,
                      VkBuffer *buffers, VkDeviceSize *sizes, int num_buffers,
                      const void *push_data, uint32_t push_size,
                      uint32_t group_x, uint32_t group_y, uint32_t group_z);

/* ---- gpu_buffer.c ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n);
void       destroy_buffer(mrb_state *mrb, GpuBuffer *buf);  /* GC finalizer + create_buffer unwind */
mrb_value  wrap_buffer(mrb_state *mrb, struct RClass *klass, GpuBuffer *buf);
float     *map_buffer(mrb_state *mrb, GpuBuffer *buf);
void       unmap_buffer(GpuBuffer *buf);

#endif
