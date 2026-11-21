/* gpu_internal.h -- declarations shared by mruby-gpu-narray's sources.
 *
 * The Vulkan core (device and queue, buffers and memory, the descriptor
 * layouts, and the deferred batch) lives in mruby-gpu-runtime, whose
 * gpu_runtime.h declares it. What is left here is narray's own: the table of
 * built-in pipelines and the shaders behind them.
 *
 * mruby-gpu-kernel still includes this header to reach the core, which keeps
 * it building until it moves to gpu_runtime.h itself
 * (yujiteshima/mruby-gpu-kernel#7).
 */
#ifndef GPU_NARRAY_INTERNAL_H
#define GPU_NARRAY_INTERNAL_H

#include <gpu_runtime.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Built-in pipelines ----
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

/* Why a pipeline handle is missing. Without this, "the .spv file is absent"
 * and "the driver rejected the shader" both look like VK_NULL_HANDLE, and the
 * second one gets reported as the first -- telling the user to run a build
 * step that has already run. */
typedef enum {
  PIPE_READY = 0,     /* created and usable                         */
  PIPE_MISSING_SPV,   /* <name>.spv could not be read               */
  PIPE_CREATE_FAILED  /* the driver rejected it; see error[]         */
} PipeState;

/* The built-in pipelines, created together from <shader_dir>/<name>.spv the
 * first time an operation needs the GPU. */
typedef struct {
  VkPipeline pipelines[PIPE_COUNT];
  PipeState  state[PIPE_COUNT];
  VkResult   error[PIPE_COUNT];
  int        loaded;
} NArrayPipelines;

extern NArrayPipelines g_pipes;

/* ---- gpu_pipelines.c ---- */

/* Initialize the runtime if needed, then create the built-in pipelines from
 * `shader_dir` unless they already exist. A shader that is missing or
 * rejected does not raise here; ensure_pipeline reports it when used. */
void narray_init(mrb_state *mrb, const char *shader_dir);

/* Destroy the built-in pipelines. Called from the gem's final, before the
 * runtime's final tears the device down. */
void narray_destroy_pipelines(void);

const char *gpu_pipe_name(PipeId pipe_id);

/* Record one dispatch of a built-in pipeline into the runtime's batch. */
void dispatch_compute(mrb_state *mrb, PipeId pipe_id,
                      GpuBuffer **bufs, int num_buffers,
                      const void *push_data, uint32_t push_size,
                      uint32_t group_x, uint32_t group_y, uint32_t group_z);

#endif
