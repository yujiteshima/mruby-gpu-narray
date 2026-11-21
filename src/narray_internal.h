/* narray_internal.h -- declarations shared by mruby-gpu-narray's sources.
 *
 * The Vulkan core (device and queue, buffers and memory, pipelines, and the
 * deferred batch) is mruby-gpu-runtime, and this gem reaches it through its
 * C API, gpu_runtime.h. What is declared here is narray's own: the table of
 * built-in pipelines and the shaders behind them.
 *
 * This header is not for other gems. A gem that works on GPU::NArray includes
 * gpu_runtime.h itself and reads an array with gpu_rt_buffer_get().
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
 * Three buffers (a, b, c): element-wise binary ops.
 * Two buffers (a, b):      scalar ops, reduction, and the FFT's real->complex
 *                          and complex->real passes.
 * One buffer (a):          the in-place FFT butterfly pass.
 */
typedef enum {
  PIPE_ADD = 0, PIPE_SUB, PIPE_MUL, PIPE_DIV,  /* 3: c = a (op) b   */
  PIPE_SCALE,                                  /* 2: b = a * scalar */
  PIPE_ADDS,                                   /* 2: b = a + scalar */
  PIPE_SUM,                                    /* 2: partial sums   */
  PIPE_FFT_BITREV,                             /* 2: real -> bit-reversed complex */
  PIPE_CMAG,                                   /* 2: complex -> magnitude / power */
  PIPE_FFT_STAGE,                              /* 1: in-place butterfly pass */
  PIPE_COUNT
} PipeId;

/* Why a pipeline is missing. Without this, "the .spv file is absent" and
 * "the driver rejected the shader" both look like a NULL handle, and the
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
  gpu_rt_pipeline *pipelines[PIPE_COUNT];
  PipeState        state[PIPE_COUNT];
  gpu_rt_error     error[PIPE_COUNT];   /* filled when state is PIPE_CREATE_FAILED */
  int              loaded;
} NArrayPipelines;

extern NArrayPipelines g_pipes;

/* ---- gpu_pipelines.c ---- */

/* Start the runtime's context if needed, then create the built-in pipelines from
 * `shader_dir` unless they already exist. A shader that is missing or
 * rejected does not raise here; ensure_pipeline reports it when used. */
void narray_init(mrb_state *mrb, const char *shader_dir);

/* Give the built-in pipelines back to the runtime. Called from the gem's
 * final, before the runtime's final tears the device down. */
void narray_destroy_pipelines(mrb_state *mrb);

const char *gpu_pipe_name(PipeId pipe_id);

/* Record one dispatch of a built-in pipeline into the runtime's batch,
 * labelled with the pipeline's name. */
void dispatch_compute(mrb_state *mrb, PipeId pipe_id,
                      gpu_rt_buffer **bufs, int num_buffers,
                      const void *push_data, uint32_t push_size,
                      uint32_t group_x, uint32_t group_y, uint32_t group_z);

#endif
