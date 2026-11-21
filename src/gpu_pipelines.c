/* gpu_pipelines.c -- narray's built-in compute pipelines.
 *
 * The table and the shader loading used to sit in gpu_vulkan.c, next to the
 * Vulkan initialization. The rest of that file moved to mruby-gpu-runtime;
 * this is the part that is about narray's own shaders.
 */
#include "narray_internal.h"

NArrayPipelines g_pipes = {0};

/* What each shader binds: its storage buffers, and the bytes of its
 * push-constant block. */
static const gpu_rt_layout pipe_layouts[PIPE_COUNT] = {
  { .storage_buffers = 3, .push_constant_size = 4 },  /* ADD:   { uint n; }          */
  { .storage_buffers = 3, .push_constant_size = 4 },  /* SUB   */
  { .storage_buffers = 3, .push_constant_size = 4 },  /* MUL   */
  { .storage_buffers = 3, .push_constant_size = 4 },  /* DIV   */
  { .storage_buffers = 2, .push_constant_size = 8 },  /* SCALE: { uint n; float s; } */
  { .storage_buffers = 2, .push_constant_size = 8 },  /* ADDS  */
  { .storage_buffers = 2, .push_constant_size = 4 },  /* SUM:   { uint n; }          */
  { .storage_buffers = 2, .push_constant_size = 8 },  /* FFT_BITREV: { uint n, log2n; }      */
  { .storage_buffers = 2, .push_constant_size = 8 },  /* CMAG:       { uint out_n, square; } */
  { .storage_buffers = 1, .push_constant_size = 8 },  /* FFT_STAGE:  { uint n, h; }          */
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

void narray_init(mrb_state *mrb, const char *shader_dir) {
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  if (g_pipes.loaded) return;

  for (int p = 0; p < PIPE_COUNT; p++) {
    char spv_path[512];
    snprintf(spv_path, sizeof(spv_path), "%s/%s.spv", shader_dir, pipe_names[p]);

    size_t spv_size;
    uint8_t *spv_code = load_spv(spv_path, &spv_size);
    if (!spv_code) {
      fprintf(stderr, "mruby-gpu-narray: could not load %s (run `make -C shader`)\n", spv_path);
      g_pipes.pipelines[p] = NULL;
      g_pipes.state[p]     = PIPE_MISSING_SPV;
      continue;
    }

    /* A missing shader is not fatal -- only the operations that need it fail,
     * and ensure_pipeline reports why at that point. The distinction between
     * "no file" and "the driver rejected it" is recorded here because both
     * leave the handle NULL but call for opposite fixes. Passing an error
     * struct makes the runtime report a rejected shader instead of raising. */
    g_pipes.pipelines[p] = gpu_rt_pipeline_create(mrb, ctx, spv_code, spv_size,
                                                  &pipe_layouts[p], pipe_names[p],
                                                  &g_pipes.error[p]);
    free(spv_code);
    g_pipes.state[p] = g_pipes.pipelines[p] ? PIPE_READY : PIPE_CREATE_FAILED;
  }
  g_pipes.loaded = 1;
}

void narray_destroy_pipelines(mrb_state *mrb) {
  for (int p = 0; p < PIPE_COUNT; p++) {
    gpu_rt_pipeline_release(mrb, g_pipes.pipelines[p]);   /* NULL is ignored */
  }
  memset(&g_pipes, 0, sizeof(g_pipes));
}

void dispatch_compute(
    mrb_state *mrb, PipeId pipe_id,
    gpu_rt_buffer **bufs, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  gpu_rt_dispatch_desc d = {
    .pipeline = g_pipes.pipelines[pipe_id],
    .buffers = bufs, .buffer_count = (uint32_t)num_buffers,
    .push_constants = push_data, .push_constant_size = push_size,
    .groups = { group_x, group_y, group_z },
    .label = pipe_names[pipe_id],
  };
  gpu_rt_dispatch(mrb, &d);
}
