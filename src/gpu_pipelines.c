/* gpu_pipelines.c -- narray's built-in compute pipelines.
 *
 * The table and the shader loading used to sit in gpu_vulkan.c, next to the
 * Vulkan initialization. The rest of that file moved to mruby-gpu-runtime;
 * this is the part that is about narray's own shaders.
 */
#include "gpu_internal.h"

NArrayPipelines g_pipes = {0};

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
  gpu_ensure_initialized(mrb);
  if (g_pipes.loaded) return;

  for (int p = 0; p < PIPE_COUNT; p++) {
    char spv_path[512];
    snprintf(spv_path, sizeof(spv_path), "%s/%s.spv", shader_dir, pipe_names[p]);

    size_t spv_size;
    uint8_t *spv_code = load_spv(spv_path, &spv_size);
    if (!spv_code) {
      fprintf(stderr, "mruby-gpu-narray: could not load %s (run `make -C shader`)\n", spv_path);
      g_pipes.pipelines[p] = VK_NULL_HANDLE;
      g_pipes.state[p]     = PIPE_MISSING_SPV;
      continue;
    }

    /* A missing shader is not fatal -- only the operations that need it fail,
     * and ensure_pipeline reports why at that point. The distinction between
     * "no file" and "the driver rejected it" is recorded here because both
     * leave the handle VK_NULL_HANDLE but call for opposite fixes. */
    VkResult r = gpu_create_pipeline((const uint32_t *)spv_code, spv_size,
                                     pipe_to_layout[p], &g_pipes.pipelines[p]);
    free(spv_code);
    if (r == VK_SUCCESS) {
      g_pipes.state[p] = PIPE_READY;
    } else {
      g_pipes.state[p] = PIPE_CREATE_FAILED;
      g_pipes.error[p] = r;
    }
  }
  g_pipes.loaded = 1;
}

void narray_destroy_pipelines(void) {
  if (g_pipes.loaded && g_ctx.initialized) {
    for (int p = 0; p < PIPE_COUNT; p++) {
      if (g_pipes.pipelines[p] != VK_NULL_HANDLE) {
        vkDestroyPipeline(g_ctx.device, g_pipes.pipelines[p], NULL);
      }
    }
  }
  memset(&g_pipes, 0, sizeof(g_pipes));
}

void dispatch_compute(
    mrb_state *mrb, PipeId pipe_id,
    GpuBuffer **bufs, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  dispatch_pipeline(mrb, g_pipes.pipelines[pipe_id], pipe_to_layout[pipe_id],
                    bufs, num_buffers, push_data, push_size,
                    group_x, group_y, group_z);
}
