/* gpu_narray.c -- Numo-like N-dimensional array on top of the Vulkan core.
 *
 * Classes:
 *   GPU::NArray  -- abstract base (holds all shared instance methods)
 *   GPU::SFloat  -- FP32, 1-D. The only dtype for now.
 *
 * The data always lives in a GPU (VkBuffer-backed) GpuBuffer. It is copied to
 * the host only on #to_a / #head. Arithmetic runs as compute dispatches.
 */
#include "gpu_internal.h"
#include <mruby/hash.h>
#include <mruby/numeric.h>

/* The shader directory is embedded by mrbgem.rake into a generated header so
 * that GPU.init is optional. When editing outside a build (no header yet),
 * fall back to the relative "shader" dir so the file still compiles. */
#if defined(__has_include)
#  if __has_include("shader_dir.h")
#    include "shader_dir.h"
#  endif
#endif
#ifndef GPU_NARRAY_SHADER_DIR
#  define GPU_NARRAY_SHADER_DIR "shader"
#endif

/* ---- lazy initialization ---- */
static void ensure_initialized(mrb_state *mrb) {
  (void)mrb;
  if (!g_ctx.initialized) gpu_init(GPU_NARRAY_SHADER_DIR);
}

static void ensure_pipeline(mrb_state *mrb, PipeId pipe) {
  ensure_initialized(mrb);
  if (g_ctx.pipelines[pipe] == VK_NULL_HANDLE) {
    mrb_raisef(mrb, E_RUNTIME_ERROR,
      "shader '%s.spv' is not compiled. Run `make -C shader` first.",
      gpu_pipe_name(pipe));
  }
}

/* Return the GpuBuffer if v is one of our arrays, else NULL (no raise). */
static GpuBuffer *as_narray(mrb_value v) {
  if (mrb_type(v) == MRB_TT_CDATA && DATA_TYPE(v) == &gpu_buffer_type) {
    return (GpuBuffer *)DATA_PTR(v);
  }
  return NULL;
}

/* =========================================================================
 * Class methods: GPU::SFloat.new / GPU::SFloat.cast
 * ========================================================================= */

/* GPU::SFloat.new(n) -> uninitialized array of n floats (like Numo::SFloat.new) */
static mrb_value sfloat_s_new(mrb_state *mrb, mrb_value self) {
  mrb_int n;
  mrb_get_args(mrb, "i", &n);
  if (n < 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "negative array size");
  ensure_initialized(mrb);
  GpuBuffer *buf = create_buffer(mrb, (uint32_t)n);
  return wrap_buffer(mrb, mrb_class_ptr(self), buf);
}

/* GPU::SFloat.cast([...]) -> array copied from a Ruby Array (ints or floats) */
static mrb_value sfloat_s_cast(mrb_state *mrb, mrb_value self) {
  mrb_value ary;
  mrb_get_args(mrb, "A", &ary);
  ensure_initialized(mrb);

  mrb_int n = RARRAY_LEN(ary);
  GpuBuffer *buf = create_buffer(mrb, (uint32_t)n);
  float *m = map_buffer(buf);
  for (mrb_int i = 0; i < n; i++) {
    m[i] = (float)mrb_as_float(mrb, mrb_ary_ref(mrb, ary, i));
  }
  unmap_buffer(buf);
  return wrap_buffer(mrb, mrb_class_ptr(self), buf);
}

/* =========================================================================
 * Instance methods: host transfer
 * ========================================================================= */

/* #to_a -> Ruby Array of every element (host copy) */
static mrb_value narray_to_a(mrb_state *mrb, mrb_value self) {
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  float *m = map_buffer(buf);
  mrb_value ary = mrb_ary_new_capa(mrb, buf->n);
  for (uint32_t i = 0; i < buf->n; i++) {
    mrb_ary_push(mrb, ary, mrb_float_value(mrb, (mrb_float)m[i]));
  }
  unmap_buffer(buf);
  return ary;
}

/* #head(k) -> Ruby Array of the first k elements (debugging aid) */
static mrb_value narray_head(mrb_state *mrb, mrb_value self) {
  mrb_int k;
  mrb_get_args(mrb, "i", &k);
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  if (k < 0) k = 0;
  if ((uint32_t)k > buf->n) k = buf->n;
  float *m = map_buffer(buf);
  mrb_value ary = mrb_ary_new_capa(mrb, k);
  for (mrb_int i = 0; i < k; i++) {
    mrb_ary_push(mrb, ary, mrb_float_value(mrb, (mrb_float)m[i]));
  }
  unmap_buffer(buf);
  return ary;
}

/* #size / #length -> element count */
static mrb_value narray_size(mrb_state *mrb, mrb_value self) {
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  return mrb_fixnum_value(buf->n);
}

/* #seq(start=0, step=1) -> self, filled with start, start+step, ... (host write) */
static mrb_value narray_seq(mrb_state *mrb, mrb_value self) {
  mrb_float start = 0.0, step = 1.0;
  mrb_get_args(mrb, "|ff", &start, &step);
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  float *m = map_buffer(buf);
  for (uint32_t i = 0; i < buf->n; i++) {
    m[i] = (float)(start + step * (double)i);
  }
  unmap_buffer(buf);
  return self;
}

/* #fill(value) -> self, every element set to value (host write) */
static mrb_value narray_fill(mrb_state *mrb, mrb_value self) {
  mrb_float v;
  mrb_get_args(mrb, "f", &v);
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  float *m = map_buffer(buf);
  for (uint32_t i = 0; i < buf->n; i++) m[i] = (float)v;
  unmap_buffer(buf);
  return self;
}

/* =========================================================================
 * Instance methods: arithmetic (runs on GPU)
 * ========================================================================= */

/* element-wise binary op with another array: c = a (op) b */
static mrb_value binop_nn(mrb_state *mrb, mrb_value self, GpuBuffer *rhs, PipeId pipe) {
  GpuBuffer *a = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  if (a->n != rhs->n) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "shape mismatch: %d vs %d",
               (int)a->n, (int)rhs->n);
  }
  ensure_pipeline(mrb, pipe);
  GpuBuffer *c = create_buffer(mrb, a->n);
  VkBuffer bufs[3]      = {a->buffer, rhs->buffer, c->buffer};
  VkDeviceSize sizes[3] = {a->bytes,  rhs->bytes,  c->bytes};
  uint32_t push = a->n;
  dispatch_compute(pipe, bufs, sizes, 3, &push, sizeof(uint32_t),
                   (a->n + 255) / 256, 1, 1);
  return wrap_buffer(mrb, mrb_obj_class(mrb, self), c);
}

/* scalar op: PIPE_SCALE gives b = a * scalar, PIPE_ADDS gives b = a + scalar */
static mrb_value scalar_op(mrb_state *mrb, mrb_value self, float scalar, PipeId pipe) {
  GpuBuffer *a = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  ensure_pipeline(mrb, pipe);
  GpuBuffer *b = create_buffer(mrb, a->n);
  VkBuffer bufs[2]      = {a->buffer, b->buffer};
  VkDeviceSize sizes[2] = {a->bytes,  b->bytes};
  struct { uint32_t n; float s; } push = {a->n, scalar};
  dispatch_compute(pipe, bufs, sizes, 2, &push, sizeof(push),
                   (a->n + 255) / 256, 1, 1);
  return wrap_buffer(mrb, mrb_obj_class(mrb, self), b);
}

static mrb_value type_err(mrb_state *mrb, mrb_value o, const char *op) {
  mrb_raisef(mrb, E_TYPE_ERROR,
    "GPU::NArray#%s expects an NArray or a Numeric on the right "
    "(write `narray %s scalar`, not `scalar %s narray`); got %s",
    op, op, op, mrb_obj_classname(mrb, o));
  return mrb_nil_value(); /* unreachable */
}

static mrb_value narray_add(mrb_state *mrb, mrb_value self) {
  mrb_value o; mrb_get_args(mrb, "o", &o);
  GpuBuffer *r = as_narray(o);
  if (r) return binop_nn(mrb, self, r, PIPE_ADD);
  if (mrb_integer_p(o) || mrb_float_p(o))
    return scalar_op(mrb, self, (float)mrb_as_float(mrb, o), PIPE_ADDS);
  return type_err(mrb, o, "+");
}

static mrb_value narray_sub(mrb_state *mrb, mrb_value self) {
  mrb_value o; mrb_get_args(mrb, "o", &o);
  GpuBuffer *r = as_narray(o);
  if (r) return binop_nn(mrb, self, r, PIPE_SUB);
  if (mrb_integer_p(o) || mrb_float_p(o))
    return scalar_op(mrb, self, -(float)mrb_as_float(mrb, o), PIPE_ADDS);
  return type_err(mrb, o, "-");
}

static mrb_value narray_mul(mrb_state *mrb, mrb_value self) {
  mrb_value o; mrb_get_args(mrb, "o", &o);
  GpuBuffer *r = as_narray(o);
  if (r) return binop_nn(mrb, self, r, PIPE_MUL);
  if (mrb_integer_p(o) || mrb_float_p(o))
    return scalar_op(mrb, self, (float)mrb_as_float(mrb, o), PIPE_SCALE);
  return type_err(mrb, o, "*");
}

static mrb_value narray_div(mrb_state *mrb, mrb_value self) {
  mrb_value o; mrb_get_args(mrb, "o", &o);
  GpuBuffer *r = as_narray(o);
  if (r) return binop_nn(mrb, self, r, PIPE_DIV);
  if (mrb_integer_p(o) || mrb_float_p(o)) {
    double s = mrb_as_float(mrb, o);        /* a / s == a * (1/s) via SCALE */
    return scalar_op(mrb, self, (float)(1.0 / s), PIPE_SCALE);
  }
  return type_err(mrb, o, "/");
}

/* #-@ -> negation (b = a * -1) */
static mrb_value narray_neg(mrb_state *mrb, mrb_value self) {
  return scalar_op(mrb, self, -1.0f, PIPE_SCALE);
}

/* #sum -> Float. GPU produces one partial per workgroup; host sums them
 * in double precision. The partial buffer is scratch (not a Ruby object),
 * so it is freed explicitly. */
static mrb_value narray_sum(mrb_state *mrb, mrb_value self) {
  GpuBuffer *a = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  if (a->n == 0) return mrb_float_value(mrb, 0.0);
  ensure_pipeline(mrb, PIPE_SUM);

  uint32_t groups = (a->n + 255) / 256;
  GpuBuffer *partial = create_buffer(mrb, groups);
  VkBuffer bufs[2]      = {a->buffer, partial->buffer};
  VkDeviceSize sizes[2] = {a->bytes,  partial->bytes};
  uint32_t push = a->n;
  dispatch_compute(PIPE_SUM, bufs, sizes, 2, &push, sizeof(uint32_t),
                   groups, 1, 1);

  float *pm = map_buffer(partial);
  double total = 0.0;
  for (uint32_t i = 0; i < groups; i++) total += (double)pm[i];
  unmap_buffer(partial);
  destroy_buffer(mrb, partial);

  return mrb_float_value(mrb, (mrb_float)total);
}

/* =========================================================================
 * GPU module functions
 * ========================================================================= */

static mrb_value gpu_s_init(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_get_args(mrb, "z", &path);
  gpu_init(path);
  return mrb_nil_value();
}

static mrb_value gpu_s_device_name(mrb_state *mrb, mrb_value self) {
  ensure_initialized(mrb);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
  return mrb_str_new_cstr(mrb, props.deviceName);
}

static mrb_value gpu_s_info(mrb_state *mrb, mrb_value self) {
  ensure_initialized(mrb);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);

  char api_ver[32];
  snprintf(api_ver, sizeof(api_ver), "%d.%d.%d",
    VK_VERSION_MAJOR(props.apiVersion),
    VK_VERSION_MINOR(props.apiVersion),
    VK_VERSION_PATCH(props.apiVersion));

  mrb_value h = mrb_hash_new(mrb);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "device")),
               mrb_str_new_cstr(mrb, props.deviceName));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "api_version")),
               mrb_str_new_cstr(mrb, api_ver));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "backend")),
               mrb_str_new_cstr(mrb, "Vulkan"));
  return h;
}

/* =========================================================================
 * gem init / final
 * ========================================================================= */

void mrb_mruby_gpu_narray_gem_init(mrb_state *mrb) {
  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  mrb_define_module_function(mrb, gpu, "init",        gpu_s_init,        MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "device_name", gpu_s_device_name, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "info",        gpu_s_info,        MRB_ARGS_NONE());

  struct RClass *narray = mrb_define_class_under(mrb, gpu, "NArray", mrb->object_class);
  MRB_SET_INSTANCE_TT(narray, MRB_TT_CDATA);
  mrb_define_method(mrb, narray, "to_a",   narray_to_a, MRB_ARGS_NONE());
  mrb_define_method(mrb, narray, "head",   narray_head, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "size",   narray_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, narray, "length", narray_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, narray, "seq",    narray_seq,  MRB_ARGS_OPT(2));
  mrb_define_method(mrb, narray, "fill",   narray_fill, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "+",      narray_add,  MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "-",      narray_sub,  MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "*",      narray_mul,  MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "/",      narray_div,  MRB_ARGS_REQ(1));
  mrb_define_method(mrb, narray, "-@",     narray_neg,  MRB_ARGS_NONE());
  mrb_define_method(mrb, narray, "sum",    narray_sum,  MRB_ARGS_NONE());

  struct RClass *sfloat = mrb_define_class_under(mrb, gpu, "SFloat", narray);
  MRB_SET_INSTANCE_TT(sfloat, MRB_TT_CDATA);
  mrb_define_class_method(mrb, sfloat, "new",  sfloat_s_new,  MRB_ARGS_REQ(1));
  mrb_define_class_method(mrb, sfloat, "cast", sfloat_s_cast, MRB_ARGS_REQ(1));
}

void mrb_mruby_gpu_narray_gem_final(mrb_state *mrb) {
  (void)mrb;
  if (g_ctx.initialized) {
    vkDestroyDescriptorPool(g_ctx.device, g_ctx.desc_pool, NULL);
    for (int p = 0; p < PIPE_COUNT; p++) {
      if (g_ctx.pipelines[p] != VK_NULL_HANDLE) {
        vkDestroyPipeline(g_ctx.device, g_ctx.pipelines[p], NULL);
      }
    }
    for (int l = 0; l < LAYOUT_COUNT; l++) {
      vkDestroyPipelineLayout(g_ctx.device, g_ctx.pipe_layouts[l], NULL);
      vkDestroyDescriptorSetLayout(g_ctx.device, g_ctx.desc_layouts[l], NULL);
    }
    vkDestroyCommandPool(g_ctx.device, g_ctx.cmd_pool, NULL);
    vkDestroyDevice(g_ctx.device, NULL);
    vkDestroyInstance(g_ctx.instance, NULL);
    g_ctx.initialized = 0;
  }
}
