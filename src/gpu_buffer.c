/* gpu_buffer.c -- GpuBuffer lifetime + host mapping.
 *
 * Ported from yujiteshima/mruby-gpu (src/gpu_buffer.c). The host-visible +
 * host-coherent allocation strategy and the GC finalizer are unchanged; on
 * Raspberry Pi 5 the GPU memory is unified, so a mapped pointer is a cheap
 * view of the same bytes the shader reads/writes.
 *
 * Every buffer ends up owned by a Ruby object and freed by the GC finalizer.
 * Callers wrap as soon as create_buffer returns, before anything that can
 * raise, so a Vulkan failure unwinds without stranding GPU memory.
 * destroy_buffer is the finalizer itself, plus create_buffer's own unwind.
 *
 * With deferred submission a buffer can be referenced by a command buffer that
 * has not run yet. Two rules keep that safe:
 *   - map_buffer flushes the batch first if it touches this buffer, so the
 *     host never reads a result that is still queued, and never overwrites an
 *     input that a queued dispatch has yet to read.
 *   - destroy_buffer parks such a buffer in the graveyard instead of freeing
 *     it; gpu_flush frees the graveyard once the batch has completed.
 */
#include "gpu_internal.h"

/* ---- GC finalizer for wrapped buffers ---- */
static void gpu_buffer_free(mrb_state *mrb, void *p) {
  destroy_buffer(mrb, (GpuBuffer *)p);
}

const struct mrb_data_type gpu_buffer_type = {"GPU::NArray", gpu_buffer_free};

/* ---- Create a host-visible FP32 buffer of n elements ----
 *
 * Raises rather than returning a half-built buffer. Because a raise unwinds
 * out of this function, each step releases what earlier steps created before
 * handing the failure to gpu_check. */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n) {
  GpuBuffer *buf = mrb_malloc(mrb, sizeof(GpuBuffer));
  buf->n = n;
  buf->buffer = VK_NULL_HANDLE;
  buf->memory = VK_NULL_HANDLE;
  buf->bytes = sizeof(float) * (VkDeviceSize)n;
  buf->epoch = 0;   /* never bound; no batch can be waiting on it */
  if (buf->bytes == 0) buf->bytes = sizeof(float); /* avoid zero-size allocation */

  VkBufferCreateInfo bi = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buf->bytes,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
  };
  VkResult r = vkCreateBuffer(g_ctx.device, &bi, NULL, &buf->buffer);
  if (r != VK_SUCCESS) {
    mrb_free(mrb, buf);
    gpu_check(mrb, r, "vkCreateBuffer");
  }

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_ctx.device, buf->buffer, &req);

  /* The mapped pointer is how every host read and write reaches this buffer,
   * so a memory type that is not both host-visible and host-coherent is no
   * use. Defaulting to index 0 when none matches would allocate from an
   * arbitrary heap and turn every later map_buffer into a NULL dereference. */
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(g_ctx.physical_device, &mem_props);
  const VkMemoryPropertyFlags want =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t mem_idx = UINT32_MAX;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((req.memoryTypeBits & (1u << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & want) == want) {
      mem_idx = i;
      break;
    }
  }
  if (mem_idx == UINT32_MAX) {
    vkDestroyBuffer(g_ctx.device, buf->buffer, NULL);
    mrb_free(mrb, buf);
    mrb_raise(mrb, E_RUNTIME_ERROR,
      "no host-visible, host-coherent memory type is available for a storage buffer");
  }

  VkMemoryAllocateInfo ai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size,
    .memoryTypeIndex = mem_idx
  };
  r = vkAllocateMemory(g_ctx.device, &ai, NULL, &buf->memory);
  if (r != VK_SUCCESS) {
    vkDestroyBuffer(g_ctx.device, buf->buffer, NULL);
    mrb_free(mrb, buf);
    gpu_check(mrb, r, "vkAllocateMemory");
  }

  r = vkBindBufferMemory(g_ctx.device, buf->buffer, buf->memory, 0);
  if (r != VK_SUCCESS) {
    vkFreeMemory(g_ctx.device, buf->memory, NULL);
    vkDestroyBuffer(g_ctx.device, buf->buffer, NULL);
    mrb_free(mrb, buf);
    gpu_check(mrb, r, "vkBindBufferMemory");
  }

  return buf;
}

/* ---- Free a buffer's GPU resources + the struct itself ----
 *
 * Runs as the GC finalizer, so it must not raise and must not block: a buffer
 * the pending batch still references is parked in the graveyard and freed by
 * the next gpu_flush, after the GPU is done with it. */
void destroy_buffer(mrb_state *mrb, GpuBuffer *buf) {
  if (!buf) return;
  if (g_ctx.initialized) {
    if (g_ctx.batch_recording && buf->epoch == g_ctx.epoch) {
      if (g_ctx.graveyard_len == g_ctx.graveyard_cap) {
        size_t cap = g_ctx.graveyard_cap ? g_ctx.graveyard_cap * 2 : 64;
        GpuBuffer **g = realloc(g_ctx.graveyard, cap * sizeof(GpuBuffer *));
        if (!g) {
          /* Cannot park it and must not free it under the GPU's feet. Leaking
           * one buffer is the least bad outcome inside a finalizer. */
          fprintf(stderr, "mruby-gpu-narray: out of memory deferring a buffer free; leaking it\n");
          return;
        }
        g_ctx.graveyard = g;
        g_ctx.graveyard_cap = cap;
      }
      g_ctx.graveyard[g_ctx.graveyard_len++] = buf;
      return;
    }
    vkDestroyBuffer(g_ctx.device, buf->buffer, NULL);
    vkFreeMemory(g_ctx.device, buf->memory, NULL);
  }
  mrb_free(mrb, buf);
}

/* ---- Wrap a buffer in a Ruby object of the given class ---- */
mrb_value wrap_buffer(mrb_state *mrb, struct RClass *klass, GpuBuffer *buf) {
  struct RData *data = mrb_data_object_alloc(mrb, klass, buf, &gpu_buffer_type);
  return mrb_obj_value(data);
}

/* ---- Host mapping helpers (coherent memory, no flush needed) ----
 *
 * Never returns NULL: every caller writes through the pointer immediately, so
 * an unreported map failure would be a NULL dereference inside the VM.
 *
 * This is the sync point of the whole library: the host is about to look at
 * (or change) the bytes, so any queued dispatch that reads or writes this
 * buffer has to run first. A buffer the batch never touched maps at once. */
float *map_buffer(mrb_state *mrb, GpuBuffer *buf) {
  gpu_sync_buffer(mrb, buf);
  float *mapped = NULL;
  VK_CHECK(mrb, vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped));
  if (!mapped) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "vkMapMemory returned no pointer");
  }
  return mapped;
}

void unmap_buffer(GpuBuffer *buf) {
  vkUnmapMemory(g_ctx.device, buf->memory);
}
