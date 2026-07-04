/* gpu_buffer.c -- GpuBuffer lifetime + host mapping.
 *
 * Ported from yujiteshima/mruby-gpu (src/gpu_buffer.c). The host-visible +
 * host-coherent allocation strategy and the GC finalizer are unchanged; on
 * Raspberry Pi 5 the GPU memory is unified, so a mapped pointer is a cheap
 * view of the same bytes the shader reads/writes.
 *
 * Two ownership models share one GpuBuffer type:
 *   - wrap_buffer()   -> owned by a Ruby object, freed by the GC finalizer.
 *   - create_buffer() + destroy_buffer() -> caller-managed scratch buffers
 *     (used for reduction partials that never become Ruby objects).
 */
#include "gpu_internal.h"

/* ---- GC finalizer for wrapped buffers ---- */
static void gpu_buffer_free(mrb_state *mrb, void *p) {
  destroy_buffer(mrb, (GpuBuffer *)p);
}

const struct mrb_data_type gpu_buffer_type = {"GPU::NArray", gpu_buffer_free};

/* ---- Create a host-visible FP32 buffer of n elements ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n) {
  GpuBuffer *buf = mrb_malloc(mrb, sizeof(GpuBuffer));
  buf->n = n;
  buf->bytes = sizeof(float) * (VkDeviceSize)n;
  if (buf->bytes == 0) buf->bytes = sizeof(float); /* avoid zero-size allocation */

  VkBufferCreateInfo bi = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buf->bytes,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
  };
  vkCreateBuffer(g_ctx.device, &bi, NULL, &buf->buffer);

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_ctx.device, buf->buffer, &req);

  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(g_ctx.physical_device, &mem_props);
  uint32_t mem_idx = 0;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((req.memoryTypeBits & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags &
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      mem_idx = i;
      break;
    }
  }

  VkMemoryAllocateInfo ai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size,
    .memoryTypeIndex = mem_idx
  };
  vkAllocateMemory(g_ctx.device, &ai, NULL, &buf->memory);
  vkBindBufferMemory(g_ctx.device, buf->buffer, buf->memory, 0);

  return buf;
}

/* ---- Free a buffer's GPU resources + the struct itself ---- */
void destroy_buffer(mrb_state *mrb, GpuBuffer *buf) {
  if (!buf) return;
  if (g_ctx.initialized) {
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

/* ---- Host mapping helpers (coherent memory, no flush needed) ---- */
float *map_buffer(GpuBuffer *buf) {
  float *mapped = NULL;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  return mapped;
}

void unmap_buffer(GpuBuffer *buf) {
  vkUnmapMemory(g_ctx.device, buf->memory);
}
