#include "common.h"
#include <string.h>

#define NB_TENSORS (tensors[0])

#define TENSOR_OFFSET_POS(ID) \
   (1 + 2*NB_TENSORS + (ID))

#define TENSOR_LENGTH_POS(ID) \
   (1 + (ID))


/**
 * @brief      Gets the tensor length in bytes
 *
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 *
 * @return     The tensor length in bytes.
 */
byte_length_t get_tensor_length(const unsigned char *network,
                                const int tensor_nb)
{
    const uint32_t *tensors = (const uint32_t *)network;
    return(tensors[TENSOR_LENGTH_POS(tensor_nb)]);
}

/**
 * @brief      Gets the f32 tensor pointer.
 *
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 *
 * @return     The pointer to the f32 tensor
 */
float32_t* get_f32_tensor(const unsigned char *network,
                          const int tensor_nb)
{
   const uint32_t *tensors = (const uint32_t *)network;
   const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
   return((float32_t*)(network+offset));
}

/**
 * @brief      Gets the f8 tensor pointer.
 *
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 *
 * @return     The pointer to the f8 tensor
 * 
 * @par        float8 format
 *             Format with 2 bit of mantissa
 */
float8_t *get_f8_tensor(const unsigned char *network,
                        const int tensor_nb)
{
   const uint32_t *tensors = (const uint32_t *)network;
   const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
   return((float8_t*)(network+offset));
}

/**
 * @brief      Gets the i8 tensor pointer.
 *
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 *
 * @return     The pointer to the i8 tensor
 * 
 * @par        int8 format
 */
int8_t *get_i8_tensor(const unsigned char *network,
                        const int tensor_nb)
{
   const uint32_t *tensors = (const uint32_t *)network;
   const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
   return((int8_t*)(network+offset));
}

#if defined(ARM_FLOAT16_SUPPORTED)
/**
 * @brief      Gets the f16 tensor pointer.
 *
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 *
 * @return     The pointer to the f16 tensor
 */
float16_t* get_f16_tensor(const unsigned char *network,
                          const int tensor_nb)
{
   const uint32_t *tensors = (const uint32_t *)network;
   const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
   return((float16_t*)(network+offset));
}
#endif 

/**
 * @brief      Convenience function to copy a tensor to internal memory
 *
 * @param      dst        The destination pointer
 * @param[in]  network    The network description
 * @param[in]  tensor_nb  The tensor number
 * 
 * In can also be implemented using the get_tensor_length and
 * get_f16_tensor / get_f32_tensor.
 * 
 */
void copy_tensor(unsigned char *dst,
                 const unsigned char *network,
                 const int tensor_nb)
{
    const uint32_t *tensors = (const uint32_t *)network;
    const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
    const unsigned char * src = (const unsigned char *)(network+offset);
    memcpy(dst,src,get_tensor_length(network,tensor_nb));
}

/**
 * @brief      Convenience function to copy a part of a 
 *             tensor to internal memory
 *
 * @param      dst          The destination buffer
 * @param[in]  network      The network description
 * @param[in]  tensor_nb    The tensor number
 * @param[in]  start_index  The start index in bytes
 * @param[in]  stop_index   The stop index in bytes
 * 
 * If the copied part is too small, the overhead of finding
 * the tensor in the networm description will be too high.
 * In that case, it may be better to use the function like
 * get_f32_tensor / get_f16_tensor and use those pointers
 * directly.
 */
void sub_copy_tensor(unsigned char *dst,
                     const unsigned char *network,
                     const int tensor_nb,
                     const byte_index_t start_index,
                     const byte_index_t stop_index
                     )
{
   const uint32_t *tensors = (const uint32_t *)network;
   const uint32_t offset = tensors[TENSOR_OFFSET_POS(tensor_nb)];
   const unsigned char * src = (const unsigned char *)(network+offset);
   memcpy(dst,src+start_index,stop_index - start_index);
}

/**
 * @brief      Memory allocation with alignment
 *
 * @param[in]  alignment         The alignment in bytes
 * @param[in]  size              The size in bytes
 * @param      allocated_memory  The amount of reserved bytes (may be a bit bigger then size)
 *
 * @return     Pointer to the allocated buffer
 */
void* aligned_malloc(size_t alignment, size_t size,size_t *allocated_memory)
{
   void *ptr=malloc(size+alignment+sizeof(void*));
   if (!ptr)
   {
      *allocated_memory = 0;
      return(NULL);
   }
   *allocated_memory = size+alignment+sizeof(void*);
   void *aligned = (char*)(((size_t)(ptr)+sizeof(void*)+alignment) & ~(alignment-1));

   *((void**)(aligned) - 1) = ptr;
   return(aligned);
}

/**
 * @brief      Free a buffer allocated with the aligned malloc
 *
 * @param      ptr   The pointer
 */
void aligned_free(void* ptr)
{
    if (ptr) {
        free(*((void**)(ptr) - 1));
    }
};

/**
 * @brief      Free the memory area
 *
 * @param      area  The memory area
 */
void free_area(memory_area_t * area)
{
   if (area)
   {
      area->current_bytes = 0;
   }
}

/**
 * @brief      Adds an aligned buffer to area.
 *
 * @param      area       The memory area
 * @param[in]  bytes      The number of bytes to allocate
 * @param[in]  alignment  The alignment
 *
 * @return     Pointer to the reserved an aligned buffer
 */
void* add_aligned_buffer_to_area(memory_area_t * area,size_t bytes,size_t alignment)
{
   if (area)
   {
      unsigned char *ptr=area->memory+area->current_bytes;

      if ((size_t)ptr & alignment)
      {
          size_t delta = alignment - ((size_t)ptr & alignment);
          ptr = (unsigned char*)((size_t)ptr + delta);
          if ((area->current_bytes + bytes + delta)>area->maximum_bytes)
          {
             return(NULL);
          }
          area->current_bytes += bytes + delta;
          return((void*)ptr);
      }
      else 
      {
        if ((area->current_bytes + bytes)>area->maximum_bytes)
        {
             return(NULL);
        }
        area->current_bytes += bytes;
        return((void*)ptr);
      }
   }
   return(NULL);
}

/**
 * @brief      Initializes a memory area from an internal memory buffer
 *
 * @param      area      The memory area
 * @param      buffer    The internal buffer
 * @param[in]  nb_bytes  The length in bytes of the internal buffer
 */
void init_memory_area(memory_area_t *area,unsigned char * buffer,size_t nb_bytes)
{
    if (area)
    {
       area->memory = buffer;
       area->maximum_bytes = nb_bytes;
       area->current_bytes = 0;
    }
}