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