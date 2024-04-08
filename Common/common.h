#ifndef ML_HELIUM_COMMON_H
#define ML_HELIUM_COMMON_H

#include <stdint.h>
#include "arm_math_types.h"
#include "arm_math_types_f16.h"

typedef uint32_t byte_index_t;
typedef uint32_t byte_length_t;

#ifdef   __cplusplus
extern "C"
{
#endif

extern byte_length_t get_tensor_length(const unsigned char *,
                                       const int tensor_nb);

extern float32_t *get_f32_tensor(const unsigned char *,
                                 const int nb);

#if defined(ARM_FLOAT16_SUPPORTED)
extern float16_t *get_f16_tensor(const unsigned char *,
                                 const int nb);
#endif 

void copy_tensor(unsigned char *dst,
                 const unsigned char *network,
                 const int tensor_nb);

void sub_copy_tensor(unsigned char *dst,
                     const unsigned char *network,
                     const int tensor_nb,
                     const byte_index_t start_index,
                     const byte_index_t stop_index
                     );

#ifdef   __cplusplus
}
#endif

#endif