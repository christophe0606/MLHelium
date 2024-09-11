#ifndef ML_HELIUM_COMMON_H
#define ML_HELIUM_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include "arm_math_types.h"
#include "arm_math_types_f16.h"

typedef uint32_t byte_index_t;
typedef uint32_t byte_length_t;

#ifdef   __cplusplus
extern "C"
{
#endif

typedef int8_t float8_t;

extern byte_length_t get_tensor_length(const unsigned char *,
                                       const int tensor_nb);

extern float32_t *get_f32_tensor(const unsigned char *,
                                 const int nb);

extern float8_t *get_f8_tensor(const unsigned char *,
                               const int nb);

extern int8_t *get_i8_tensor(const unsigned char *,
                             const int nb);

#if defined(ARM_FLOAT16_SUPPORTED)
extern float16_t *get_f16_tensor(const unsigned char *,
                                 const int nb);
#endif 

extern void copy_tensor(unsigned char *dst,
                        const unsigned char *network,
                        const int tensor_nb);

extern void sub_copy_tensor(unsigned char *dst,
                            const unsigned char *network,
                            const int tensor_nb,
                            const byte_index_t start_index,
                            const byte_index_t stop_index);

extern void* aligned_malloc(size_t alignment, size_t size,size_t *allocated_memory);
extern void  aligned_free(void* ptr);

typedef struct {
size_t current_bytes;
size_t maximum_bytes;
unsigned char *memory;
} memory_area_t;

extern void free_area(memory_area_t * area);
extern void* add_aligned_buffer_to_area(memory_area_t * area,size_t bytes,size_t alignment);
extern void init_memory_area(memory_area_t *area,unsigned char *buffer,size_t nb_bytes);

#ifdef   __cplusplus
}
#endif

#endif