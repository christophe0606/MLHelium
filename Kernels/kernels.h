#ifndef ML_KERNELS_H
#define ML_KERNELS_H

#include "arm_math_types.h"
#include "arm_math_types_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

extern void arm_rms_norm_f16(float16_t* o, float16_t* x, float16_t* weight, int size);
extern void arm_softmax_f16(float16_t* x, int size);

#ifdef   __cplusplus
}
#endif

#endif
