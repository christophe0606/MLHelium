#ifndef ML_KERNELS_H
#define ML_KERNELS_H

#include "arm_math_types.h"
#include "arm_math_types_f16.h"
#include "common.h"

#ifdef   __cplusplus
extern "C"
{
#endif

typedef struct
{
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    float8_t *pData;     /**< points to the data of the matrix. */
} arm_matrix_instance_f8;


extern void arm_rms_norm_f16(float16_t* o, 
                             float16_t* x, 
                             float32_t* weight, 
                             int size);
extern void arm_softmax_f16(float16_t* x, int size);
extern void arm_swiglu_f16(float16_t* hb, const float16_t* hb2,int size);

extern void arm_rms_norm_f32(float32_t* o, 
                             float32_t* x, 
                             float32_t* weight, 
                             int size);

extern void arm_softmax_f32(float32_t* x, int size);
extern void arm_softmax_f32_f16(float32_t* x, int size,float16_t *tmp);

extern void arm_swiglu_f32(float32_t* hb, 
                           const float32_t* hb2,
                           int size);

/*
  Mixed arithmetic : weight in float8 and activations in float16
*/
extern void arm_mat_vec_mult_f8_f16(
  const arm_matrix_instance_f8 *pSrcMat, 
  const float16_t *pVec, 
  float16_t *pDst);

extern void arm_rms_norm_f8_f16(float16_t* o, float16_t* x, float8_t* weight, int size);
extern void arm_mult_f8_f16(
  const float8_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t blockSize);
extern void arm_copy_f8_to_f16(float16_t *dst, const float8_t *src, int nb);

#ifdef   __cplusplus
}
#endif

#endif
