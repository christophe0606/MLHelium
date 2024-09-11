#include "kernels.h"

#include <math.h>
#include "arm_math_types_f16.h"

void arm_mult_f8_f16(
  const float8_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t nb)
{
   uint32_t blkCnt;  
   int8x16_t vec;
   float16x8_t vecLA;
   float16x8_t vecLB;
   float16x8_t vecR;
   float16x8_t tmp;

   blkCnt = nb >> 4U;
   while (blkCnt > 0U)
   {
        

        vec = vld1q((int8_t const *) pSrcA);

        vecR = vld1q(pSrcB);
        vecLA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        tmp = vmulq(vecLA,vecR);
        vstrhq_f16(pDst,tmp);

        vecR = vld1q(pSrcB+8);
        vecLB = vreinterpretq_f16_s16(vshlltq_n_s8(vec,8));
        tmp = vmulq(vecLB,vecR);
        vstrhq_f16(pDst+8,tmp);
        /*
         * Decrement the blockSize loop counter
         * Advance vector source and destination pointers
         */
        pSrcA += 16;
        pSrcB += 16;
        pDst += 16;
        blkCnt--;
   }

   blkCnt = nb & 0xF;
   if (blkCnt > 8U)
   {
        mve_pred16_t p0 = vctp16q(blkCnt-8);

        vec = vld1q((int8_t const *) pSrcA);
        vecR = vld1q(pSrcB);
        vecLA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        tmp = vmulq(vecLA,vecR);
        vstrhq_f16(pDst,tmp);

        vecR = vld1q(pSrcB+8);
        vecLB = vreinterpretq_f16_s16(vshlltq_n_s8(vec,8));
        tmp = vmulq(vecLB,vecR);
        vstrhq_p_f16(pDst+8, tmp, p0);
   }
   else if (blkCnt > 0)
   {
        mve_pred16_t p0 = vctp16q(blkCnt-8);

        vec = vld1q((int8_t const *) pSrcA);
        vecR = vld1q(pSrcB);
        vecLA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        tmp = vmulq(vecLA,vecR);
        vstrhq_p_f16(pDst, tmp, p0);
   }

}