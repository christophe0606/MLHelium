#include "kernels.h"

#include <math.h>
#include "arm_math_types_f16.h"
#include <cstdint>

#include <cstdio>

void arm_copy_f8_to_f16(float16_t *dst, const float8_t *src, int nb)
{ 
   uint32_t blkCnt;  
   int8x16_t vec;
   float16x8_t vecA;
   float16x8_t vecB;

   blkCnt = nb >> 4U;
   while (blkCnt > 0U)
   {
        

        vec = vld1q((int8_t const *) src);
        vecA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        vstrhq_f16(dst,vecA);

        vecB = vreinterpretq_f16_s16(vshlltq_n_s8(vec,8));
        vstrhq_f16(dst+8,vecB);
        /*
         * Decrement the blockSize loop counter
         * Advance vector source and destination pointers
         */
        src += 16;
        dst += 16;
        blkCnt--;
   }

   blkCnt = nb & 0xF;
   if (blkCnt > 8U)
   {
        mve_pred16_t p0 = vctp16q(blkCnt-8);

        vec = vld1q((int8_t const *) src);
        vecA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        vstrhq_f16(dst, vecA);

        vecB = vreinterpretq_f16_s16(vshlltq_n_s8(vec,8));
        vstrhq_p_f16(dst+8, vecB, p0);
   }
   else if (blkCnt > 0)
   {
        mve_pred16_t p0 = vctp16q(blkCnt-8);

        vec = vld1q((int8_t const *) src);
        vecA = vreinterpretq_f16_s16(vshllbq_n_s8(vec,8));
        vstrhq_p_f16(dst, vecA, p0);
   }

    

}