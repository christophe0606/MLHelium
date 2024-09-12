#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"
#include "dsp/support_functions_f16.h"


void arm_softmax_f32_f16(float32_t* x, int size,float16_t *tmp) 
{

    arm_float_to_f16(x, tmp, size);

    // find max value (for numerical stability)
    float16_t max_val;
    arm_max_no_idx_f16(tmp, size, &max_val);

    // exp and sum
   
    arm_offset_f16(tmp,-(_Float16)max_val,tmp,size);
    arm_vexp_f16(tmp,tmp,size);

    float16_t sum = 0.0f16;
    for (int i = 0; i < size; i++) 
    {
        sum += (_Float16)tmp[i];
    }

    float32_t scale = 1.0f / (float)sum;

    //arm_scale_f16(tmp,1.0f / (float)sum,tmp,size);

    int32_t  blkCnt;           /* loop counters */
    float16x8_t vecDst;
    float32x4x2_t vtmp;

    blkCnt = size >> 3;
    while (blkCnt > 0)
    {
        vecDst = vldrhq_f16(tmp);          
        tmp += 8;

        vtmp.val[0] = vcvtbq_f32_f16(vecDst);
        vtmp.val[1] = vcvttq_f32_f16(vecDst);
        vtmp.val[0] = vmulq_n_f32(vtmp.val[0],scale);
        vtmp.val[1] = vmulq_n_f32(vtmp.val[1],scale);
        vst2q(x,vtmp);
        
        x += 8;
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
    }
    /*
     * tail
     * (will be merged thru tail predication)
     */
    blkCnt = size & 7;
    while (blkCnt > 0)
    {

        *x++ = (float32_t) *tmp++ * scale;
        /*
         * Decrement the loop counter
         */
        blkCnt--;
    }
}


/*
void arm_softmax_f32_f16(float32_t* x, int size,float16_t *tmp) 
{

    arm_float_to_f16(x, tmp, size);

    // find max value (for numerical stability)
    float16_t max_val;
    arm_max_no_idx_f16(tmp, size, &max_val);

    // exp and sum
   
    arm_offset_f16(tmp,-(_Float16)max_val,tmp,size);
    arm_vexp_f16(tmp,tmp,size);

    float16_t sum = 0.0f16;
    for (int i = 0; i < size; i++) 
    {
        sum += (_Float16)tmp[i];
    }

    arm_scale_f16(tmp,1.0f / (float)sum,tmp,size);

    arm_f16_to_float(tmp, x,size);
}
*/