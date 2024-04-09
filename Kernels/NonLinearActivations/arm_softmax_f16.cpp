#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"


void arm_softmax_f16(float16_t* x, int size) {

    // find max value (for numerical stability)
    float16_t max_val;
    arm_max_no_idx_f16(x, size, &max_val);

    // exp and sum
   
    arm_offset_f16(x,-max_val,x,size);
    arm_vexp_f16(x,x,size);

    float16_t sum = 0.0f16;
    for (int i = 0; i < size; i++) 
    {
        sum += (_Float16)x[i];
    }

    arm_scale_f16(x,1.0f / (float)sum,x,size);

}
