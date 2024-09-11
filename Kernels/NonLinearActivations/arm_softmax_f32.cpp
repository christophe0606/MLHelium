#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions.h"
#include "dsp/basic_math_functions.h"


void arm_softmax_f32(float32_t* x, int size) {

    // find max value (for numerical stability)
    float32_t max_val;
    arm_max_no_idx_f32(x, size, &max_val);

    // exp and sum
   
    arm_offset_f32(x,-max_val,x,size);
    arm_vexp_f32(x,x,size);

    float32_t sum = 0.0f;
    for (int i = 0; i < size; i++) 
    {
        sum += x[i];
    }

    arm_scale_f32(x,1.0f / (float)sum,x,size);

}
