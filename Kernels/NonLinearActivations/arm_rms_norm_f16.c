#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"

void arm_rms_norm_f16(float16_t* o, float16_t* x, float16_t* weight, int size) {

    // calculate sum of squares
    float16_t ss = 0.0f16;
    arm_power_f16(x, size, &ss);
    ss /= (_Float16)size;
    ss += 1e-5f16;
    ss = 1.0f16 / (_Float16)sqrtf((float)ss);

    // normalize and scale
    arm_scale_f16(x,ss,o,size);
    arm_mult_f16(weight,o,o,size);
    

}
