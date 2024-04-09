#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"

#include <dsppp/memory_pool.hpp>
#include <dsppp/matrix.hpp>

using namespace arm_cmsis_dsp;

void arm_rms_norm_f16(float16_t* o, float16_t* x, float16_t* weight, int size) {

    // calculate sum of squares
    VectorView<float16_t> xV(x,0,size);
    VectorView<float16_t> oV(o,0,size);
    VectorView<float16_t> weightV(weight,0,size);
     
    float16_t ss = dot(xV,xV);
    ss /= (_Float16)size;
    ss += 1e-5f16;
    ss = 1.0f16 / (_Float16)sqrtf((float)ss);

    // normalize and scale

    oV = weightV * (xV * ss);

}
