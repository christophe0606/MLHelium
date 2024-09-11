#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions.h"
#include "dsp/basic_math_functions.h"

#include <dsppp/memory_pool.hpp>
#include <dsppp/matrix.hpp>

using namespace arm_cmsis_dsp;

void arm_rms_norm_f32(float32_t* o, 
                      float32_t* x, 
                      float32_t* weight, 
                      int size) {

    // calculate sum of squares
    VectorView<float32_t> xV(x,0,size);
    VectorView<float32_t> oV(o,0,size);
    VectorView<float32_t> weightV(weight,0,size);
     
    float32_t ss = dot(xV,xV);
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf((float)ss);

    // normalize and scale

    oV = weightV * (xV * ss);

}
