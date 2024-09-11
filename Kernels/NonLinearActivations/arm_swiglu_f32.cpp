#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions.h"
#include "dsp/basic_math_functions.h"

void arm_swiglu_f32(float32_t* hb, 
                    const float32_t* hb2,
                    int size)
{
   for (int i = 0; i < size; i++) 
   {
        float32_t val = hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf((float)-val)));
        // elementwise multiply with w3(x)
        val *= (hb2[i]);
        hb[i] = val;
    }
}