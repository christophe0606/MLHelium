#include "kernels.h"

#include <math.h>
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"

void arm_swiglu_f16(float16_t* hb, const float16_t* hb2,int size)
{
   for (int i = 0; i < size; i++) 
   {
        float16_t val = hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (_Float16)((_Float16)1.0f16 / (_Float16)((_Float16)1.0f16 + (_Float16)expf((float)-val)));
        // elementwise multiply with w3(x)
        val *= (_Float16)(hb2[i]);
        hb[i] = val;
    }
}