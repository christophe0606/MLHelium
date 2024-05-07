/*
 * Copyright (c) 2024 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "network.h"
#include "common.h"
#include <cstdio>

extern "C" {
    #include "GLCD_MPS3.h"
    extern void demo();
}

// Tensor descriptions
#define TA_ID 0
#define TA_LEN 15 

#define TB_ID 1
#define TB_LEN 3


__ALIGNED(16)
float32_t internal_ta[TA_LEN];



void demo() 
{
   GLCD_Initialize();
   GLCD_WindowMax();
   GLCD_Clear(Black);
   GLCD_SetTextColor(White);
   GLCD_SetBackColor(Black);
   disp  ("012345678901234567890123456789");
   /* Read first tensor from network description */
   float32_t *ta = get_f32_tensor(network,TA_ID);
   printf("TA\r\n");
   for(int i=0;i<TA_LEN;i++)
   {
     printf("%f\r\n",ta[i]);
   }
   printf("\r\n");

   /* Read second tensor from network description */
   float16_t *tb = get_f16_tensor(network,TB_ID);
   printf("TB\r\n");
   for(int i=0;i<TB_LEN;i++)
   {
     printf("%f\r\n",tb[i]);
   }
   printf("\r\n");

   /* Copy first tensor to internal memory */
   copy_tensor((uint8_t*)internal_ta,network,TA_ID);
   
   printf("INTERNAL TA\r\n");
   for(int i=0;i<TA_LEN;i++)
   {
     printf("%f\r\n",internal_ta[i]);
   }
   printf("\r\n");
}
