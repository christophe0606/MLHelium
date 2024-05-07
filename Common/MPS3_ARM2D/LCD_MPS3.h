/*
 * Copyright (c) 2019 Arm Limited. All rights reserved.
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

/*-----------------------------------------------------------------------------
 * File     MPS3_CM55.h
 * Brief    CMSIS Core Peripheral Access Layer Header File for
 *          MPS3_CM55 Device (AN540)
 * Version  1.0.0
 *----------------------------------------------------------------------------*/

#ifndef MPS3_CM55_H
#define MPS3_CM55_H

#ifdef __cplusplus
extern "C" {
#endif






/*------------------- Color LCD (CLCD) ------------------------*/
typedef struct /* see "?" */
{
  __IOM uint32_t COM;                        /* Offset: 0x000 (R/W)   */
  __IOM uint32_t DAT;                        /* Offset: 0x004 (R/W)   */
  __IOM uint32_t RD;                         /* Offset: 0x008 (R/W)   */
  __IOM uint32_t RAW;                        /* Offset: 0x00C (R/W)   */
  __IOM uint32_t MASK;                       /* Offset: 0x010 (R/W)   */
  __IOM uint32_t STAT;                       /* Offset: 0x014 (R/W)   */
        uint32_t RESERVED0[13];              /* Offset: 0x018 (R/W)   */
  __IOM uint32_t MISC;                       /* Offset: 0x04C (R/W)   */
} CLCD_TypeDef;

/* CLDC_MISC Register Definitions */
#define CLCD_MISC_CLCDBLCTRL_Pos          6                                              /* CLCD MISC.CLCDBLCTRL: CLCD_BL_CTRL position */
#define CLCD_MISC_CLCDBLCTRL_Msk         (1UL << CLCD_MISC_CLCDBLCTRL_Pos)               /* CLCD MISC.CLCDBLCTRL: CLCD_BL_CTRL mask */

#define CLCD_MISC_CLCDRD_Pos              5                                              /* CLCD MISC.CLCDRD: CLCD_RD position */
#define CLCD_MISC_CLCDRD_Msk             (1UL << CLCD_MISC_CLCDRD_Pos)                   /* CLCD MISC.CLCDRD: CLCD_RD mask */

#define CLCD_MISC_CLCDRS_Pos              4                                              /* CLCD MISC.CLCDRS: CLCD_RS position */
#define CLCD_MISC_CLCDRS_Msk             (1UL << CLCD_MISC_CLCDRS_Pos)                   /* CLCD MISC.CLCDRS: CLCD_RS mask */

#define CLCD_MISC_CLCDRESET_Pos           3                                              /* CLCD MISC.CLCDRESET: CLCD_RESET position */
#define CLCD_MISC_CLCDRESET_Msk          (1UL << CLCD_MISC_CLCDRESET_Pos)                /* CLCD MISC.CLCDRESET: CLCD_RESET mask */

#define CLCD_MISC_CLCDCS_Pos              0                                              /* CLCD MISC.CLCDCS: CLCD_CS position */
#define CLCD_MISC_CLCDCS_Msk             (1UL /*<< CLCD_MISC_CLCDCS_Pos*/)               /* CLCD MISC.CLCDCS: CLCD_CS mask */



#define        CLCD          ((CLCD_TypeDef             *)        CLCD_Config_Reg_BASE_NS)
#define SECURE_CLCD          ((CLCD_TypeDef             *) CLCD_Config_Reg_BASE_S)

#ifdef __cplusplus
}
#endif

#endif  /* MPS3_CM55_H */
