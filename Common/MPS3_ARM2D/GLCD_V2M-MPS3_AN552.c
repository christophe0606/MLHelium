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
 *  File     GLCD_V2M-MPS3_AN540.c
 *  Brief    Graphic LCD interface for V2M-MPS3 evaluation board (AN540)
 *  Version  1.0.0
 *  Note     possible defines select the used communication interface:
 *            __USE_SECURE  - use register in secure address space
 *                          - use register in non-secure address space (default)
 *----------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include  CMSIS_device_header

#if defined(__clang__)
#   pragma clang diagnostic ignored "-Wunknown-warning-option"
#   pragma clang diagnostic ignored "-Wreserved-identifier"
#   pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#endif

#include <stddef.h>

#include "LCD_MPS3.h"
#include "GLCD_Config.h"
#include "Board_GLCD.h"

#define UNUSED(x) (void)(x)             /* macro to get rid of 'unused parameter' warning */


#define GLCD_PORT   CLCD


#define CLCD_CS_Msk      CLCD_MISC_CLCDCS_Msk
#define CLCD_RESET_Msk   CLCD_MISC_CLCDRESET_Msk
#define CLCD_BL_Msk      CLCD_MISC_CLCDBLCTRL_Msk

/*********************** Hardware specific configuration **********************/

/*--------------- Graphic LCD interface hardware definitions -----------------*/

/* Pin CS setting to 0 or 1                                                   */
#define LCD_CS(x)   ((x) ? (GLCD_PORT->MISC |= CLCD_CS_Msk)    : (GLCD_PORT->MISC &= ~CLCD_CS_Msk))
#define LCD_RST(x)  ((x) ? (GLCD_PORT->MISC |= CLCD_RESET_Msk) : (GLCD_PORT->MISC &= ~CLCD_RESET_Msk))
#define LCD_BL(x)   ((x) ? (GLCD_PORT->MISC |= CLCD_BL_Msk)    : (GLCD_PORT->MISC &= ~CLCD_BL_Msk))


/************************ Global variables ************************************/

/* Font definitions */
typedef struct {
        uint16_t width;         ///< Character width
        uint16_t height;        ///< Character height
        uint32_t offset;        ///< Character offset
        uint32_t count;         ///< Character count
  const uint8_t *bitmap;        ///< Characters bitmaps
} GLCD_FONT;


static uint16_t   foreground_color = 0x0000;
static uint16_t   background_color = 0xFFFF;


static volatile unsigned short Color[2] = {0xFFFF, 0x0000};


static enum standardFont_t active_font=kFont6x8;

#if (GLCD_SWAP_XY == 0)
static uint32_t   Scroll;
#endif

/************************ Local auxiliary functions ***************************/

#ifndef EXTERANL_DELAY_MS_DECLARE
/**
  \fn          void delay_ms (int ms)
  \brief       Delay execution for a specified number of milliseconds
  \param[in]   ms     number of while loop iterations
*/
static void delay_ms (uint32_t ms) {
#if defined (RTE_CMSIS_RTOS) && !defined (RTE_CMSIS_RTOS2)
    osDelay(ms);
#elif defined (RTE_CMSIS_RTOS2)
  if (osKernelGetState () == osKernelRunning) {
    osDelay(ms);
  } else {
    ms *= (SystemCoreClock/1000);
    while (ms--) { __NOP(); __NOP(); __NOP(); __NOP(); __NOP(); __NOP(); }
  }
#else
    ms *= (SystemCoreClock/1000);
    while (ms--) { __NOP(); __NOP(); __NOP(); __NOP(); __NOP(); __NOP(); }
#endif
}
#else
EXTERANL_DELAY_MS_DECLARE
#endif

/**
  \fn          void wr_cmd (uint8_t cmd)
  \brief       Write a command to the LCD controller
  \param[in]   cmd  Command to write
*/
static __inline void wr_cmd (uint_fast8_t cmd) {
  LCD_CS(0);
  GLCD_PORT->COM = cmd;
  LCD_CS(1);
}

/**
  \fn          void wr_dat (uint16_t dat)
  \brief       Write data to the LCD controller
  \param[in]   dat  Data to write
*/
static __inline void wr_dat (uint_fast16_t dat) {
  LCD_CS(0);
  GLCD_PORT->DAT = (dat >>   8);                   /* Write D8..D15 */
  GLCD_PORT->DAT = (dat & 0xFF);                   /* Write D0..D7 */
  LCD_CS(1);
}

/**
  \fn          void wr_dat_start (void)
  \brief       Start writing data to LCD controller
*/
static __inline void wr_dat_start (void) {
  LCD_CS(0);
}

/**
  \fn          void wr_dat_stop (void)
  \brief       Stop writing data to LCD controller
*/
static __inline void wr_dat_stop (void) {
  LCD_CS(1);
}

/**
  \fn          void wr_dat_only (uint16_t dat)
  \brief       Write data to the LCD controller (without start/stop sequence)
  \param[in]   dat  Data to write
*/
static __inline void wr_dat_only (uint_fast16_t dat) {
  GLCD_PORT->DAT = (dat >>   8);                   /* Write D8..D15 */
  //GLCD_PORT->DAT = (dat & 0xFF);                   /* Write D0..D7 */
  GLCD_PORT->DAT = dat;
}


/**
  \fn          void wr_reg (uint8_t reg, uint16_t val)
  \brief       Write a value to the LCD register
  \param[in]   reg  Register to be written
  \param[in]   val  Value to write to the register
*/
static __inline void wr_reg (uint_fast8_t reg, uint_fast16_t val) {
  LCD_CS(0);
  GLCD_PORT->COM = reg;
  wr_dat_only(val);
  LCD_CS(1);
}

extern const uint8_t Font_6x8_h[(144-32)*8];
extern const unsigned short Font_16x24_h[(144-32)*24];


void GLCD_SetFontID(const enum standardFont_t f)
{
   active_font = f;
}
/**
  \fn          void GLCD_SetWindow (uint32_t x, uint32_t y, uint32_t w, uint32_t h)
  \brief       Set draw window region
  \param[in]   x  Horizontal position in pixel
  \param[in]   y  Vertical position in pixel
  \param[in]   w  Window width in pixel
  \param[in]   h  Window height in pixel
*/
static void GLCD_SetWindow (uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
  uint32_t xe, ye;

    xe = x+w-1;
    ye = y+h-1;

    wr_reg(0x02, (uint16_t)(x  >>    8));           /* Column address start MSB           */
    wr_reg(0x03, (uint16_t)(x  &  0xFF));           /* Column address start LSB           */
    wr_reg(0x04, (uint16_t)(xe >>    8));           /* Column address end MSB             */
    wr_reg(0x05, (uint16_t)(xe &  0xFF));           /* Column address end LSB             */

    wr_reg(0x06, (uint16_t)(y  >>    8));           /* Row address start MSB              */
    wr_reg(0x07, (uint16_t)(y  &  0xFF));           /* Row address start LSB              */
    wr_reg(0x08, (uint16_t)(ye >>    8));           /* Row address end MSB                */
    wr_reg(0x09, (uint16_t)(ye &  0xFF));           /* Row address end LSB                */
}


/************************ Exported functions **********************************/

/**
  \fn          int32_t GLCD_Initialize (void)
  \brief       Initialize Graphic LCD (HX8347-D LCD Controller)
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_Initialize (void) {

    /* CLCD screen setup (Default CLCD screen interface state) */
    LCD_CS(1);                          /* deassert nCS0 */
    LCD_RST(1);                         /* deassert Reset */
    LCD_BL(0);                          /* switch off backlight */

    // Reset CLCD screen
    LCD_RST(0);                         /* assert Reset */
    delay_ms(1);
    LCD_RST(1);                         /* deassert Reset */
    delay_ms(1);

    /* Driving ability settings ----------------------------------------------*/
    wr_reg(0xEA, 0x00);                 /* Power control internal used (1)    */
    wr_reg(0xEB, 0x20);                 /* Power control internal used (2)    */
    wr_reg(0xEC, 0x0C);                 /* Source control internal used (1)   */
    wr_reg(0xED, 0xC7);                 /* Source control internal used (2)   */
    wr_reg(0xE8, 0x38);                 /* Source output period Normal mode   */
    wr_reg(0xE9, 0x10);                 /* Source output period Idle mode     */
    wr_reg(0xF1, 0x01);                 /* RGB 18-bit interface ;0x0110       */
    wr_reg(0xF2, 0x10);

    /* Adjust the Gamma Curve ------------------------------------------------*/
    wr_reg(0x40, 0x01);
    wr_reg(0x41, 0x00);
    wr_reg(0x42, 0x00);
    wr_reg(0x43, 0x10);
    wr_reg(0x44, 0x0E);
    wr_reg(0x45, 0x24);
    wr_reg(0x46, 0x04);
    wr_reg(0x47, 0x50);
    wr_reg(0x48, 0x02);
    wr_reg(0x49, 0x13);
    wr_reg(0x4A, 0x19);
    wr_reg(0x4B, 0x19);
    wr_reg(0x4C, 0x16);

    wr_reg(0x50, 0x1B);
    wr_reg(0x51, 0x31);
    wr_reg(0x52, 0x2F);
    wr_reg(0x53, 0x3F);
    wr_reg(0x54, 0x3F);
    wr_reg(0x55, 0x3E);
    wr_reg(0x56, 0x2F);
    wr_reg(0x57, 0x7B);
    wr_reg(0x58, 0x09);
    wr_reg(0x59, 0x06);
    wr_reg(0x5A, 0x06);
    wr_reg(0x5B, 0x0C);
    wr_reg(0x5C, 0x1D);
    wr_reg(0x5D, 0xCC);

    /* Power voltage setting -------------------------------------------------*/
    wr_reg(0x1B, 0x1B);
    wr_reg(0x1A, 0x01);
    wr_reg(0x24, 0x2F);
    wr_reg(0x25, 0x57);
    wr_reg(0x23, 0x88);

    /* Power on setting ------------------------------------------------------*/
    wr_reg(0x18, 0x36);                 /* Internal oscillator frequency adj  */
    wr_reg(0x19, 0x01);                 /* Enable internal oscillator         */
    wr_reg(0x01, 0x00);                 /* Normal mode, no scrool             */
    wr_reg(0x1F, 0x88);                 /* Power control 6 - DDVDH Off        */
    delay_ms(20);
    wr_reg(0x1F, 0x82);                 /* Power control 6 - Step-up: 3 x VCI */
    delay_ms(5);
    wr_reg(0x1F, 0x92);                 /* Power control 6 - Step-up: On      */
    delay_ms(5);
    wr_reg(0x1F, 0xD2);                 /* Power control 6 - VCOML active     */
    delay_ms(5);

    /* Color selection -------------------------------------------------------*/
    wr_reg(0x17, 0x55);                 /* RGB, System interface: 16 Bit/Pixel*/
    wr_reg(0x00, 0x00);                 /* Scrolling off, no standby          */

    /* Interface config ------------------------------------------------------*/
    wr_reg(0x2F, 0x11);                 /* LCD Drive: 1-line inversion        */
    wr_reg(0x31, 0x00);
    wr_reg(0x32, 0x00);                 /* DPL=0, HSPL=0, VSPL=0, EPL=0       */

    /* Display on setting ----------------------------------------------------*/
    wr_reg(0x28, 0x38);                 /* PT(0,0) active, VGL/VGL            */
    delay_ms(20);
    wr_reg(0x28, 0x3C);                 /* Display active, VGL/VGL            */

    /* Miror and Swap                                                         */
    wr_reg (0x16, ((GLCD_MIRROR_Y << 7) |   /* MY                             */
                   (GLCD_MIRROR_X << 6) |   /* MX                             */
                   (GLCD_SWAP_XY  << 5) |   /* MV                             */
                   (0 << 4) |               /* ML                             */
                   (1 << 3)));              /* BGR                            */

    /* Display scrolling settings --------------------------------------------*/
    wr_reg(0x0E, 0x00);                 /* TFA MSB                            */
    wr_reg(0x0F, 0x00);                 /* TFA LSB                            */
    wr_reg(0x10, 320 >> 8);             /* VSA MSB                            */
    wr_reg(0x11, 320 &  0xFF);          /* VSA LSB                            */
    wr_reg(0x12, 0x00);                 /* BFA MSB                            */
    wr_reg(0x13, 0x00);                 /* BFA LSB                            */
//  }

    LCD_BL(1);                          /* turn on backlight */

  return 0;
}

/**
  \fn          int32_t GLCD_Uninitialize (void)
  \brief       De-initialize Graphic LCD
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_Uninitialize (void) {

  return 0;
}

/**
  \fn          int32_t GLCD_SetForegroundColor (uint32_t color)
  \brief       Set foreground color
  \param[in]   color  Color value
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_SetForegroundColor (uint32_t color) {
  foreground_color = (uint16_t)color;
  return 0;
}

int32_t GLCD_SetTextForegroundColor (uint32_t color) {
  Color[1] = (uint16_t)color;
  return 0;
}

/**
  \fn          int32_t GLCD_SetBackgroundColor (uint32_t color)
  \brief       Set background color
  \param[in]   color  Color value
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_SetBackgroundColor (uint32_t color) {
  background_color = (uint16_t)color;
  return 0;
}

int32_t GLCD_SetTextBackgroundColor (uint32_t color) {
  Color[0] = (uint16_t)color;
  return 0;
}

/**
  \fn          int32_t GLCD_ClearScreen (void)
  \brief       Clear screen (with active background color)
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_ClearScreen (void) {
  uint32_t i;

  GLCD_SetWindow(0, 0, GLCD_WIDTH, GLCD_HEIGHT);

  wr_cmd(0x22);
  wr_dat_start();
  for (i = 0; i < (GLCD_WIDTH * GLCD_HEIGHT); i++) {
    wr_dat_only(background_color);
  }
  wr_dat_stop();
  return 0;
}


/**
  \fn          int32_t GLCD_DrawPixel (uint32_t x, uint32_t y)
  \brief       Draw pixel (in active foreground color)
  \param[in]   x  x position in pixels (0 = left corner)
  \param[in]   y  y position in pixels (0 = upper corner)
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawPixel (uint32_t x, uint32_t y) {
#if (GLCD_SWAP_XY == 0)
  y = (y + Scroll) % GLCD_HEIGHT;
#endif

    wr_reg(0x02, (uint16_t)(x >>    8));            /* Column address start MSB           */
    wr_reg(0x03, (uint16_t)(x &  0xFF));            /* Column address start LSB           */
    wr_reg(0x04, (uint16_t)(x >>    8));            /* Column address end MSB             */
    wr_reg(0x05, (uint16_t)(x &  0xFF));            /* Column address end LSB             */

    wr_reg(0x06, (uint16_t)(y >>    8));            /* Row address start MSB              */
    wr_reg(0x07, (uint16_t)(y &  0xFF));            /* Row address start LSB              */
    wr_reg(0x08, (uint16_t)(y >>    8));            /* Row address end MSB                */
    wr_reg(0x09, (uint16_t)(y &  0xFF));            /* Row address end LSB                */

  wr_cmd(0x22);
  wr_dat(foreground_color);

  return 0;
}

/**
  \fn          int32_t GLCD_DrawHLine (uint32_t x, uint32_t y, uint32_t length)
  \brief       Draw horizontal line (in active foreground color)
  \param[in]   x      Start x position in pixels (0 = left corner)
  \param[in]   y      Start y position in pixels (0 = upper corner)
  \param[in]   length Line length
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawHLine (uint32_t x, uint32_t y, uint32_t length) {
#if (GLCD_SWAP_XY == 0)
  y = (y + Scroll) % GLCD_HEIGHT;
#endif

  GLCD_SetWindow(x, y, length, 1);
  wr_cmd(0x22);
  wr_dat_start();

  while (length--) {
    wr_dat_only(foreground_color);
  }
  wr_dat_stop();
  return 0;
}

/**
  \fn          int32_t GLCD_DrawVLine (uint32_t x, uint32_t y, uint32_t length)
  \brief       Draw vertical line (in active foreground color)
  \param[in]   x      Start x position in pixels (0 = left corner)
  \param[in]   y      Start y position in pixels (0 = upper corner)
  \param[in]   length Line length in pixels
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawVLine (uint32_t x, uint32_t y, uint32_t length) {
#if (GLCD_SWAP_XY == 0)
  y = (y + Scroll) % GLCD_HEIGHT;
#endif

  GLCD_SetWindow(x, y, 1, length);
  wr_cmd(0x22);
  wr_dat_start();

  while (length--) {
    wr_dat_only(foreground_color);
  }
  wr_dat_stop();
  return 0;
}

/**
  \fn          int32_t GLCD_DrawRectangle (uint32_t x, uint32_t y, uint32_t width, uint32_t height)
  \brief       Draw rectangle (in active foreground color)
  \param[in]   x      Start x position in pixels (0 = left corner)
  \param[in]   y      Start y position in pixels (0 = upper corner)
  \param[in]   width  Rectangle width in pixels
  \param[in]   height Rectangle height in pixels
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawRectangle (uint32_t x, uint32_t y, uint32_t width, uint32_t height) {

  GLCD_DrawHLine (x,         y,          width);
  GLCD_DrawHLine (x,         y + height, width);
  GLCD_DrawVLine (x,         y,          height);
  GLCD_DrawVLine (x + width, y,          height);

  return 0;
}

/**
  \fn          int32_t GLCD_DrawChar (uint32_t x, uint32_t y, int32_t ch)
  \brief       Draw character (in active foreground color)
  \param[in]   x      x position in pixels (0 = left corner)
  \param[in]   y      y position in pixels (0 = upper corner)
  \param[in]   ch     Character
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
void GLCD_DrawChar (unsigned int x, unsigned int y, unsigned int cw, unsigned int ch, const unsigned char *c) {
  unsigned int i, j, k, pixs;

  GLCD_SetWindow(x, y, cw, ch);

  wr_cmd(0x22);
  wr_dat_start();

  k  = (cw + 7)/8;

  if (k == 1) {
    for (j = 0; j < ch; j++) {
      pixs = *(unsigned char  *)c;
      c += 1;

      for (i = 0; i < cw; i++) {
        wr_dat_only (Color[(pixs >> i) & 1]);
      }
    }
  }
  else if (k == 2) {
    for (j = 0; j < ch; j++) {
      pixs = *(unsigned short *)c;
      c += 2;

      for (i = 0; i < cw; i++) {
        wr_dat_only (Color[(pixs >> i) & 1]);
      }
    }
  }
  wr_dat_stop();
}

void GLCD_DisplayChar (unsigned int ln, unsigned int col, unsigned char c) {
  c -= 32;
  switch (active_font) {
    case kFont6x8:  /* Font 6 x 8 */
      GLCD_DrawChar(col *  6, ln *  8,  6,  8, (unsigned char *)&Font_6x8_h  [c * 8]);
      break;
    case kFont16x24:  /* Font 16 x 24 */
      GLCD_DrawChar(col * 16, ln * 24, 16, 24, (unsigned char *)&Font_16x24_h[c * 24]);
      break;
  }
}




/**
  \fn          int32_t GLCD_DrawBargraph (uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t val)
  \brief       Draw bargraph (in active foreground color)
  \param[in]   x      Start x position in pixels (0 = left corner)
  \param[in]   y      Start y position in pixels (0 = upper corner)
  \param[in]   width  Full bargraph width in pixels
  \param[in]   height Full bargraph height in pixels
  \param[in]   val    Active bargraph value (maximum value is 100)
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawBargraph (uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t val) {
  uint32_t i, j;

  val = (val * width) / 100;            /* Scale value                        */

#if (GLCD_SWAP_XY == 0)
  y = (y + Scroll) % GLCD_HEIGHT;
#endif

  GLCD_SetWindow(x, y, width, height);

  wr_cmd(0x22);
  wr_dat_start();
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      if(j >= val) {
        wr_dat_only(background_color);
      } else {
        wr_dat_only(foreground_color);
      }
    }
  }
  wr_dat_stop();

  return 0;
}


/**
  \fn          int32_t GLCD_DrawBitmap (uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t *bitmap)
  \brief       Draw bitmap (bitmap from BMP file without header)
  \param[in]   x      Start x position in pixels (0 = left corner)
  \param[in]   y      Start y position in pixels (0 = upper corner)
  \param[in]   width  Bitmap width in pixels
  \param[in]   height Bitmap height in pixels
  \param[in]   bitmap Bitmap data
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_DrawBitmap (uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t *bitmap) {
  const uint16_t * __RESTRICT ptr_bmp;

  ptr_bmp = (const uint16_t *)((const void *)bitmap);

#if (GLCD_SWAP_XY == 0)
  y = (y + Scroll) % GLCD_HEIGHT;
#endif

  GLCD_SetWindow(x, y, width, height);

  wr_cmd(0x22);
  wr_dat_start();
#if 0
  for (int32_t i = (int32_t)((height-1)*width); i > -1; i -= width) {
    for (int32_t j = 0; j < (int32_t)width; j++) {
      wr_dat_only(ptr_bmp[i+j]);
    }
  }
#else
    uint32_t i = width * height;
    do {
        wr_dat_only(*ptr_bmp++);
    } while(--i);
#endif
  wr_dat_stop();

  return 0;
}


/**
  \fn          int32_t GLCD_VScroll (uint32_t dy)
  \brief       Scroll picture on display vertically
  \param[in]   dy  Scroll size in pixels
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t GLCD_VScroll (uint32_t dy) {
#if (GLCD_SWAP_XY == 0)
  uint32_t i;

  Scroll = (Scroll + dy) % GLCD_HEIGHT;

    wr_reg(0x01, 0x08);
    wr_reg(0x14, Scroll >> 8);          /* VSP MSB                            */
    wr_reg(0x15, Scroll & 0xFF);        /* VSP LSB                            */

  /* Clear new space */
  GLCD_SetWindow(0, Scroll, GLCD_WIDTH, dy);

  wr_cmd(0x22);
  wr_dat_start();
  for(i = 0; i < (GLCD_WIDTH * dy); i++) {
    wr_dat_only(background_color);
  }
  wr_dat_stop();
  return 0;
#else
  UNUSED(dy);
  return -1;
#endif
}

/**
  \fn          int32_t GLCD_FrameBufferAccess (bool enable)
  \brief       Enable or disable direct access to FrameBuffer
  \param[in]   enable Enable or disable request
                - true (1): enable
                - false(0): disable
  \returns
   - \b  0: function succeeded
   - \b -1: function failed
*/
int32_t  GLCD_FrameBufferAccess (bool enable) {
  UNUSED(enable);

  GLCD_SetWindow(0, 0, GLCD_WIDTH, GLCD_HEIGHT);

  wr_reg (0x20, 0x0000);
  wr_reg (0x21, 0x0000);
  wr_cmd (0x22);

  return 0;
}

/**
  \fn          uint32_t GLCD_FrameBufferAddress (void)
  \brief       Retrieve FrameBuffer address
  \returns     FrameBuffer address
*/
uint32_t GLCD_FrameBufferAddress (void) {
  return (uint32_t)NULL;   
}

int GLCD_FontWidth()
{
   if (active_font == kFont6x8)
   {
      return(6);
   }
   else
   {
      return(16);
   }
}

int GLCD_FontHeight()
{
  if (active_font == kFont6x8)
   {
      return(8);
   }
   else
   {
      return(24);
   }
}