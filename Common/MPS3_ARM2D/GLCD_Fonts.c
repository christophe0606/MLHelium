/*-----------------------------------------------------------------------------
 * Name:    GLCD_Fonts.c
 * Purpose: Graphic fonts 6x8 (WxH) and 16x24 with horizontal pixel packing
 * Rev.:    1.0.1
 *----------------------------------------------------------------------------*/

/* Copyright (c) 2013 - 2017 ARM LIMITED

   All rights reserved.
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   - Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   - Neither the name of ARM nor the names of its contributors may be used
     to endorse or promote products derived from this software without
     specific prior written permission.
   *
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS AND CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
   ---------------------------------------------------------------------------*/

#include <stdint.h>

#if defined(__clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wunknown-warning-option"
#   pragma clang diagnostic ignored "-Wreserved-identifier"
#   pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#   pragma clang diagnostic ignored "-Wcast-qual"
#endif

extern const uint8_t Font_6x8_h[(144-32)*8];
extern const unsigned short Font_16x24_h[(144-32)*24];

const uint8_t Font_6x8_h[(144-32)*8] = {
  /* 0x20: Space ' ' */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  /* 0x21: '!' */
  0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00,
  /* 0x22: '"' */
  0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00,
  /* 0x23: '#' */
  0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A, 0x00,
  /* 0x24: '$' */
  0x04, 0x1E, 0x05, 0x0E, 0x14, 0x0F, 0x04, 0x00,
  /* 0x25: '%' */
  0x03, 0x13, 0x08, 0x04, 0x02, 0x19, 0x18, 0x00,
  /* 0x26: '&' */
  0x02, 0x05, 0x05, 0x02, 0x15, 0x09, 0x16, 0x00,
  /* 0x27: ''' */
  0x0C, 0x0C, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00,
  /* 0x28: '(' */
  0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00,
  /* 0x29: ')' */
  0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00,
  /* 0x2A: '*' */
  0x00, 0x04, 0x15, 0x0E, 0x0E, 0x15, 0x04, 0x00,
  /* 0x2B: '+' */
  0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00, 0x00,
  /* 0x2C: ',' */
  0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x04, 0x02,
  /* 0x2D: '-' */
  0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00,
  /* 0x2E: '.' */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00,
  /* 0x2F: '/' */
  0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00, 0x00,
  /* 0x30: '0' */
  0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00,
  /* 0x31: '1' */
  0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00,
  /* 0x32: '2' */
  0x0E, 0x11, 0x10, 0x0E, 0x01, 0x01, 0x1F, 0x00,
  /* 0x33: '3' */
  0x1F, 0x10, 0x08, 0x0C, 0x10, 0x11, 0x0E, 0x00,
  /* 0x34: '4' */
  0x08, 0x0C, 0x0A, 0x09, 0x1F, 0x08, 0x08, 0x00,
  /* 0x35: '5' */
  0x1F, 0x01, 0x0F, 0x10, 0x10, 0x11, 0x0E, 0x00,
  /* 0x36: '6' */
  0x1C, 0x02, 0x01, 0x0F, 0x11, 0x11, 0x0E, 0x00,
  /* 0x37: '7' */
  0x1F, 0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00,
  /* 0x38: '8' */
  0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00,
  /* 0x39: '9' */
  0x0E, 0x11, 0x11, 0x1E, 0x10, 0x08, 0x07, 0x00,
  /* 0x3A: ':' */
  0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  /* 0x3B: ';' */
  0x00, 0x00, 0x04, 0x00, 0x04, 0x04, 0x02, 0x00,
  /* 0x3C: '<' */
  0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10, 0x00,
  /* 0x3D: '=' */
  0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00, 0x00,
  /* 0x3E: '>' */
  0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02, 0x00,
  /* 0x3F: '?' */
  0x0E, 0x11, 0x10, 0x0C, 0x04, 0x00, 0x04, 0x00,
  /* 0x40: '@' */
  0x0E, 0x11, 0x15, 0x1D, 0x0D, 0x01, 0x1E, 0x00,
  /* 0x41: 'A' */
  0x04, 0x0A, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x00,
  /* 0x42: 'B' */
  0x0F, 0x11, 0x11, 0x0F, 0x11, 0x11, 0x0F, 0x00,
  /* 0x43: 'C' */
  0x0E, 0x11, 0x01, 0x01, 0x01, 0x11, 0x0E, 0x00,
  /* 0x44: 'D' */
  0x0F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0F, 0x00,
  /* 0x45: 'E' */
  0x1F, 0x01, 0x01, 0x0F, 0x01, 0x01, 0x1F, 0x00,
  /* 0x46: 'F' */
  0x1F, 0x01, 0x01, 0x0F, 0x01, 0x01, 0x01, 0x00,
  /* 0x47: 'G' */
  0x1E, 0x11, 0x01, 0x01, 0x19, 0x11, 0x1E, 0x00,
  /* 0x48: 'H' */
  0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00,
  /* 0x49: 'I' */
  0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00,
  /* 0x4A: 'J' */
  0x1C, 0x08, 0x08, 0x08, 0x08, 0x09, 0x06, 0x00,
  /* 0x4B: 'K' */
  0x11, 0x09, 0x05, 0x03, 0x05, 0x09, 0x11, 0x00,
  /* 0x4C: 'L' */
  0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x1F, 0x00,
  /* 0x4D: 'M' */
  0x11, 0x1B, 0x15, 0x15, 0x15, 0x11, 0x11, 0x00,
  /* 0x4E: 'N' */
  0x11, 0x11, 0x13, 0x15, 0x19, 0x11, 0x11, 0x00,
  /* 0x4F: 'O' */
  0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00,
  /* 0x50: 'P' */
  0x0F, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x01, 0x00,
  /* 0x51: 'Q' */
  0x0E, 0x11, 0x11, 0x11, 0x15, 0x09, 0x16, 0x00,
  /* 0x52: 'R' */
  0x0F, 0x11, 0x11, 0x0F, 0x05, 0x09, 0x11, 0x00,
  /* 0x53: 'S' */
  0x0E, 0x11, 0x01, 0x0E, 0x10, 0x11, 0x0E, 0x00,
  /* 0x54: 'T' */
  0x1F, 0x15, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00,
  /* 0x55: 'U' */
  0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00,
  /* 0x56: 'V' */
  0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04, 0x00,
  /* 0x57: 'W' */
  0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A, 0x00,
  /* 0x58: 'X' */
  0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11, 0x00,
  /* 0x59: 'Y' */
  0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x00,
  /* 0x5A: 'Z' */
  0x1F, 0x10, 0x08, 0x0E, 0x02, 0x01, 0x1F, 0x00,
  /* 0x5B: '[' */
  0x1E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x1E, 0x00,
  /* 0x5C: '\' */
  0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00,
  /* 0x5D: ']' */
  0x1E, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1E, 0x00,
  /* 0x5E: '^' */
  0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00,
  /* 0x5F: '_' */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00,
  /* 0x60: ''' */
  0x06, 0x06, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00,
  /* 0x61: 'a' */
  0x00, 0x00, 0x06, 0x08, 0x0E, 0x09, 0x1E, 0x00,
  /* 0x62: 'b' */
  0x01, 0x01, 0x0D, 0x13, 0x11, 0x13, 0x0D, 0x00,
  /* 0x63: 'c' */
  0x00, 0x00, 0x0E, 0x11, 0x01, 0x11, 0x0E, 0x00,
  /* 0x64: 'd' */
  0x10, 0x10, 0x16, 0x19, 0x11, 0x19, 0x16, 0x00,
  /* 0x65: 'e' */
  0x00, 0x00, 0x0E, 0x11, 0x1F, 0x01, 0x0E, 0x00,
  /* 0x66: 'f' */
  0x08, 0x14, 0x04, 0x0E, 0x04, 0x04, 0x04, 0x00,
  /* 0x67: 'g' */
  0x00, 0x00, 0x0E, 0x19, 0x19, 0x16, 0x10, 0x0E,
  /* 0x68: 'h' */
  0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x11, 0x00,
  /* 0x69: 'i' */
  0x04, 0x00, 0x06, 0x04, 0x04, 0x04, 0x0E, 0x00,
  /* 0x6A: 'j' */
  0x08, 0x00, 0x08, 0x08, 0x08, 0x09, 0x06, 0x00,
  /* 0x6B: 'k' */
  0x01, 0x01, 0x09, 0x05, 0x03, 0x05, 0x09, 0x00,
  /* 0x6C: 'l' */
  0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00,
  /* 0x6D: 'm' */
  0x00, 0x00, 0x0B, 0x15, 0x15, 0x15, 0x15, 0x00,
  /* 0x6E: 'n' */
  0x00, 0x00, 0x0D, 0x13, 0x11, 0x11, 0x11, 0x00,
  /* 0x6F: 'o' */
  0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E, 0x00,
  /* 0x70: 'p' */
  0x00, 0x00, 0x0D, 0x13, 0x13, 0x0D, 0x01, 0x01,
  /* 0x71: 'q' */
  0x00, 0x00, 0x16, 0x19, 0x19, 0x16, 0x10, 0x10,
  /* 0x72: 'r' */
  0x00, 0x00, 0x0D, 0x13, 0x01, 0x01, 0x01, 0x00,
  /* 0x73: 's' */
  0x00, 0x00, 0x1E, 0x01, 0x0E, 0x10, 0x0F, 0x00,
  /* 0x74: 't' */
  0x04, 0x04, 0x1F, 0x04, 0x04, 0x14, 0x08, 0x00,
  /* 0x75: 'u' */
  0x00, 0x00, 0x11, 0x11, 0x11, 0x19, 0x16, 0x00,
  /* 0x76: 'v' */
  0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04, 0x00,
  /* 0x77: 'w' */
  0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A, 0x00,
  /* 0x78: 'x' */
  0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x00,
  /* 0x79: 'y' */
  0x00, 0x00, 0x11, 0x11, 0x1E, 0x10, 0x11, 0x0E,
  /* 0x7A: 'z' */
  0x00, 0x00, 0x1F, 0x08, 0x04, 0x02, 0x1F, 0x00,
  /* 0x7B: '{' */
  0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08, 0x00,
  /* 0x7C: '|' */
  0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04, 0x00,
  /* 0x7D: '}' */
  0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02, 0x00,
  /* 0x7E: '~' */
  0x02, 0x15, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
  /* 0x7F: ' ' */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

  /* Special Symbols  starting at character 0x80 */
  /* 0x80: Circle - Empty */
  0x00, 0x00, 0x0C, 0x12, 0x12, 0x0C, 0x00, 0x00,
  /* 0x81: Circle - Full */
  0x00, 0x00, 0x0C, 0x1E, 0x1E, 0x0C, 0x00, 0x00,
  /* 0x82: Square - Empty */
  0x00, 0x00, 0x1E, 0x12, 0x12, 0x1E, 0x00, 0x00,
  /* 0x83: Square - Full */
  0x00, 0x00, 0x1E, 0x1E, 0x1E, 0x1E, 0x00, 0x00,
  /* 0x84: Up - Empty */
  0x00, 0x00, 0x0C, 0x0C, 0x12, 0x1E, 0x00, 0x00,
  /* 0x85: Up - Full */
  0x00, 0x00, 0x0C, 0x0C, 0x1E, 0x1E, 0x00, 0x00,
  /* 0x86: Down - Empty */
  0x00, 0x00, 0x1E, 0x12, 0x0C, 0x0C, 0x00, 0x00,
  /* 0x87: Down - Full */
  0x00, 0x00, 0x1E, 0x1E, 0x0C, 0x0C, 0x00, 0x00,
  /* 0x88: Left - Empty */
  0x00, 0x00, 0x18, 0x16, 0x16, 0x18, 0x00, 0x00,
  /* 0x89: Left - Full */
  0x00, 0x00, 0x18, 0x1E, 0x1E, 0x18, 0x00, 0x00,
  /* 0x8A: Right - Empty */
  0x00, 0x00, 0x06, 0x1A, 0x1A, 0x06, 0x00, 0x00,
  /* 0x8B: Right - Full */
  0x00, 0x00, 0x06, 0x1E, 0x1E, 0x06, 0x00, 0x00,
  /* 0x8C: Wait - Empty */
  0x00, 0x00, 0x0C, 0x12, 0x12, 0x0C, 0x00, 0x00,
  /* 0x8D: Wait - Full */
  0x00, 0x00, 0x0C, 0x1E, 0x1E, 0x0C, 0x00, 0x00,
  /* 0x8E: Walk - Empty */
  0x00, 0x00, 0x1E, 0x12, 0x12, 0x1E, 0x00, 0x00,
  /* 0x8F: Walk - Full */
  0x00, 0x00, 0x1E, 0x1E, 0x1E, 0x1E, 0x00, 0x00,
};

const unsigned short Font_16x24_h[(144-32)*24] = {
  /* 0x20: Space ' ' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x21: '!' */
  0x0000, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0000, 0x0000,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x22: '"' */
  0x0000, 0x0000, 0x00CC, 0x00CC, 0x00CC, 0x00CC, 0x00CC, 0x00CC,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x23: '#' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0C60, 0x0C60,
  0x0C60, 0x0630, 0x0630, 0x1FFE, 0x1FFE, 0x0630, 0x0738, 0x0318,
  0x1FFE, 0x1FFE, 0x0318, 0x0318, 0x018C, 0x018C, 0x018C, 0x0000,
  /* 0x24: '$' */
  0x0000, 0x0080, 0x03E0, 0x0FF8, 0x0E9C, 0x1C8C, 0x188C, 0x008C,
  0x0098, 0x01F8, 0x07E0, 0x0E80, 0x1C80, 0x188C, 0x188C, 0x189C,
  0x0CB8, 0x0FF0, 0x03E0, 0x0080, 0x0080, 0x0000, 0x0000, 0x0000,
  /* 0x25: '%' */
  0x0000, 0x0000, 0x0000, 0x180E, 0x0C1B, 0x0C11, 0x0611, 0x0611,
  0x0311, 0x0311, 0x019B, 0x018E, 0x38C0, 0x6CC0, 0x4460, 0x4460,
  0x4430, 0x4430, 0x4418, 0x6C18, 0x380C, 0x0000, 0x0000, 0x0000,
  /* 0x26: '&' */
  0x0000, 0x01E0, 0x03F0, 0x0738, 0x0618, 0x0618, 0x0330, 0x01F0,
  0x00F0, 0x00F8, 0x319C, 0x330E, 0x1E06, 0x1C06, 0x1C06, 0x3F06,
  0x73FC, 0x21F0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x27: ''' */
  0x0000, 0x0000, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x28: '(' */
  0x0000, 0x0200, 0x0300, 0x0180, 0x00C0, 0x00C0, 0x0060, 0x0060,
  0x0030, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030,
  0x0060, 0x0060, 0x00C0, 0x00C0, 0x0180, 0x0300, 0x0200, 0x0000,
  /* 0x29: ')' */
  0x0000, 0x0020, 0x0060, 0x00C0, 0x0180, 0x0180, 0x0300, 0x0300,
  0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600,
  0x0300, 0x0300, 0x0180, 0x0180, 0x00C0, 0x0060, 0x0020, 0x0000,
  /* 0x2A: '*' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x00C0, 0x00C0,
  0x06D8, 0x07F8, 0x01E0, 0x0330, 0x0738, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x2B: '+' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x3FFC, 0x3FFC, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x2C: ',' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0180, 0x0180, 0x0100, 0x0100, 0x0080, 0x0000, 0x0000,
  /* 0x2D: '-' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x07E0, 0x07E0, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x2E: '.' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x2F: '/' */
  0x0000, 0x0C00, 0x0C00, 0x0600, 0x0600, 0x0600, 0x0300, 0x0300,
  0x0300, 0x0380, 0x0180, 0x0180, 0x0180, 0x00C0, 0x00C0, 0x00C0,
  0x0060, 0x0060, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x30: '0' */
  0x0000, 0x03E0, 0x07F0, 0x0E38, 0x0C18, 0x180C, 0x180C, 0x180C,
  0x180C, 0x180C, 0x180C, 0x180C, 0x180C, 0x180C, 0x0C18, 0x0E38,
  0x07F0, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x31: '1' */
  0x0000, 0x0100, 0x0180, 0x01C0, 0x01F0, 0x0198, 0x0188, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x32: '2' */
  0x0000, 0x03E0, 0x0FF8, 0x0C18, 0x180C, 0x180C, 0x1800, 0x1800,
  0x0C00, 0x0600, 0x0300, 0x0180, 0x00C0, 0x0060, 0x0030, 0x0018,
  0x1FFC, 0x1FFC, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x33: '3' */
  0x0000, 0x01E0, 0x07F8, 0x0E18, 0x0C0C, 0x0C0C, 0x0C00, 0x0600,
  0x03C0, 0x07C0, 0x0C00, 0x1800, 0x1800, 0x180C, 0x180C, 0x0C18,
  0x07F8, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x34: '4' */
  0x0000, 0x0C00, 0x0E00, 0x0F00, 0x0F00, 0x0D80, 0x0CC0, 0x0C60,
  0x0C60, 0x0C30, 0x0C18, 0x0C0C, 0x3FFC, 0x3FFC, 0x0C00, 0x0C00,
  0x0C00, 0x0C00, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x35: '5' */
  0x0000, 0x0FF8, 0x0FF8, 0x0018, 0x0018, 0x000C, 0x03EC, 0x07FC,
  0x0E1C, 0x1C00, 0x1800, 0x1800, 0x1800, 0x180C, 0x0C1C, 0x0E18,
  0x07F8, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x36: '6' */
  0x0000, 0x07C0, 0x0FF0, 0x1C38, 0x1818, 0x0018, 0x000C, 0x03CC,
  0x0FEC, 0x0E3C, 0x1C1C, 0x180C, 0x180C, 0x180C, 0x1C18, 0x0E38,
  0x07F0, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x37: '7' */
  0x0000, 0x1FFC, 0x1FFC, 0x0C00, 0x0600, 0x0600, 0x0300, 0x0380,
  0x0180, 0x01C0, 0x00C0, 0x00E0, 0x0060, 0x0060, 0x0070, 0x0030,
  0x0030, 0x0030, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x38: '8' */
  0x0000, 0x03E0, 0x07F0, 0x0E38, 0x0C18, 0x0C18, 0x0C18, 0x0638,
  0x07F0, 0x07F0, 0x0C18, 0x180C, 0x180C, 0x180C, 0x180C, 0x0C38,
  0x0FF8, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x39: '9' */
  0x0000, 0x03E0, 0x07F0, 0x0E38, 0x0C1C, 0x180C, 0x180C, 0x180C,
  0x1C1C, 0x1E38, 0x1BF8, 0x19E0, 0x1800, 0x0C00, 0x0C00, 0x0E1C,
  0x07F8, 0x01F0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x3A: ':' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0180, 0x0180,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x3B: ';' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0180, 0x0180,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0180, 0x0180, 0x0100, 0x0100, 0x0080, 0x0000, 0x0000, 0x0000,
  /* 0x3C: '<' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x1000, 0x1C00, 0x0F80, 0x03E0, 0x00F8, 0x0018, 0x00F8, 0x03E0,
  0x0F80, 0x1C00, 0x1000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x3D: '=' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x1FF8, 0x0000, 0x0000, 0x0000, 0x1FF8, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x3E: '>' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0008, 0x0038, 0x01F0, 0x07C0, 0x1F00, 0x1800, 0x1F00, 0x07C0,
  0x01F0, 0x0038, 0x0008, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x3F: '?' */
  0x0000, 0x03E0, 0x0FF8, 0x0C18, 0x180C, 0x180C, 0x1800, 0x0C00,
  0x0600, 0x0300, 0x0180, 0x00C0, 0x00C0, 0x00C0, 0x0000, 0x0000,
  0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x40: '@' */
  0x0000, 0x0000, 0x07E0, 0x1818, 0x2004, 0x29C2, 0x4A22, 0x4411,
  0x4409, 0x4409, 0x4409, 0x2209, 0x1311, 0x0CE2, 0x4002, 0x2004,
  0x1818, 0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x41: 'A' */
  0x0000, 0x0380, 0x0380, 0x06C0, 0x06C0, 0x06C0, 0x0C60, 0x0C60,
  0x1830, 0x1830, 0x1830, 0x3FF8, 0x3FF8, 0x701C, 0x600C, 0x600C,
  0xC006, 0xC006, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x42: 'B' */
  0x0000, 0x03FC, 0x0FFC, 0x0C0C, 0x180C, 0x180C, 0x180C, 0x0C0C,
  0x07FC, 0x0FFC, 0x180C, 0x300C, 0x300C, 0x300C, 0x300C, 0x180C,
  0x1FFC, 0x07FC, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x43: 'C' */
  0x0000, 0x07C0, 0x1FF0, 0x3838, 0x301C, 0x700C, 0x6006, 0x0006,
  0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x6006, 0x700C, 0x301C,
  0x1FF0, 0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x44: 'D' */
  0x0000, 0x03FE, 0x0FFE, 0x0E06, 0x1806, 0x1806, 0x3006, 0x3006,
  0x3006, 0x3006, 0x3006, 0x3006, 0x3006, 0x1806, 0x1806, 0x0E06,
  0x0FFE, 0x03FE, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x45: 'E' */
  0x0000, 0x3FFC, 0x3FFC, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C,
  0x1FFC, 0x1FFC, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C,
  0x3FFC, 0x3FFC, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x46: 'F' */
  0x0000, 0x3FF8, 0x3FF8, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018,
  0x1FF8, 0x1FF8, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018,
  0x0018, 0x0018, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x47: 'G' */
  0x0000, 0x0FE0, 0x3FF8, 0x783C, 0x600E, 0xE006, 0xC007, 0x0003,
  0x0003, 0xFE03, 0xFE03, 0xC003, 0xC007, 0xC006, 0xC00E, 0xF03C,
  0x3FF8, 0x0FE0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x48: 'H' */
  0x0000, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C,
  0x3FFC, 0x3FFC, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C,
  0x300C, 0x300C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x49: 'I' */
  0x0000, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4A: 'J' */
  0x0000, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0600,
  0x0600, 0x0600, 0x0600, 0x0600, 0x0600, 0x0618, 0x0618, 0x0738,
  0x03F0, 0x01E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4B: 'K' */
  0x0000, 0x3006, 0x1806, 0x0C06, 0x0606, 0x0306, 0x0186, 0x00C6,
  0x0066, 0x0076, 0x00DE, 0x018E, 0x0306, 0x0606, 0x0C06, 0x1806,
  0x3006, 0x6006, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4C: 'L' */
  0x0000, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018,
  0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018,
  0x1FF8, 0x1FF8, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4D: 'M' */
  0x0000, 0xE00E, 0xF01E, 0xF01E, 0xF01E, 0xD836, 0xD836, 0xD836,
  0xD836, 0xCC66, 0xCC66, 0xCC66, 0xC6C6, 0xC6C6, 0xC6C6, 0xC6C6,
  0xC386, 0xC386, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4E: 'N' */
  0x0000, 0x300C, 0x301C, 0x303C, 0x303C, 0x306C, 0x306C, 0x30CC,
  0x30CC, 0x318C, 0x330C, 0x330C, 0x360C, 0x360C, 0x3C0C, 0x3C0C,
  0x380C, 0x300C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x4F: 'O' */
  0x0000, 0x07E0, 0x1FF8, 0x381C, 0x700E, 0x6006, 0xC003, 0xC003,
  0xC003, 0xC003, 0xC003, 0xC003, 0xC003, 0x6006, 0x700E, 0x381C,
  0x1FF8, 0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x50: 'P' */
  0x0000, 0x0FFC, 0x1FFC, 0x380C, 0x300C, 0x300C, 0x300C, 0x300C,
  0x180C, 0x1FFC, 0x07FC, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C,
  0x000C, 0x000C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x51: 'Q' */
  0x0000, 0x07E0, 0x1FF8, 0x381C, 0x700E, 0x6006, 0xE003, 0xC003,
  0xC003, 0xC003, 0xC003, 0xC003, 0xE007, 0x6306, 0x3F0E, 0x3C1C,
  0x3FF8, 0xF7E0, 0xC000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x52: 'R' */
  0x0000, 0x0FFE, 0x1FFE, 0x3806, 0x3006, 0x3006, 0x3006, 0x3806,
  0x1FFE, 0x07FE, 0x0306, 0x0606, 0x0C06, 0x1806, 0x1806, 0x3006,
  0x3006, 0x6006, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x53: 'S' */
  0x0000, 0x03E0, 0x0FF8, 0x0C1C, 0x180C, 0x180C, 0x000C, 0x001C,
  0x03F8, 0x0FE0, 0x1E00, 0x3800, 0x3006, 0x3006, 0x300E, 0x1C1C,
  0x0FF8, 0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x54: 'T' */
  0x0000, 0x7FFE, 0x7FFE, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x55: 'U' */
  0x0000, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C,
  0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x300C, 0x1818,
  0x1FF8, 0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x56: 'V' */
  0x0000, 0x6003, 0x3006, 0x3006, 0x3006, 0x180C, 0x180C, 0x180C,
  0x0C18, 0x0C18, 0x0E38, 0x0630, 0x0630, 0x0770, 0x0360, 0x0360,
  0x01C0, 0x01C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x57: 'W' */
  0x0000, 0x6003, 0x61C3, 0x61C3, 0x61C3, 0x3366, 0x3366, 0x3366,
  0x3366, 0x3366, 0x3366, 0x1B6C, 0x1B6C, 0x1B6C, 0x1A2C, 0x1E3C,
  0x0E38, 0x0E38, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x58: 'X' */
  0x0000, 0xE00F, 0x700C, 0x3018, 0x1830, 0x0C70, 0x0E60, 0x07C0,
  0x0380, 0x0380, 0x03C0, 0x06E0, 0x0C70, 0x1C30, 0x1818, 0x300C,
  0x600E, 0xE007, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x59: 'Y' */
  0x0000, 0xC003, 0x6006, 0x300C, 0x381C, 0x1838, 0x0C30, 0x0660,
  0x07E0, 0x03C0, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x5A: 'Z' */
  0x0000, 0x7FFC, 0x7FFC, 0x6000, 0x3000, 0x1800, 0x0C00, 0x0600,
  0x0300, 0x0180, 0x00C0, 0x0060, 0x0030, 0x0018, 0x000C, 0x0006,
  0x7FFE, 0x7FFE, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x5B: '[' */
  0x0000, 0x03E0, 0x03E0, 0x0060, 0x0060, 0x0060, 0x0060, 0x0060,
  0x0060, 0x0060, 0x0060, 0x0060, 0x0060, 0x0060, 0x0060, 0x0060,
  0x0060, 0x0060, 0x0060, 0x0060, 0x0060, 0x03E0, 0x03E0, 0x0000,
  /* 0x5C: '\' */
  0x0000, 0x0030, 0x0030, 0x0060, 0x0060, 0x0060, 0x00C0, 0x00C0,
  0x00C0, 0x01C0, 0x0180, 0x0180, 0x0180, 0x0300, 0x0300, 0x0300,
  0x0600, 0x0600, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x5D: ']' */
  0x0000, 0x03E0, 0x03E0, 0x0300, 0x0300, 0x0300, 0x0300, 0x0300,
  0x0300, 0x0300, 0x0300, 0x0300, 0x0300, 0x0300, 0x0300, 0x0300,
  0x0300, 0x0300, 0x0300, 0x0300, 0x0300, 0x03E0, 0x03E0, 0x0000,
  /* 0x5E: '^' */
  0x0000, 0x0000, 0x01C0, 0x01C0, 0x0360, 0x0360, 0x0360, 0x0630,
  0x0630, 0x0C18, 0x0C18, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x5F: '_' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0xFFFF, 0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x60: ''' */
  0x0000, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x61: 'a' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03F0, 0x07F8,
  0x0C1C, 0x0C0C, 0x0F00, 0x0FF0, 0x0CF8, 0x0C0C, 0x0C0C, 0x0F1C,
  0x0FF8, 0x18F0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x62: 'b' */
  0x0000, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x03D8, 0x0FF8,
  0x0C38, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x0C38,
  0x0FF8, 0x03D8, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x63: 'c' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03C0, 0x07F0,
  0x0E30, 0x0C18, 0x0018, 0x0018, 0x0018, 0x0018, 0x0C18, 0x0E30,
  0x07F0, 0x03C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x64: 'd' */
  0x0000, 0x1800, 0x1800, 0x1800, 0x1800, 0x1800, 0x1BC0, 0x1FF0,
  0x1C30, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1C30,
  0x1FF0, 0x1BC0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x65: 'e' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03C0, 0x0FF0,
  0x0C30, 0x1818, 0x1FF8, 0x1FF8, 0x0018, 0x0018, 0x1838, 0x1C30,
  0x0FF0, 0x07C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x66: 'f' */
  0x0000, 0x0F80, 0x0FC0, 0x00C0, 0x00C0, 0x00C0, 0x07F0, 0x07F0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x67: 'g' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0DE0, 0x0FF8,
  0x0E18, 0x0C0C, 0x0C0C, 0x0C0C, 0x0C0C, 0x0C0C, 0x0C0C, 0x0E18,
  0x0FF8, 0x0DE0, 0x0C00, 0x0C0C, 0x061C, 0x07F8, 0x01F0, 0x0000,
  /* 0x68: 'h' */
  0x0000, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x07D8, 0x0FF8,
  0x1C38, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818,
  0x1818, 0x1818, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x69: 'i' */
  0x0000, 0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x6A: 'j' */
  0x0000, 0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00F8, 0x0078, 0x0000,
  /* 0x6B: 'k' */
  0x0000, 0x000C, 0x000C, 0x000C, 0x000C, 0x000C, 0x0C0C, 0x060C,
  0x030C, 0x018C, 0x00CC, 0x006C, 0x00FC, 0x019C, 0x038C, 0x030C,
  0x060C, 0x0C0C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x6C: 'l' */
  0x0000, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x6D: 'm' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3C7C, 0x7EFF,
  0xE3C7, 0xC183, 0xC183, 0xC183, 0xC183, 0xC183, 0xC183, 0xC183,
  0xC183, 0xC183, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x6E: 'n' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0798, 0x0FF8,
  0x1C38, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818,
  0x1818, 0x1818, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x6F: 'o' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03C0, 0x0FF0,
  0x0C30, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x0C30,
  0x0FF0, 0x03C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x70: 'p' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03D8, 0x0FF8,
  0x0C38, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x0C38,
  0x0FF8, 0x03D8, 0x0018, 0x0018, 0x0018, 0x0018, 0x0018, 0x0000,
  /* 0x71: 'q' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x1BC0, 0x1FF0,
  0x1C30, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1C30,
  0x1FF0, 0x1BC0, 0x1800, 0x1800, 0x1800, 0x1800, 0x1800, 0x0000,
  /* 0x72: 'r' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x07B0, 0x03F0,
  0x0070, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030, 0x0030,
  0x0030, 0x0030, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x73: 's' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03E0, 0x03F0,
  0x0E38, 0x0C18, 0x0038, 0x03F0, 0x07C0, 0x0C00, 0x0C18, 0x0E38,
  0x07F0, 0x03E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x74: 't' */
  0x0000, 0x0000, 0x0080, 0x00C0, 0x00C0, 0x00C0, 0x07F0, 0x07F0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x07C0, 0x0780, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x75: 'u' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x1818, 0x1818,
  0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1C38,
  0x1FF0, 0x19E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x76: 'v' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x180C, 0x0C18,
  0x0C18, 0x0C18, 0x0630, 0x0630, 0x0630, 0x0360, 0x0360, 0x0360,
  0x01C0, 0x01C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x77: 'w' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x41C1, 0x41C1,
  0x61C3, 0x6363, 0x6363, 0x6363, 0x3636, 0x3636, 0x3636, 0x1C1C,
  0x1C1C, 0x1C1C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x78: 'x' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x381C, 0x1C38,
  0x0C30, 0x0660, 0x03C0, 0x03C0, 0x03C0, 0x03C0, 0x0660, 0x0C30,
  0x1C38, 0x381C, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x79: 'y' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3018, 0x1830,
  0x1830, 0x1870, 0x0C60, 0x0C60, 0x0CE0, 0x06C0, 0x06C0, 0x0380,
  0x0380, 0x0380, 0x0180, 0x0180, 0x01C0, 0x00F0, 0x0070, 0x0000,
  /* 0x7A: 'z' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x1FFC, 0x1FFC,
  0x0C00, 0x0600, 0x0300, 0x0180, 0x00C0, 0x0060, 0x0030, 0x0018,
  0x1FFC, 0x1FFC, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x7B: '{' */
  0x0000, 0x0300, 0x0180, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0,
  0x00C0, 0x0060, 0x0060, 0x0030, 0x0060, 0x0040, 0x00C0, 0x00C0,
  0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x0180, 0x0300, 0x0000, 0x0000,
  /* 0x7C: '|' */
  0x0000, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0000,
  /* 0x7D: '}' */
  0x0000, 0x0060, 0x00C0, 0x01C0, 0x0180, 0x0180, 0x0180, 0x0180,
  0x0180, 0x0300, 0x0300, 0x0600, 0x0300, 0x0100, 0x0180, 0x0180,
  0x0180, 0x0180, 0x0180, 0x0180, 0x00C0, 0x0060, 0x0000, 0x0000,
  /* 0x7E: '~' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x10F0, 0x1FF8, 0x0F08, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x7F: ' ' */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,

  /* Special Symbols  starting at character 0x80 */
  /* 0x80: Circle - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03C0, 0x0C30, 0x1008,
  0x2004, 0x2004, 0x4002, 0x4002, 0x4002, 0x4002, 0x4002, 0x2004,
  0x2004, 0x1008, 0x0C30, 0x03C0, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x81: Circle - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x03C0, 0x0FF0, 0x1FF8,
  0x3FFC, 0x3FFC, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x3FFC,
  0x3FFC, 0x1FF8, 0x0FF0, 0x03C0, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x82: Square - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x07E0,
  0x0FF0, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x1818, 0x0FF0,
  0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x83: Square - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x07E0,
  0x0FF0, 0x1FF8, 0x1FF8, 0x1FF8, 0x1FF8, 0x1FF8, 0x1FF8, 0x0FF0,
  0x07E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x84: Up - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0180, 0x03C0, 0x0660, 0x0C30,
  0x1818, 0x1818, 0x1FF8, 0x1FF8, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x85: Up - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0180, 0x03C0, 0x07E0, 0x0FF0,
  0x1FF8, 0x1FF8, 0x1FF8, 0x1FF8, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x86: Down - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x1FF8, 0x1FF8, 0x1818, 0x1818,
  0x0C30, 0x0660, 0x03C0, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x87: Down - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x1FF8, 0x1FF8, 0x1FF8, 0x1FF8,
  0x0FF0, 0x07E0, 0x03C0, 0x0180, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x88: Left - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x01E0,
  0x01F0, 0x0198, 0x018C, 0x0186, 0x0186, 0x018C, 0x0198, 0x01F0,
  0x01E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x89: Left - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x01E0,
  0x01F0, 0x01F8, 0x01FC, 0x01FE, 0x01FE, 0x01FC, 0x01F8, 0x01F0,
  0x01E0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x8A: Right - Empty */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0780,
  0x0F80, 0x1980, 0x3180, 0x6180, 0x6180, 0x3180, 0x1980, 0x0F80,
  0x0780, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x8B: Right - Full */
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0780,
  0x0F80, 0x1F80, 0x3F80, 0x7F80, 0x7F80, 0x3F80, 0x1F80, 0x0F80,
  0x0780, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  /* 0x8C: Wait - Empty */
  0x0000, 0x01C0, 0x0220, 0x0220, 0x0140, 0x0630, 0x0808, 0x0808,
  0x0808, 0x0808, 0x0808, 0x0808, 0x0808, 0x0220, 0x0220, 0x0220,
  0x0220, 0x0220, 0x0220, 0x0220, 0x0220, 0x0220, 0x0220, 0x0000,
  /* 0x8D: Wait - Full */
  0x0000, 0x01C0, 0x03E0, 0x03E0, 0x01C0, 0x07F0, 0x0DD8, 0x0DD8,
  0x0DD8, 0x0DD8, 0x0DD8, 0x0DD8, 0x0DD8, 0x0360, 0x0360, 0x0360,
  0x0360, 0x0360, 0x0360, 0x0360, 0x0360, 0x0360, 0x0360, 0x0000,
  /* 0x8E: Walk - Empty */
  0x0000, 0x01C0, 0x0220, 0x0220, 0x0140, 0x0630, 0x0808, 0x0808,
  0x0808, 0x1004, 0x2002, 0x2002, 0x0140, 0x0220, 0x0220, 0x0410,
  0x0808, 0x0808, 0x1004, 0x1004, 0x2004, 0x4004, 0x0000, 0x0000,
  /* 0x8F: Walk - Full */
  0x0000, 0x01C0, 0x03E0, 0x03E0, 0x01C0, 0x07F0, 0x0DD8, 0x0DD8,
  0x0DD8, 0x19CC, 0x31C6, 0x61C2, 0x01C0, 0x0360, 0x0360, 0x0670,
  0x0C38, 0x0C18, 0x180C, 0x180C, 0x300C, 0x600C, 0x0000, 0x0000,
};

