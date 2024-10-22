#if defined(MPS3)

#include "Board_GLCD.h"
#include "GLCD_Config.h"
#include "audio_init.h"

static int ln = 0;
static int col = 0;



void disp_token(const unsigned char *s)
{
  int maxh = (int)(GLCD_WIDTH / GLCD_FontWidth())-1;
  int maxv = (int)(GLCD_HEIGHT / GLCD_FontHeight())-1;
  while (*s) 
  { 
    if (*s == '\n')
    {
       ln++;
       s++;
       sam_process();
       reset_text();
    }
    else if (*s == '\r')
    {
       col = 0;
       s++;
    }
    else
    {
       if (*s > 0x8F)
       {
          return;
       }
       if (*s < 0x20)
       {
          return;
       }
       if ((ln == 0) && (col==0))
       {
          GLCD_ClearScreen();
       }
       add_char(*s);

       if ((*s == '.') 
           || (*s == '!')
           || (*s == '?'))
       {
         sam_process();
         reset_text();
       }
       
       GLCD_DisplayChar(ln,col++, *s++);

    }
    
    if (col == maxh)
    {
      col =0;
      ln++;
    }
    if (ln == maxv)
    { 
      ln = 0;
    }
  }
}

#endif