#include "audio_init.h"

#include "SSE300MPS3.h"
#include "cmsis_driver_config.h"
#include "mps3_audio_an547.h"
#include "i2c_sbcon_drv.h"
#include "audio_i2s_mps3_drv.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "error.h"

#include "reciter.h"
#include "sam.h"
#include <string.h>
#include <ctype.h>


__attribute__((section("dma")))                                                             
__ALIGNED(16) volatile uint16_t 
audio_buffer_output[FIFO_SIZE];

volatile int readPos,writePos;


volatile static int32_t pos;
volatile static int32_t speech_pos;
static uint32_t phase;
int debug;

#define NBSIN 110
static const int sinewave[NBSIN] = {      0.,   1886.,   3766.,   5634.,   7483.,   9307.,  11100.,
        12856.,  14570.,  16235.,  17847.,  19399.,  20887.,  22306.,
        23650.,  24917.,  26100.,  27197.,  28204.,  29118.,  29934.,
        30652.,  31268.,  31780.,  32187.,  32487.,  32679.,  32763.,
        32738.,  32605.,  32363.,  32015.,  31560.,  31000.,  30337.,
        29574.,  28713.,  27757.,  26708.,  25571.,  24350.,  23047.,
        21668.,  20217.,  18699.,  17119.,  15482.,  13794.,  12061.,
        10287.,   8479.,   6643.,   4785.,   2911.,   1027.,   -860.,
        -2744.,  -4619.,  -6479.,  -8317., -10128., -11905., -13642.,
       -15335., -16976., -18562., -20085., -21542., -22928., -24237.,
       -25466., -26611., -27668., -28632., -29502., -30274., -30945.,
       -31514., -31979., -32337., -32588., -32731., -32765., -32691.,
       -32508., -32218., -31820., -31317., -30711., -30002., -29194.,
       -28289., -27290., -26201., -25025., -23766., -22428., -21015.,
       -19533., -17987., -16380., -14719., -13010., -11257.,  -9467.,
        -7645.,  -5798.,  -3932.,  -2053.,   -167. };

volatile int16_t *pushBuffer(int nb)
{
   volatile uint16_t *r=NULL;
   if (readPos >= BUF_SIZE)
   {
     memmove((void*)audio_buffer_output,(void *)(audio_buffer_output+readPos),(writePos-readPos)*sizeof(int16_t));
     writePos -= readPos;
     readPos=0;
   }
   if (writePos+nb < FIFO_SIZE)
   {
      r = audio_buffer_output+writePos;
      writePos += nb;
   }
   return((volatile int16_t *)r);
}

int16_t getSample()
{
   // Empty FIFO
   if ((readPos+1) > writePos)
   {
     int16_t res = (int16_t)((float)rand() / (float)RAND_MAX * 0x800);
     return(res);
   }
   else
   {
     int16_t res = audio_buffer_output[readPos];
     readPos++;
     return(res);
   }
}

void gen_noise()
{

   volatile int16_t *src = pushBuffer(BUF_SIZE);
   for (int i = 0; i < BUF_SIZE; i++)
   {
      src[i] = (int16_t)((float)rand() / (float)RAND_MAX * 0x800);
   }

}

void gen_sin()
{
   volatile int16_t *src = pushBuffer(BUF_SIZE);
   for(int i = 0; i < BUF_SIZE; i++)
   {
         src[i] = sinewave[phase];
         phase++;
         if (phase == NBSIN)
         {
            phase=0;
         }
   }
}

void gen_sam()
{

   int8_t* src=(int8_t*)GetBuffer();
   int len = GetBufferLength();
   len /= 50;
   if (speech_pos+(BUF_SIZE>>1)>=len)
   {
      gen_noise();
      return;
   }

   volatile int16_t *dst = pushBuffer(BUF_SIZE);


   for(int i = 0; i < (BUF_SIZE>>1); i++)
   {
      dst[2*i] = src[i+speech_pos]<<7;
      dst[2*i+1] = src[i+speech_pos]<<7;
   }
   speech_pos += (BUF_SIZE>>1);
}

void audio_event(uint32_t event)
{
   if (event & AUDIO_DRV_EVENT_OUTPUT_DATA)
   {
      gen_sam();
   }
}



void I2S_Handler(void) 
{
  struct audio_i2s_mps3_sample_t audio_out_sample;


  uint16_t sample=getSample();
  audio_out_sample.left_channel=sample ^ 0x8000;
  audio_out_sample.right_channel=sample ^ 0x8000;


  write_sample(&MPS3_I2S_DEV,audio_out_sample);

  pos++;
  if (pos==BUF_SIZE)
  {
     audio_event(AUDIO_DRV_EVENT_OUTPUT_DATA);
     pos = 0;
  }


}

unsigned char* myinput;

void init_sam(char *str)
{
  SetPitch(70); //59 70
  SetSpeed(66);
  
  memset(myinput,0,256);
  for(int i=0; str[i] != 0; i++)
  {
      myinput[i] = (unsigned char)toupper((int)str[i]);
  }
 
  strcat((char*)myinput, "[");
  int err = TextToPhonemes(myinput);
  if (err==1)
  {
      SetInput(myinput);
      err=SAMMain();
  }
}

void audio_init()
{

   phase = 0;
   readPos = 0;
   writePos=0;
   pos=0;
   speech_pos=0;
   //gen_sin();


   SAMInit();
   myinput=(unsigned char*)malloc(256);

   init_sam("This is a test");
   gen_sam();


   mps3_audio_init(INPUT_SAMPLING_FREQ);
 
}

void audio_stop()
{
   

   mps3_audio_stop();
   free(myinput);



}