#ifndef _AUDIO_INIT_H_
#define _AUDIO_INIT_H_ 


#include <stdint.h>

#define INPUT_SAMPLING_FREQ 48000
#define BUF_SIZE 480 // 10 ms
#define FIFO_SIZE (500*BUF_SIZE)

#define AUDIO_DRV_EVENT_OUTPUT_DATA         (1UL << 0)  ///< Data block transmitted
#define AUDIO_DRV_EVENT_INPUT_DATA    (1UL << 1)  ///< Data block received for far end

#ifdef   __cplusplus
extern "C"
{
#endif

extern volatile uint16_t *audio_buffer_output;


extern void audio_init();
extern void audio_stop();

extern void reset_text();
extern void add_text(const char *txt);
extern void add_char(const char c);
extern void sam_process();


#ifdef   __cplusplus
}
#endif

#endif