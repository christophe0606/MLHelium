#include "mps3_audio_an547.h"

/*
 * Copyright (c) 2021, Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#include "SSE300MPS3.h"
#include "cmsis_driver_config.h"
#include "timeout.h"
#include "device_definition.h"
//#include "system_core_init.h" /* for version 1.2 of CMSIS-PACK */
#include "i2c_sbcon_drv.h"
#include "audio_i2s_mps3_drv.h"

#define CHIP_ADDR_WRITE     0x96
#define CHIP_ADDR_READ      0x97
/**
 * \brief CS42L52 Audio Codec registers
 */
#define AUDIO_CODEC_MPS3_CHIP_ID     0x01 /*!< Chip ID and Revision Register */
#define AUDIO_CODEC_MPS3_PWR_CTRL1   0x02 /*!< Power Control 1 */
#define AUDIO_CODEC_MPS3_PWR_CTRL2   0x03 /*!< Power Control 2 */
#define AUDIO_CODEC_MPS3_PWR_CTRL3   0x04 /*!< Power Control 3 */
#define AUDIO_CODEC_MPS3_CLK_CTRL    0x05 /*!< Clocking Control */
#define AUDIO_CODEC_MPS3_INT_CTRL1   0x06 /*!< Interface Control 1 */
#define AUDIO_CODEC_MPS3_INT_CTRL2   0x07 /*!< Interface Control 2 */
#define AUDIO_CODEC_MPS3_INPUT_A     0x08 /*!< Input x Select: ADCA and PGAA */
#define AUDIO_CODEC_MPS3_INPUT_B     0x09 /*!< Input x Select: ADCB and PGAB */
#define AUDIO_CODEC_MPS3_ADC_CTRL    0x0C /*!< Input x Select: ADCB and PGAB */
#define AUDIO_CODEC_MPS3_PLAYBACK_CTRL    0x0F /*!< Playback ctrl */

#define AUDIO_CODEC_MPS3_AMP_A       0x10 /*!< MICx Amp Control:MIC A */
#define AUDIO_CODEC_MPS3_AMP_B       0x11 /*!< MICx Amp Control:MIC B */
#define AUDIO_CODEC_MPS3_MISC_CTRL   0x0E /*!< Miscellaneous Controls */

#define AUDIO_CODEC_MPS3_MIXER       0x26 /*!<  ADC PCM Mixer */
#define AUDIO_CODEC_MPS3_ALC         0x2A /*!< Audio Level Control */
#define AUDIO_CODEC_MPS3_NOISE_GATE  0x2D /*!< Audio Level Control */

static enum audio_codec_mps3_error_t audio_codec_mps3_write(uint8_t map_byte, uint8_t data)
{
    struct i2c_sbcon_dev_t* i2c_sbcon_dev = &I2C0_SBCON_DEV;
    uint32_t i;
    uint8_t to_write[2];
    to_write[0] = map_byte;
    to_write[1] = data;
    i2c_sbcon_master_transmit(i2c_sbcon_dev, CHIP_ADDR_WRITE, to_write, 2, 0, &i);
    return AUDIO_CODEC_MPS3_ERR_NONE;
}
static uint8_t audio_codec_mps3_read(uint8_t map_byte)
{
    struct i2c_sbcon_dev_t* i2c_sbcon_dev = &I2C0_SBCON_DEV;
    uint32_t i;
    uint8_t data;
    i2c_sbcon_master_transmit(i2c_sbcon_dev, CHIP_ADDR_WRITE, &map_byte, 1, 0, &i);
    i2c_sbcon_master_receive(i2c_sbcon_dev, CHIP_ADDR_READ, &data, 1, 0, &i);
    return data;
}
enum audio_codec_mps3_error_t audio_codec_mps3_init(void)
{
    struct audio_i2s_mps3_dev_t* audio_i2s_mps3_dev = &MPS3_I2S_DEV;
    struct i2c_sbcon_dev_t* i2c_sbcon_dev = &I2C0_SBCON_DEV;
    uint8_t reg_32;
    i2c_sbcon_init(i2c_sbcon_dev, SystemCoreClock);
    audio_i2s_mps3_set_codec_reset(audio_i2s_mps3_dev);
    wait_ms(1);
    audio_i2s_mps3_clear_codec_reset(audio_i2s_mps3_dev);
    wait_ms(1);
    /* Initialization with values given in the Reference Manual */
    
    audio_codec_mps3_write(0x00, 0x99);
    audio_codec_mps3_write(0x3E, 0xBA);
    audio_codec_mps3_write(0x47, 0x80);
    //reg_32 = audio_codec_mps3_read(0x32);
    //audio_codec_mps3_write(0x32, reg_32 | 0x80);
    //audio_codec_mps3_write(0x32, reg_32 & 0x7F);
    (void)reg_32;

    audio_codec_mps3_write(0x32, 0xBB);
    audio_codec_mps3_write(0x32, 0x3B);

    audio_codec_mps3_write(0x00, 0x00);
    wait_ms(1);
    /* Single-speed mode */
    // Enable MCLK and set frequency (LRCK=48KHz, MCLK=12.288MHz, /256)
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_CLK_CTRL, 0xA0); // MODIFIED
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_CLK_CTRL, 0x80); // MODIFIED
    
    /* Disable ALC */
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_ALC, 0x00);
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_NOISE_GATE,0b11110100);


    /* ADC charge pump and PGA & ADC channels powered up */
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL1, 0x00);
    /* MIC powered up */
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL2, 0x00);
    /* Headphone and Speaker channel always on */
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL3, 0xAA);
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL3, 0xAA);

    /* Select analog input for PGA AIN4A and AIN4B */
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_INPUT_A, 0x00); // MODIFIED
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_INPUT_B, 0x00); // MODIFIED
    /* Select MIC inputs and sets microphone pre-amplifier 32 dB */
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_AMP_A, 0x5F);  // Optional
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_AMP_B, 0x5F);  // Optional
    /* De-emphasis filter enabled */
    //audio_codec_mps3_write(AUDIO_CODEC_MPS3_MISC_CTRL, 0x04);

    
    wait_ms(1);
    return AUDIO_CODEC_MPS3_ERR_NONE;
}

/************************************************************************/
/* The Audio codec has I2C and I2S interfaces from the FPGA             */
/* The IC2 interface is a simple GPIO interface and the AAIC_I2C_       */
/* software functions generate the correct I2C protocol.                */
/* The I2S is a simple FIFO buffer in the FPGA with a FIFO full         */
/* flag to indicate the FIFO status, the FIFO is shifted out            */
/* serially to the CODEC.                                               */
/************************************************************************/
void mps3_audio_init(int freq)
{
    // See power-up sequence (see DS680F2 page 37)
    // set resets
    audio_i2s_mps3_set_codec_reset(&MPS3_I2S_DEV);
    audio_i2s_mps3_set_fifo_reset(&MPS3_I2S_DEV);
    audio_i2s_mps3_enable_rxbuf(&MPS3_I2S_DEV);
    audio_i2s_mps3_enable_txbuf(&MPS3_I2S_DEV);
    //audio_i2s_mps3_enable_rxinterrupt(&MPS3_I2S_DEV);
    audio_i2s_mps3_enable_txinterrupt(&MPS3_I2S_DEV); 
    wait_ms(10);
    
    // Release AACI nRESET
    audio_i2s_mps3_clear_codec_reset(&MPS3_I2S_DEV);    
    wait_ms(100);

    // AACI clocks MCLK = 12.288MHz, SCLK = 3.072MHz, LRCLK = 48KHz
    // LRCLK divide ratio [9:0], 3.072MHz (SCLK) / 48KHz / 2 (L+R) = 32
    
    uint32_t lrdiv = (uint32_t)(3072000.0 / freq / 2);
    audio_i2s_mps3_speed_config(&MPS3_I2S_DEV,lrdiv);

    audio_codec_mps3_init();
    // Audio setup complete
    
    wait_ms(10);
  
    // Release I2S FIFO reset
    audio_i2s_mps3_clear_fifo_reset(&MPS3_I2S_DEV);
    
    // Make the audio interface interrupt based by registering I2S
    // at the NVIC controller
    NVIC_EnableIRQ(I2S_IRQn);

}

void mps3_audio_stop()
{
    NVIC_DisableIRQ(I2S_IRQn);
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL1, 0xFF);
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL2, 0xFF);
    audio_codec_mps3_write(AUDIO_CODEC_MPS3_PWR_CTRL3, 0xFF);

}