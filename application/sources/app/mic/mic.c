/**
 ******************************************************************************
 * @file    mic.c
 * @brief   MIC analog PCM sampling at 16kHz using TIM6 + interrupt
 ******************************************************************************
 */
#include "mic.h"
#include "io_cfg.h"
#include "app_dbg.h"

/* TIM6 clock: APB1 = 32MHz
 * Target: 16000 Hz (period 62.5 us)
 * Prescaler: 32 -> 1 MHz (1 us tick)
 * Period: 62 -> 1 MHz / (62 + 1) = 15873 Hz (~16kHz)
 */
#define TIM_PRESCALER       (32 - 1)
#define TIM_PERIOD          62

mic_pcm_t mic_pcm;
static uint16_t mic_bias;

static void mic_calibrate(void) {
    uint32_t sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += ADC_GetConversionValue(ADC1);
    }
    mic_bias = sum / 100;
}

void mic_init(void) {
    GPIO_InitTypeDef GPIO_InitStructure;
    TIM_TimeBaseInitTypeDef TIM_InitStructure;
    NVIC_InitTypeDef NVIC_InitStructure;

    /* Init GPIO analog for MIC */
    RCC_AHBPeriphClockCmd(A0_ADC_IO_CLOCK, ENABLE);
    GPIO_InitStructure.GPIO_Pin = A0_ADC_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AN;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
    GPIO_Init(A0_ADC_PORT, &GPIO_InitStructure);

    /* Init ADC1 */
    io_cfg_adc1();

    /* Init ring buffer */
    mic_pcm.head = 0;
    mic_pcm.tail = 0;

    /* Init TIM6 for 16kHz sampling */
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6, ENABLE);
    TIM_InitStructure.TIM_Prescaler = TIM_PRESCALER;
    TIM_InitStructure.TIM_Period = TIM_PERIOD;
    TIM_InitStructure.TIM_ClockDivision = 0;
    TIM_InitStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM6, &TIM_InitStructure);

    TIM_ITConfig(TIM6, TIM_IT_Update, ENABLE);

    NVIC_InitStructure.NVIC_IRQChannel = TIM6_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
    mic_bias = 0;
    mic_calibrate();
    APP_DBG("Use bias: %d\n", mic_bias);
}

void mic_start(void) {
    TIM_Cmd(TIM6, ENABLE);
}

void mic_stop(void) {
    TIM_Cmd(TIM6, DISABLE);
}

uint32_t mic_available(void) {
    uint32_t h = mic_pcm.head;
    uint32_t t = mic_pcm.tail;
    if (h >= t) {
        return h - t;
    } else {
        return MIC_PCM_BUFFER_SIZE - t + h;
    }
}

int16_t mic_read_sample(void) {
    while (mic_available() == 0);
    int16_t sample = mic_pcm.buf[mic_pcm.tail];
    mic_pcm.tail = (mic_pcm.tail + 1) % MIC_PCM_BUFFER_SIZE;
    return sample;
}

bool mic_insert_sample(uint16_t adc) {
    int16_t pcm = (int16_t)(((int32_t)adc - mic_bias) * 16);
    uint32_t next_head = (mic_pcm.head + 1) % MIC_PCM_BUFFER_SIZE;
    if (next_head != mic_pcm.tail) {
        mic_pcm.buf[mic_pcm.head] = pcm;
		mic_pcm.head = next_head;
        return true;
    }
    return false;
}

