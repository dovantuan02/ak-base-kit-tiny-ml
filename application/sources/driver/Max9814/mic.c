#include <stdio.h>

#include "mic.h"
#include "app_dbg.h"

int mic_init(struct mic_pcm_t *mic, pf_read_adc pf_read) {
    if (pf_read == NULL) {
        return -1;
    }
    
    mic->bias = 0;
    mic->head = 0;
    mic->tail = 0;
    for (int i = 0; i < MIC_PCM_BUFFER_SIZE; i++) mic->buf[i] = 0;
    mic->pf_read = pf_read;
    return 0;
}

int mic_calibrate(struct mic_pcm_t *mic) {
    if (!mic->pf_read) {
        return -1;
    }

    uint32_t sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += mic->pf_read();
    }
    mic->bias = sum / 100;
    APP_DBG("Mic bias: %u\n", mic->bias);
    return 0;
}

uint16_t mic_available(struct mic_pcm_t *mic) {
    uint32_t h = mic->head;
    uint32_t t = mic->tail;
    if (h >= t) {
        return h - t;
    } else {
        return MIC_PCM_BUFFER_SIZE - t + h;
    }
}

int16_t mic_read(struct mic_pcm_t *mic) {
    while (mic_available(mic) == 0);
    int16_t sample = mic->buf[mic->tail];
    mic->tail = (mic->tail + 1) % MIC_PCM_BUFFER_SIZE;
    // APP_DBG("Sample: %d\n", sample);
    return sample;
}

int mic_timer_handle(struct mic_pcm_t *mic) {
    if (!mic->pf_read) {
        return -1;
    }
    uint16_t adc = mic->pf_read();
    int16_t pcm = (int16_t)(((int32_t)adc - mic->bias) * 16);
    uint32_t next_head = (mic->head + 1) % MIC_PCM_BUFFER_SIZE;
    if (next_head != mic->tail) {
        mic->buf[mic->head] = pcm;
		mic->head = next_head;
        return 0;
    }
    return -1;
}

