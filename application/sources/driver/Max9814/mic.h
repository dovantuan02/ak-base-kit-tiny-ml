#ifndef _MIC_H_
#define _MIC_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MIC_SAMPLE_RATE         16000   /* 16kHz */
#define MIC_PCM_BUFFER_SIZE     256     /* ~16ms buffer */

typedef uint16_t (*pf_read_adc)();
typedef int (*pf_timer_handle)();

/* PCM 16-bit signed buffer */
typedef struct mic_pcm_t {
    int16_t buf[MIC_PCM_BUFFER_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
    uint16_t bias;
    pf_read_adc pf_read;
} mic_pcm_t;

int mic_init(struct mic_pcm_t *mic, pf_read_adc pf_read);
int mic_calibrate(struct mic_pcm_t *mic);
uint16_t mic_available(struct mic_pcm_t *mic);
int16_t mic_read(struct mic_pcm_t *mic);
int mic_timer_handle(struct mic_pcm_t *mic);

#ifdef __cplusplus
}
#endif
#endif