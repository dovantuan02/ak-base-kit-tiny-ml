#ifndef __MIC_H__
#define __MIC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define MIC_SAMPLE_RATE         16000   /* 16kHz */
#define MIC_PCM_BUFFER_SIZE     256     /* ~16ms buffer */

/* PCM 16-bit signed buffer */
typedef struct {
    int16_t buf[MIC_PCM_BUFFER_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} mic_pcm_t;

extern mic_pcm_t mic_pcm;

extern void     mic_init(void);
extern void     mic_start(void);
extern void     mic_stop(void);
extern uint32_t mic_available(void);
extern int16_t  mic_read_sample(void);
extern bool     mic_insert_sample(uint16_t adc);

#ifdef __cplusplus
}
#endif

#endif /* __MIC_H__ */
