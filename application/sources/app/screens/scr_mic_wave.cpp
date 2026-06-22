#include "ak.h"
#include "app.h"
#include "app_dbg.h"
#include "timer.h"
#include "task_list.h"
#include "mic.h"
#include "screen_manager.h"
#include "io_cfg.h"

#define WAVE_HEIGHT         64
#define WAVE_CENTER_HEIGHT  (WAVE_HEIGHT / 2)
#define WAVE_WIDTH          128
#define SAMPLES_PER_COL     8

static int16_t wave_buf[WAVE_WIDTH];
static void mic_wave_init(void);
extern mic_pcm_t mic_pcm;
view_dynamic_t dyn_view_mic_wave = {
	{
		.item_type = ITEM_TYPE_DYNAMIC,
	},
	mic_wave_init
};

view_screen_t scr_mic_wave = {
	&dyn_view_mic_wave,
	ITEM_NULL,
	ITEM_NULL,

	.focus_item = 0,
};

void mic_wave_init(void) {
    for (int i = 0; i < WAVE_WIDTH; i++) {
        wave_buf[i] = 0;
    }
}

void task_polling_mic_wave() {
    static int32_t acc = 0;
    static uint32_t count = 0;
    static int drawn = 0;
    int16_t sample;
    while (mic_available(&mic_pcm) > 0) {
        sample = mic_read(&mic_pcm);
        if (sample < 0) {
            acc -= sample;
        }
        else {
            acc += sample;
        }
        count++;

        if (count >= SAMPLES_PER_COL) {
            int32_t avg = acc / SAMPLES_PER_COL;
            int16_t h = (int16_t)((avg * WAVE_CENTER_HEIGHT) / 4000);
            if (h > WAVE_CENTER_HEIGHT) {
                h = WAVE_CENTER_HEIGHT;
            }

            for (int i = 0; i < WAVE_WIDTH - 1; i++) {
                wave_buf[i] = wave_buf[i + 1];
            }
            wave_buf[WAVE_WIDTH - 1] = h;

            acc = 0;
            count = 0;
            drawn = 1;
        }
    }
    // APP_DBG("Draw: %d\n", drawn);
    if (drawn) {
        view_render.clear(false);
        for (int x = 0; x < WAVE_WIDTH; x++) {
            int16_t h = wave_buf[x];
            // APP_DBG("\t- sample:%d, x0: %d, y0: %d, x1: %d, y1: %d\n", sample, x, 32 - h, x, 32 + h);
            view_render.drawLine(x, WAVE_CENTER_HEIGHT - h, x, WAVE_CENTER_HEIGHT + h, WHITE);
        }
        view_render.update();
        drawn = 0;
    }
}

void scr_mic_wave_handle(ak_msg_t *msg) {
    switch (msg->sig) {
    case SCREEN_ENTRY: {
        APP_DBG_SIG("SCREEN_ENTRY\n");
        mic_calibrate(&mic_pcm);
        timer_set(AC_TASK_DISPLAY_ID, AC_DISPLAY_SHOW_WAVE_MIC, 100, TIMER_ONE_SHOT);

    } break;

    case AC_DISPLAY_SHOW_WAVE_MIC: {
        APP_DBG_SIG("AC_DISPLAY_SHOW_WAVE_MIC\n");
        task_polling_set_ability(AC_TASK_POLLING_MIC_WAVE_ID, AK_ENABLE);
    } break;

    default:
        break;
    }
}
