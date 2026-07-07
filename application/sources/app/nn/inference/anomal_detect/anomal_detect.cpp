#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sys_ctrl.h"
#include "app_dbg.h"

#include "arm_math.h"
#include "arm_const_structs.h"

#include "anomal_detect.h"
#include "model/anomal_detection_v1.h"

#define AXES                    (3)
#define SCALE_AXES              (0.2f)
#define FILTER_CUTOFF           (3.0f)
#define SAMPLING_FREQ           (58.0f)
#define RAW_SAMPLES_PER_AXIS    (116)
#define FFT_LENGTH              (16)
#define FFT_OVERLAP             (FFT_LENGTH / 2)
#define NUM_BINS                (FFT_LENGTH / 2 + 1)
#define STRIDE                  (FFT_LENGTH - FFT_OVERLAP)

#define S2                      (13.5f)
#define PSD_SCALE               (1.0f / (SAMPLING_FREQ * S2))
#define CONVERT_G_TO_MS2(x)     ((x) * 9.80665f)

static const float NORM_MEAN[FEATURE_LEN] = {673.8051f, -0.2186f, 0.1686f, 2.4455f, 4.0375f, 4.6634f, 2094.2317f, 0.4336f, 0.4695f, 2.4397f, 4.0200f, 5.4251f, 4572.1592f, -1.9544f, 2.9985f, 2.4415f, 4.0257f, 6.7278f};
static const float NORM_SCALE[FEATURE_LEN] = {0.002215f, 1.088804f, 0.699978f, 69.897835f, 22.677576f, 0.936788f, 0.000495f, 0.873590f, 0.474650f, 78.085106f, 25.289017f, 1.047305f, 0.001687f, 1.688269f, 0.626388f, 193.082199f, 63.170059f, 12.729128f};

static const float WINDOW[FFT_LENGTH] = {
    0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f};

static const float BIQUAD_COEFFS_DF2T[3][5] = {
    {1.0326097345e-05f, 2.0652194690e-05f, 1.0326097345e-05f, +1.4485440706e+00f, -5.2855930280e-01f},
    {1.0f, 2.0f, 1.0f, +1.5462039793e+00f, -6.3161378691e-01f},
    {1.0f, 2.0f, 1.0f, +1.7506318227e+00f, -8.4733389384e-01f},
};

AnomalyInfer::AnomalyInfer()
{
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        features[i] = 0.0f;
    }
    memset(filter_state, 0, sizeof(filter_state));
}

AnomalyInfer::~AnomalyInfer()
{
}

static void compute_psd_maxhold(const float *buf, uint32_t n, float psd_out[NUM_BINS])
{
    for (int k = 0; k < NUM_BINS; k++)
    {
        psd_out[k] = 0.0f;
    }

    int num_windows = (n - FFT_LENGTH) / STRIDE + 1;

    for (int w = 0; w < num_windows; w++)
    {
        int start = w * STRIDE;

        float cplx_buf[2 * FFT_LENGTH];
        float seg_mean = 0.0f;

        arm_mean_f32((float32_t*)&buf[start], FFT_LENGTH, &seg_mean);

        for (int i = 0; i < FFT_LENGTH; i++)
        {
            cplx_buf[2 * i]     = (buf[start + i] - seg_mean) * WINDOW[i];
            cplx_buf[2 * i + 1] = 0.0f;
        }

        arm_cfft_f32(&arm_cfft_sR_f32_len16, cplx_buf, 0, 1);

        for (int k = 0; k < NUM_BINS; k++)
        {
            float re = cplx_buf[2 * k];
            float im = cplx_buf[2 * k + 1];
            float mag2 = re * re + im * im;
            int is_pos_freq = (k > 0 && k < FFT_LENGTH / 2);
            float psd = mag2 * PSD_SCALE * (is_pos_freq ? 2.0f : 1.0f);
            if (psd > psd_out[k])
            {
                psd_out[k] = psd;
            }
        }
    }
}

static void extract_axis_features(const float *axis_data, uint32_t n,
                                  float state[3][2],
                                  float out[6])
{
    float buf[RAW_SAMPLES_PER_AXIS];
    memcpy(buf, axis_data, n * sizeof(float));

    arm_biquad_cascade_df2T_instance_f32 biquad;
    arm_biquad_cascade_df2T_init_f32(&biquad, 3, (float32_t *)BIQUAD_COEFFS_DF2T, (float32_t *)state);
    arm_biquad_cascade_df2T_f32(&biquad, buf, buf, n);

    float mean_val = 0.0f;
    arm_mean_f32(buf, n, &mean_val);
    arm_offset_f32(buf, -mean_val, buf, n);

    float sum_sq = 0.0f;
    float sum_cube = 0.0f;
    float sum_four = 0.0f;
    for (uint32_t i = 0; i < n; i++)
    {
        float v = buf[i];
        sum_sq   += v * v;
        sum_cube += v * v * v;
        sum_four += v * v * v * v;
    }

    float rms = sqrtf(sum_sq / n);
    float variance = sum_sq / n;
    float std_val = (variance > 0.0f) ? sqrtf(variance) : 1e-10f;
    float std3 = std_val * std_val * std_val;
    float std4 = std3 * std_val;
    float skewness = (std3 > 0.0f) ? (sum_cube / n) / std3 : 0.0f;
    float kurtosis = (std4 > 0.0f) ? (sum_four / n) / std4 - 3.0f : -3.0f;

    float psd[NUM_BINS];
    compute_psd_maxhold(buf, n, psd);

    float psd_safe[NUM_BINS];
    for (int k = 0; k < NUM_BINS; k++)
    {
        psd_safe[k] = (psd[k] > 1e-10f) ? psd[k] : 1e-10f;
    }

    float psd_mean = 0.0f;
    arm_mean_f32(psd_safe, NUM_BINS, &psd_mean);

    float psd_var = 0.0f;
    float fft_skew_num = 0.0f;
    float fft_kurt_num = 0.0f;
    for (int k = 0; k < NUM_BINS; k++)
    {
        float d = psd_safe[k] - psd_mean;
        psd_var      += d * d;
        fft_skew_num += d * d * d;
        fft_kurt_num += d * d * d * d;
    }
    psd_var /= NUM_BINS;
    fft_skew_num /= NUM_BINS;
    fft_kurt_num /= NUM_BINS;

    float psd_std = (psd_var > 0.0f) ? sqrtf(psd_var) : 1e-10f;
    float fft_skew = fft_skew_num / (psd_std * psd_std * psd_std + 1e-10f);
    float fft_kurt = fft_kurt_num / (psd_var * psd_var + 1e-10f) - 3.0f;

    int start_bin = 1;
    float bin_raw = FILTER_CUTOFF * FFT_LENGTH / SAMPLING_FREQ;
    int stop_bin = (int)(bin_raw + 0.5f) + 1;

    float log_val = 0.0f;
    if (start_bin < stop_bin && start_bin < NUM_BINS)
    {
        float val = psd_safe[start_bin];
        log_val = log10f(val);
    }

    out[0] = rms;
    out[1] = skewness;
    out[2] = kurtosis;
    out[3] = fft_skew;
    out[4] = fft_kurt;
    out[5] = log_val;
}

int AnomalyInfer::extract_feature(void *data, uint32_t len)
{
    memset(filter_state, 0, sizeof(filter_state));
    if (len != RAW_SAMPLES_PER_AXIS)
    {
        APP_DBG("Expected %d samples, got %d\n", RAW_SAMPLES_PER_AXIS, len);
        return -1;
    }

    int16_t *raw = (int16_t *)(data);

    float axis_buf[AXES][RAW_SAMPLES_PER_AXIS];
    for (uint32_t i = 0; i < len; i++)
    {
        axis_buf[0][i] = (float)(CONVERT_G_TO_MS2(raw[i * AXES + 0]) * SCALE_AXES);
        axis_buf[1][i] = (float)(CONVERT_G_TO_MS2(raw[i * AXES + 1]) * SCALE_AXES);
        axis_buf[2][i] = (float)(CONVERT_G_TO_MS2(raw[i * AXES + 2]) * SCALE_AXES);
    }

    float feat_per_axis[6];
    int feat_idx = 0;
    for (int a = 0; a < AXES; a++)
    {
        extract_axis_features(axis_buf[a], len, filter_state[a], feat_per_axis);
        for (int f = 0; f < 6; f++)
        {
            features[feat_idx++] = feat_per_axis[f];
        }
    }

    return 0;
}

int AnomalyInfer::inference(void *data, uint32_t len)
{
    uint32_t prev = sys_ctrl_millis();

    if (extract_feature(data, len) != 0)
    {
        return -1;
    }

    APP_DBG("- Anomaly Features:\n");
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        APP_DBG("\t[%d]=%.4f\n", i, features[i]);
    }
#define MAX_PREDICT_CLASS 4
    float out[MAX_PREDICT_CLASS];
    const char *label[MAX_PREDICT_CLASS] = {
        "Idle",
        "Left-Right",
        "Maritine",
        "Up-Down"};
    float norm_features[FEATURE_LEN];
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        norm_features[i] = (features[i] - NORM_MEAN[i]) * NORM_SCALE[i];
    }
    anomaly_model_regress(norm_features, FEATURE_LEN, out, MAX_PREDICT_CLASS);

    int predicted_class = 0;
    float max_prob = out[0];
    for (int i = 1; i < MAX_PREDICT_CLASS; i++)
    {
        if (out[i] > max_prob)
        {
            max_prob = out[i];
            predicted_class = i;
        }
    }
    if (max_prob < 0.3f && predicted_class != 0)
    {
        predicted_class = 0;
    }

    APP_DBG("[%08X] P(idle)=%.3f "
            "P(left-right)=%.3f "
            "P(maritime)=%.3f "
            "P(up-down)=%.3f >>>>>>>>> Result: %s, time: %d\n",
            (unsigned int)this,
            out[0], out[1], out[2], out[3],
            label[predicted_class], (sys_ctrl_millis() - prev));

    return predicted_class;
}
