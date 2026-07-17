#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sys_ctrl.h"
#include "app_dbg.h"

#include "arm_math.h"
#include "arm_const_structs.h"

#include "anomal_detect.h"
#include "model/anomal_detection_v2.h"

#define DBG
#define PI                      (3.14159265358979f)
#define AXES                    (3)
#define SCALE_AXES              (0.00010017430721f)
#define FILTER_CUTOFF           (26.05078125f)
#define SAMPLING_FREQ           (57.0f)
#define RAW_SAMPLES_PER_AXIS    (57)
#define FFT_OVERLAP             (0)
#define STRIDE                  (FFT_LENGTH - FFT_OVERLAP)
#define NUM_FEATURES_PER_AXIS   (FEATURES_PER_AXIS)

static const float NORM_MEAN[FEATURE_LEN] = { 0.0669f, 0.1738f, 5.2485f, 3.7878f, 14.4462f, -4.1222f, -4.6421f, -5.1395f, -5.5968f, -5.9357f, -6.0622f, -6.1874f, -6.3572f, -6.4215f, -6.5790f, -6.7188f, -6.8398f, -6.9406f, -6.9518f, -7.0500f, -7.1559f, -7.2882f, -7.3951f, -7.4870f, -7.5691f, -7.5508f, -7.5463f, -7.6120f, -7.5573f, -7.4724f, -7.3654f, -7.1272f, -6.6688f, -6.4123f, 0.0583f, 0.1580f, 4.6086f, 3.6372f, 13.3304f, -4.3380f, -4.5710f, -5.0734f, -5.5681f, -6.0614f, -6.1537f, -6.3035f, -6.6414f, -6.8312f, -6.8672f, -6.9576f, -7.0520f, -7.2260f, -7.2503f, -7.3862f, -7.4617f, -7.5752f, -7.6817f, -7.7433f, -7.8727f, -7.7652f, -7.7125f, -7.7710f, -7.8106f, -7.6888f, -7.6774f, -7.4669f, -7.1044f, -6.8000f, 0.0958f, -0.7458f, 8.2619f, 3.5773f, 12.7493f, -3.9753f, -4.2723f, -4.8114f, -5.3511f, -5.7927f, -5.9193f, -6.0949f, -6.4645f, -6.5443f, -6.6348f, -6.6979f, -6.8534f, -7.0364f, -7.0515f, -7.1267f, -7.2174f, -7.4396f, -7.5920f, -7.6500f, -7.7033f, -7.6226f, -7.6658f, -7.5946f, -7.5576f, -7.3755f, -7.0929f, -6.6543f, -5.8833f, -5.4287f };
static const float NORM_SCALE[FEATURE_LEN] = { 12.056205f, 0.578546f, 0.092263f, 1.257445f, 0.165083f, 0.630276f, 0.542757f, 0.598432f, 0.624738f, 0.590015f, 0.654539f, 0.712412f, 0.695580f, 0.678122f, 0.706663f, 0.752415f, 0.745583f, 0.712384f, 0.718260f, 0.759052f, 0.761302f, 0.748098f, 0.775649f, 0.790582f, 0.782086f, 0.848281f, 0.887050f, 0.920041f, 0.927283f, 1.055457f, 1.041128f, 1.104285f, 1.001500f, 0.892369f, 15.250112f, 0.626451f, 0.104837f, 1.139051f, 0.159177f, 0.557385f, 0.543008f, 0.610003f, 0.591315f, 0.590931f, 0.683465f, 0.737082f, 0.708660f, 0.695991f, 0.695844f, 0.728669f, 0.756773f, 0.745758f, 0.754460f, 0.746801f, 0.795571f, 0.762912f, 0.813349f, 0.800823f, 0.832173f, 0.865386f, 0.881494f, 0.948908f, 0.917318f, 1.015059f, 0.943331f, 1.003951f, 0.950474f, 0.937689f, 12.226880f, 0.537049f, 0.092244f, 1.231915f, 0.172753f, 0.567972f, 0.516567f, 0.554304f, 0.547391f, 0.541672f, 0.612458f, 0.672488f, 0.650616f, 0.691378f, 0.650743f, 0.681940f, 0.673731f, 0.642951f, 0.703408f, 0.755098f, 0.802148f, 0.733789f, 0.786492f, 0.770973f, 0.805104f, 0.848502f, 0.856631f, 0.898702f, 0.963483f, 1.039960f, 1.038607f, 1.017320f, 0.916749f, 0.893665f };

static const float BIQUAD_COEFFS_DF2T[3][5] = {
    {5.923946358216797e-01f, 1.184789271643359e+00f, 5.923946358216797e-01f, -1.532692598430516e+00f, -5.902995029924591e-01f},
    {1.000000000000000e+00f, 2.000000000000000e+00f, 1.000000000000000e+00f, -1.621707230405852e+00f, -6.826597878495740e-01f},
    {1.000000000000000e+00f, 2.000000000000000e+00f, 1.000000000000000e+00f, -1.803084587305347e+00f, -8.708542900131004e-01f},
};
#ifdef DBG
static const char *feat_name[NUM_FEATURES_PER_AXIS] = {
    "RMS", "Skewness", "Kurtosis", "Spec_Skew", "Spec_Kurt",
    "LogBin01", "LogBin02", "LogBin03", "LogBin04", "LogBin05",
    "LogBin06", "LogBin07", "LogBin08", "LogBin09", "LogBin10",
    "LogBin11", "LogBin12", "LogBin13", "LogBin14", "LogBin15",
    "LogBin16", "LogBin17", "LogBin18", "LogBin19", "LogBin20",
    "LogBin21", "LogBin22", "LogBin23", "LogBin24", "LogBin25",
    "LogBin26", "LogBin27", "LogBin28", "LogBin29"};
static const char *axis_name[AXES] = {"X", "Y", "Z"};
#endif
static const char *label[MAX_PREDICT_CLASS] = {"Down", "Idle", "Left", "Right", "Unknown", "Up"};

/*
 * Single scratch buffer (512B) shared across the entire DSP pipeline.
 * Lifetime phases per extract_feature call:
 *   1. S[0->56]  = deinterleaved axis signal (written by caller)
 *   2. S[0->56]  = filtered signal (biquad in-place)
 *   3. S[0->127] = FFT complex buffer (window+interleave in reverse order)
 *   4. S[0->32]  = PSD bins (computed from |FFT|², overwrites low indices)
 *   5. S[0->5]   = model output (during inference)
 */
static float S[2 * FFT_LENGTH];
static AnomalyConfidence_t confidence;

static void welch_max_hold_inplace(uint32_t n)
{
    float win_sum_sq = 0.0f;
    for (int i = 0; i < FFT_LENGTH; i++)
    {
        float w = 0.5f * (1.0f - cosf(2.0f * PI * i / FFT_LENGTH));
        win_sum_sq += w * w;
    }

    for (int i = FFT_LENGTH - 1; i >= 0; i--)
    {
        float w = 0.5f * (1.0f - cosf(2.0f * PI * i / FFT_LENGTH));
        S[2 * i]     = (i < (int)n) ? S[i] * w : 0.0f;
        S[2 * i + 1] = 0.0f;
    }

    arm_cfft_f32(&arm_cfft_sR_f32_len64, S, 0, 1);

    for (int k = 0; k < NUM_BINS; k++)
    {
        float re = S[2 * k];
        float im = S[2 * k + 1];
        S[k] = (re * re + im * im) / (FFT_LENGTH * win_sum_sq);
    }
}

static void extract_axis_features(uint32_t n, float state[3][2], float *out)
{
    /* Butterworth lowpass filter (3 stages, order 6) — in-place on S */
    arm_biquad_cascade_df2T_instance_f32 biquad;
    arm_biquad_cascade_df2T_init_f32(&biquad, 3, (float32_t *)BIQUAD_COEFFS_DF2T, (float32_t *)state);
    arm_biquad_cascade_df2T_f32(&biquad, S, S, n);

    /* Subtract mean — in-place on S */
    float mean_val = 0.0f;
    arm_mean_f32(S, n, &mean_val);
    arm_offset_f32(S, -mean_val, S, n);

    /* Time-domain stats from S[0..n-1] */
    float sum_sq = 0.0f;
    float sum_cube = 0.0f;
    float sum_four = 0.0f;
    for (uint32_t i = 0; i < n; i++)
    {
        float v = S[i];
        sum_sq += v * v;
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

    /* Welch max-hold: S[0..56] → S[0..32] = PSD */
    welch_max_hold_inplace(n);

    /* Spectrum skewness and kurtosis from S[0..32] */
    float psd_mean = 0.0f;
    float psd_var = 0.0f;
    float spec_skew_num = 0.0f;
    float spec_kurt_num = 0.0f;
    for (int k = 0; k < NUM_BINS; k++)
    {
        float v = (S[k] > 1e-10f) ? S[k] : 1e-10f;
        S[k] = v; /* replace in-place for log10 pass */
        psd_mean += v;
    }
    psd_mean /= NUM_BINS;
    for (int k = 0; k < NUM_BINS; k++)
    {
        float d = S[k] - psd_mean;
        psd_var += d * d;
        spec_skew_num += d * d * d;
        spec_kurt_num += d * d * d * d;
    }
    psd_var /= NUM_BINS;
    spec_skew_num /= NUM_BINS;
    spec_kurt_num /= NUM_BINS;

    float psd_std = (psd_var > 0.0f) ? sqrtf(psd_var) : 1e-10f;
    float spec_skew = spec_skew_num / (psd_std * psd_std * psd_std + 1e-10f);
    float spec_kurt = spec_kurt_num / (psd_var * psd_var + 1e-10f) - 3.0f;

    /* Log10 of bins [1..29] */
    float cutoff_hz = FILTER_CUTOFF;
    float bin_resolution = SAMPLING_FREQ / FFT_LENGTH;
    int stop_bin = (int)(cutoff_hz / bin_resolution + 0.5f) + 1;
    if (stop_bin > NUM_BINS) 
    {
        stop_bin = NUM_BINS;
    }
    int start_bin = 1;
    int num_bins_used = stop_bin - start_bin;

    out[RMS] = rms;
    out[Skewness] = skewness;
    out[Kurtosis] = kurtosis;
    out[Spec_Skew] = spec_skew;
    out[Spec_Kurt] = spec_kurt;

    for (int i = 0; i < num_bins_used; i++)
    {
        out[5 + i] = log10f(S[start_bin + i]);
    }
    for (int i = num_bins_used; i < 29; i++)
    {
        out[5 + i] = 0.0f;
    }
}

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

int AnomalyInfer::extract_feature(void *data, uint32_t len)
{
    memset(filter_state, 0, sizeof(filter_state));
    if (len != RAW_SAMPLES_PER_AXIS)
    {
        APP_DBG("Expected %d samples, got %d\n", RAW_SAMPLES_PER_AXIS, len);
        return -1;
    }

    float *raw = (float *)(data);

    for (int a = 0; a < AXES; a++)
    {
        /* Deinterleave axis a directly into S[0..56] — no axis_buf needed */
        for (uint32_t i = 0; i < len; i++)
        {
            S[i] = raw[i * AXES + a] * SCALE_AXES;
        }

        /* Extract features, write directly to features[] — no feat_per_axis needed */
        extract_axis_features(len, filter_state[a], &features[a * NUM_FEATURES_PER_AXIS]);
    }

    return 0;
}

int AnomalyInfer::inference(void *data, uint32_t len)
{
    return inference(data, len, NULL, 0);
}

int AnomalyInfer::inference(void *data, uint32_t len, float *output, uint32_t output_len) 
{
    uint32_t prev = sys_ctrl_millis();

    if (extract_feature(data, len) != 0)
    {
        return -1;
    }

    APP_DBG("- Anomaly Features:\n");
#ifdef DBG
    for (int axis = 0; axis < AXES; axis++)
    {
        xfprintf((void (*)(int))sys_ctrl_shell_put_char, "[%s] ", axis_name[axis]);
        for (int feature = 0; feature < 5; feature++)
        {
            xfprintf((void (*)(int))sys_ctrl_shell_put_char, "%s=%.4f ", feat_name[feature], features[axis * NUM_FEATURES_PER_AXIS + feature]);
        }
        xfprintf((void (*)(int))sys_ctrl_shell_put_char,"\n");
    }
#endif

    for (int i = 0; i < FEATURE_LEN; i++)
    {
        features[i] = (features[i] - NORM_MEAN[i]) * NORM_SCALE[i];
    }

    if (anomaly_model_regress(features, FEATURE_LEN, S, MAX_PREDICT_CLASS) != EmlOk)
    {
        return -1;
    }
    if (output != NULL)
    {
        for (int i = 0; i < output_len; i++)
        {
            ((float *)output)[i] = S[i];
        }
    }

    int predicted = -1;
    float max_prob = -1.0f;

    for (int i = 0; i < MAX_PREDICT_CLASS; i++)
    {
        if (S[i] >= confidence.confidence[i] &&
            S[i] > max_prob)
        {
            max_prob = S[i];
            predicted = i;
        }
    }


    APP_DBG("Pointer [%08X]\n", (unsigned int)this);
    APP_DBG("\tP(down)= %.3f\n", S[0]);
    APP_DBG("\tP(idle)= %.3f\n", S[1]);
    APP_DBG("\tP(left)= %.3f\n", S[2]);
    APP_DBG("\tP(right)= %.3f\n", S[3]);
    APP_DBG("\tP(unknown)= %.3f\n", S[4]);
    APP_DBG("\tP(up)= %.3f\n", S[5]);
    APP_DBG("----> Predict= %s\n\n", predicted == -1 ? "None" : label[predicted]);
    return predicted;
}

int AnomalyInfer::getMaxPredictClass() {
    return MAX_PREDICT_CLASS;
}

int AnomalyInfer::setConfidence(AnomalyConfidence_t conf) {
    confidence = conf;
    return 0;
}