#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sys_ctrl.h"
#include "app_dbg.h"

#include "arm_math.h"
#include "arm_const_structs.h"

#include "motion_direct_classify.h"
#include "model/motion_direct_classify_model.h"

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

static const float NORM_MEAN[FEATURE_LEN] = { 0.0719f, 0.0995f, 5.4490f, 3.7382f, 14.1261f, -4.1330f, -4.6119f, -5.0993f, -5.5457f, -5.9415f, -5.9847f, -6.1570f, -6.3290f, -6.4163f, -6.5327f, -6.6496f, -6.8356f, -6.9605f, -6.9696f, -7.0329f, -7.1368f, -7.2549f, -7.3616f, -7.4406f, -7.5584f, -7.4952f, -7.5771f, -7.5810f, -7.5687f, -7.4534f, -7.3262f, -7.0873f, -6.5876f, -6.3079f, 0.0616f, 0.3180f, 4.3104f, 3.6315f, 13.2231f, -4.2367f, -4.4527f, -5.0283f, -5.5915f, -6.0301f, -6.1213f, -6.2787f, -6.6258f, -6.8237f, -6.8758f, -6.9498f, -7.0438f, -7.1553f, -7.2402f, -7.3665f, -7.4572f, -7.6275f, -7.6902f, -7.7489f, -7.8501f, -7.7773f, -7.7671f, -7.7868f, -7.8238f, -7.6918f, -7.6657f, -7.4611f, -7.0845f, -6.7806f, 0.0884f, -0.7941f, 8.3792f, 3.6272f, 13.1404f, -4.0610f, -4.4105f, -4.8474f, -5.3470f, -5.8370f, -5.9737f, -6.1889f, -6.5016f, -6.5925f, -6.6847f, -6.7785f, -6.9151f, -7.0467f, -7.0616f, -7.1657f, -7.2663f, -7.4458f, -7.5853f, -7.7055f, -7.7336f, -7.6476f, -7.6204f, -7.6253f, -7.5314f, -7.3482f, -7.1613f, -6.6685f, -5.9613f, -5.5504f };
static const float NORM_SCALE[FEATURE_LEN] = { 10.313805f, 0.569355f, 0.092169f, 1.193996f, 0.158709f, 0.610087f, 0.543415f, 0.589654f, 0.611746f, 0.578238f, 0.649009f, 0.701902f, 0.691730f, 0.646156f, 0.679985f, 0.719828f, 0.723761f, 0.681535f, 0.737040f, 0.756151f, 0.739874f, 0.706928f, 0.755724f, 0.760480f, 0.795412f, 0.829065f, 0.845690f, 0.904734f, 0.940804f, 1.030589f, 1.050383f, 1.101235f, 1.015256f, 0.904591f, 16.746591f, 0.652848f, 0.108822f, 1.121137f, 0.157254f, 0.548363f, 0.548704f, 0.613974f, 0.596110f, 0.585630f, 0.689988f, 0.759718f, 0.702491f, 0.701941f, 0.720634f, 0.742164f, 0.754096f, 0.749647f, 0.744423f, 0.757681f, 0.792686f, 0.767760f, 0.814712f, 0.824135f, 0.832778f, 0.851554f, 0.864281f, 0.897256f, 0.881747f, 0.974453f, 0.923679f, 0.985914f, 0.949842f, 0.909363f, 13.276666f, 0.537950f, 0.090665f, 1.268610f, 0.173435f, 0.575975f, 0.502935f, 0.558554f, 0.558281f, 0.543968f, 0.627195f, 0.652300f, 0.657518f, 0.709448f, 0.667345f, 0.701980f, 0.694616f, 0.660147f, 0.701925f, 0.736040f, 0.825578f, 0.762393f, 0.813294f, 0.766670f, 0.844944f, 0.883073f, 0.921908f, 0.944610f, 1.029522f, 1.062279f, 1.117482f, 1.038127f, 0.866381f, 0.824607f };

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

enum Class 
{
    Down,
    Idle,
    Left,
    Right,
    Unknown,
    Up,
    End
};

static const char *label[Class::End] = {"Down", "Idle", "Left", "Right", "Unknown", "Up"};

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
static MotionDirectConfidence_t confidence;

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

MotionDirectInfer::MotionDirectInfer()
{
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        features[i] = 0.0f;
    }
    memset(filter_state, 0, sizeof(filter_state));
}

MotionDirectInfer::~MotionDirectInfer()
{
}

int MotionDirectInfer::extract_feature(void *data, uint32_t len)
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

int MotionDirectInfer::inference(void *data, uint32_t len)
{
    return inference(data, len, NULL, 0);
}

int MotionDirectInfer::inference(void *data, uint32_t len, float *output, uint32_t output_len) 
{
    uint32_t prev = sys_ctrl_millis();

    if (extract_feature(data, len) != 0)
    {
        return -1;
    }

    APP_DBG("- MotionDirect Features:\n");
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

    if (motion_direct_model_regress(features, FEATURE_LEN, S, MAX_PREDICT_CLASS) != EmlOk)
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
    if (predicted == Class::Right && S[Class::Left] >= confidence.left) {
        predicted = Class::Left;
    }
    if (predicted == Class::Down && S[Class::Up] >= confidence.up) {
        predicted = Class::Up;
    }

    APP_DBG("Pointer [%08X]\n", (unsigned int)this);
    APP_DBG("\tP(unknown)= %.3f\n", S[4]);
    APP_DBG("\tP(idle)= %.3f\n", S[1]);
    APP_DBG("\tP(left)= %.3f\n", S[2]);
    APP_DBG("\tP(right)= %.3f\n", S[3]);
    APP_DBG("\tP(up)= %.3f\n", S[5]);
    APP_DBG("\tP(down)= %.3f\n", S[0]);
    APP_DBG("----> Predict= %s\n\n", predicted == -1 ? "None" : label[predicted]);
    return predicted;
}

int MotionDirectInfer::getMaxPredictClass() {
    return MAX_PREDICT_CLASS;
}

int MotionDirectInfer::setConfidence(MotionDirectConfidence_t conf) {
    confidence = conf;
    return 0;
}