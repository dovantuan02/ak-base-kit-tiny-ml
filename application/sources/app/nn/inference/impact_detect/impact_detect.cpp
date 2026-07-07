#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "sys_ctrl.h"
#include "app_dbg.h"
#include "impact_detect.h"
#include "model/random_forest_impact_detector_1axis.h"

ImpactInfer::ImpactInfer(/* args */) {
    for (int i = 0; i < FEATURE_LEN; i++) {
        features[i] = 0;
    }
}

ImpactInfer::~ImpactInfer() {

}

int ImpactInfer::inference(void *data, uint32_t len) {
    uint32_t prev = sys_ctrl_millis();
    extract_feature(data, len);

    APP_DBG("- F: \n");
    for (int i = 0; i < 8; i++) {
        APP_DBG("\t[%d]=%d\n", i, features[i]);
    }
    
    float out[3];
    impact_detector_zaxis_predict_proba(features, 8, out, 3);

    int predicted_class = 0;
    float max_prob = out[0];
    for (int i = 1; i < 3; i++) {
        if (out[i] > max_prob) {
            max_prob = out[i];
            predicted_class = i;
        }
    }

    APP_DBG("[%08X] P(palm)=%.2f, P(finger)=%.2f, P(noise)=%.2f ->>>>>>>>> Result: %d, time: %d\n", (unsigned int) this, out[0], out[1], out[2], predicted_class, (sys_ctrl_millis() - prev));
    return predicted_class;
}

int ImpactInfer::extract_feature(void* data, uint32_t length) {
    int8_t *z_buf = ((int8_t*)data);

    // =========================
    // 1. peak align
    // =========================
    int peak_idx = 0;
    int8_t peak_val = 0;

    for (uint32_t i = 0; i < length; i++)
    {
        int8_t v = z_buf[i];
        if (abs(v) > abs(peak_val))
        {
            peak_val = v;
            peak_idx = i;
        }
    }

    int start = peak_idx - 20;
    int end = peak_idx + 50;

    if (start < 0) start = 0;
    if (end > length) end = length;

    // =========================
    // 2. compute features (integer math, match int16_t model input)
    // =========================
    int32_t sum = 0;
    int32_t sum_sq = 0;

    int8_t max_v = INT8_MIN;
    int8_t min_v = INT8_MAX;

    int8_t prev = z_buf[start];
    int8_t max_slope = 0;
    int32_t sum_abs_dz = 0;
    int dz_count = 0;

    for (int i = start; i < end; i++)
    {
        int8_t v = z_buf[i];

        sum += v;
        sum_sq += v * v;

        if (v > max_v) max_v = v;
        if (v < min_v) min_v = v;

        if (i > start)
        {
            int8_t slope = v - prev;
            int abs_slope = abs(slope);
            if (abs_slope > max_slope)
                max_slope = abs_slope;
            sum_abs_dz += abs_slope;
            dz_count++;
        }

        prev = v;
    }

    int len = (end - start);
    int32_t mean = sum / len;
    float std = sqrtf((float)sum_sq / len - (float)mean * (float)mean);

    // =========================
    // OUTPUT FEATURE VECTOR
    // MUST MATCH TRAINING ORDER
    // =========================
    features[0] = abs(peak_val);                     // peak magnitude
    features[1] = max_v;                              // max
    features[2] = min_v;                              // min
    features[3] = (int16_t)(sum_sq / 10);             // energy (scaled to fit int16_t)
    features[4] = (int16_t)mean;                      // mean
    features[5] = (int16_t)std;                       // std
    features[6] = max_slope;                          // max slope
    features[7] = dz_count > 0 ? (int16_t)(sum_abs_dz / dz_count) : 0; // mean abs slope
    return 0;
}

