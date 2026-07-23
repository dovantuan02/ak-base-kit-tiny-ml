#ifndef _MOTION_PREPROCESS_H_
#define _MOTION_PREPROCESS_H_

#include <stdint.h>

#define LOG_PSD_BINS 29
#define FEATURES_PER_AXIS 34

class Preprocess {
public:
    static float RMS(const float *arr, uint16_t n);
    static float Skewness(const float *arr, uint16_t n);
    static float Kurtosis(const float *arr, uint16_t n);
    static float PsdSkewness(const float *spectrum, uint16_t n);
    static float PsdKurtosis(const float *spectrum, uint16_t n);
    static void LogPSD(const float *spectrum, uint16_t n, float *out_bins, uint8_t num_bins);
};

#endif // _MOTION_PREPROCESS_H_
