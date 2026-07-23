#include "motion_preprocess.h"
#include <cmath>

float Preprocess::RMS(const float *arr, uint16_t n)
{
    if (n == 0)
        return 0.0f;

    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        sum += arr[i] * arr[i];
    }

    return std::sqrt(sum / n);
}

float Preprocess::Skewness(const float *arr, uint16_t n)
{
    if (n < 3)
        return 0.0f;

    float mean = 0.0f;
    for (uint16_t i = 0; i < n; i++) {
        mean += arr[i];
    }
    mean /= (float)n;

    float m2 = 0.0f;
    float m3 = 0.0f;

    for (uint16_t i = 0; i < n; i++)
    {
        float d = arr[i] - mean;
        float d2 = d * d;

        m2 += d2;
        m3 += d2 * d;
    }

    m2 /= (float)n;
    m3 /= (float)n;

    constexpr float EPS = 1e-12f;
    if (m2 < EPS)
        return 0.0f;

    float std_dev = std::sqrt(m2);

    return m3 / (std_dev * m2);
}

float Preprocess::Kurtosis(const float *arr, uint16_t n)
{
    if (n < 4)
        return 0.0f;

    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    float mean = sum / n;

    float m2 = 0.0f, m4 = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        float d = arr[i] - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;

    constexpr float EPS = 1e-12f;
    if (m2 < EPS) {
        return 0.0f;
    }

    return (m4 / (m2 * m2)) - 3.0f;
}

float Preprocess::PsdSkewness(const float *spectrum, uint16_t n)
{
    if (n < 3)
        return 0.0f;

    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        sum += spectrum[i];
    }
    float mean = sum / n;

    float m2 = 0.0f, m3 = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        float d = spectrum[i] - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;

     constexpr float EPS = 1e-12f;
    if (m2 < EPS) {
        return 0.0f;
    }

    float std_dev = std::sqrt(m2);

    return m3 / (std_dev * std_dev * std_dev);
}

float Preprocess::PsdKurtosis(const float *spectrum, uint16_t n)
{
    if (n < 4)
        return 0.0f;

    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        sum += spectrum[i];
    }
    float mean = sum / n;

    float m2 = 0.0f, m4 = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        float d = spectrum[i] - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;

    if (m2 == 0.0f)
        return 0.0f;

    return (m4 / (m2 * m2)) - 3.0f;
}

void Preprocess::LogPSD(const float *spectrum, uint16_t n, float *out_bins, uint8_t num_bins)
{
    if (n == 0 || num_bins == 0)
        return;

    uint16_t bin_size = n / num_bins;
    if (bin_size == 0)
        bin_size = 1;

    for (uint8_t b = 0; b < num_bins; b++)
    {
        float energy = 0.0f;
        uint16_t start = b * bin_size;
        uint16_t end = start + bin_size;
        if (b == num_bins - 1)
            end = n;

        for (uint16_t i = start; i < end; i++)
        {
            energy += spectrum[i] * spectrum[i];
        }

        out_bins[b] = std::log(energy + 1e-10f);
    }
}
