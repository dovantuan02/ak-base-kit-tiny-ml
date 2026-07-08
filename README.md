# AK TinyML — Anomaly Detection on MCU

[<img src="hardware/images/ak-foundation-logo.png" width="240"/>](https://github.com/the-ak-foundation)

TinyML-based anomaly detection system running on the AK Embedded Base Kit (STM32L151 + ICM-20948 IMU).

3-axis accelerometer data is collected at 58 Hz, processed through a DSP pipeline (Butterworth lowpass, FFT, PSD), and classified by a small fully-connected neural network (18→20→10→4, ~634 params) to detect **4 motion states**: idle, left-right, up-down, and vigorous shaking (maritine).

The entire pipeline — from sensor sampling to feature extraction to neural inference — runs **on-device** on the Cortex-M3 using CMSIS-DSP, with no cloud or external compute required.

> **Anomal Detection documentation**: [application/sources/app/nn/docs/Readme.md](application/sources/app/nn/docs/Readme.md)

## Project Structure

| Path | Description |
|------|-------------|
| `application/sources/app/nn/docs/` | Detailed documentation with feature math, model architecture, retraining guide |
| `application/sources/app/nn/trainning/` | Colab notebook + dataset + scaler computation script |
| `application/sources/app/nn/inference/anomal_detect/` | On-device inference engine (C++ + CMSIS-DSP) |
| `application/sources/app/nn/inference/anomal_detect/model/` | Trained model weights (emlearn C header) |
| `application/sources/app/task_accel_sensor.cpp` | ICM-20948 driver + 2-second ring buffer |
| `hardware/` | Schematics, board images, bootloader binary |

## Pipeline Overview

```
ICM-20948 @ 58 Hz  →  Ring buffer (2s = 116 samples)
    →  Butterworth LP @ 3 Hz (6th-order, biquad cascade)
    →  DC removal
    →  Time-domain features: RMS, Skewness, Kurtosis (×3 axes)
    →  PSD via 13 overlapping 16-pt FFTs + max-hold
    →  Frequency-domain features: FFT Skew, FFT Kurt, Log PSD bin 1 (×3 axes)
    →  18 features → Z-score normalize → FC neural network → class (0-3)
```

## Hardware

- **MCU**: STM32L151 (ARM Cortex-M3)
- **IMU**: ICM-20948 (9-DoF, I2C)
- **Sampling rate**: 58 Hz
- **Window**: 116 samples (~2 seconds)

[<img src="hardware/images/ak-embedded-base-kit-version-3.jpg" width="480"/>](<https://epcb.vn/products/ak-embedded-base-kit-lap-trinh-nhung-vi-dieu-khien-mcu>)

## Reference

| Topic | Link |
|-------|------|
| Tutorials | <https://epcb.vn/blogs/ak-embedded-software> |
| Vendor | <https://epcb.vn/products/ak-embedded-base-kit-lap-trinh-nhung-vi-dieu-khien-mcu> |
