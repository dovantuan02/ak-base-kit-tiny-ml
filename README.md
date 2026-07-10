# AK TinyML — Anomaly Detection on MCU

[<img src="hardware/images/ak-foundation-logo.png" width="240"/>](https://github.com/the-ak-foundation)

TinyML-based anomaly detection system running on the AK Embedded Base Kit (STM32L151 + ICM-20948 IMU).

3-axis accelerometer data is collected at 58 Hz, processed through a DSP pipeline (Butterworth lowpass, FFT, PSD), and classified by a small fully-connected neural network to detect **4 motion states**: idle, left-right, up-down, and vigorous shaking (maritine).

The entire pipeline from sensor sampling to feature extraction to neural inference and runs on the Cortex-M3 using CMSIS-DSP, with no cloud or external compute required.

> **Anomal Detection documentation**: [application/sources/app/nn/docs/Readme.md](application/sources/app/nn/docs/Readme.md)

## Project Structure

| Path | Description |
|------|-------------|
| [Trainning](application/sources/app/nn/trainning/) | Colab notebook, dataset, scaler computation script |
| [AI Pipeline](application/sources/app/nn/inference/anomal_detect/) | On device inference engine (C/C++ + CMSIS-DSP) |
| [Model](application/sources/app/nn/inference/anomal_detect/model/) | Model weights |
| [Sensor](application/sources/app/task_accel_sensor.cpp) | ICM-20948 handle |

## Hardware

- **MCU**: STM32L151 (ARM Cortex-M3)
- **IMU**: ICM-20948 (9-DoF, I2C)

## Reference

| Topic | Link |
|-------|------|
| Tutorials | <https://epcb.vn/blogs/ak-embedded-software> |
| Vendor | <https://epcb.vn/products/ak-embedded-base-kit-lap-trinh-nhung-vi-dieu-khien-mcu> |
