import numpy as np
from scipy import signal
import json
import os
import re

CONFIG = {
    'axes': 3,
    'scale_axes': 0.2,
    'filter_type': 'low',
    'filter_cutoff': 3.0,
    'filter_order': 6,
    'fft_length': 16,
    'sampling_freq': 58,
    'raw_samples_per_axis': 116,
}

def extract_spec_features_axis(data_axis, config):
    fs = config['sampling_freq']
    fft_len = config['fft_length']
    x = data_axis.copy().astype(np.float64)

    if config['filter_type'] == 'low' and config['filter_order'] > 0:
        sos = signal.butter(config['filter_order'],
                            config['filter_cutoff'],
                            btype='low', fs=fs, output='sos')
        x = signal.sosfilt(sos, x)

    x = x - np.mean(x)

    rms = float(np.sqrt(np.mean(x ** 2)))
    std = np.std(x, ddof=0)
    if std == 0:
        std = 1e-10
    skewness = float(np.mean(x ** 3) / (std ** 3))
    kurtosis = float(np.mean(x ** 4) / (std ** 4) - 3)

    window = np.array([0.0] + [0.5] + [1.0] * (fft_len - 3) + [0.5])
    noverlap = fft_len // 2
    _, _, Pxx_all = signal.spectrogram(x, fs=fs, window=window,
                                       nperseg=fft_len,
                                       noverlap=noverlap,
                                       mode='psd')
    Pxx = np.max(Pxx_all, axis=1)

    pxx_safe = np.maximum(Pxx, 1e-10)
    fft_skew = float(np.mean((pxx_safe - np.mean(pxx_safe)) ** 3) /
                     (np.std(pxx_safe) ** 3 + 1e-10))
    fft_kurt = float(np.mean((pxx_safe - np.mean(pxx_safe)) ** 4) /
                     (np.var(pxx_safe) ** 2 + 1e-10) - 3)

    val = max(Pxx[1], 1e-10)
    log_val = float(np.log10(val))

    return [rms, skewness, kurtosis, fft_skew, fft_kurt, log_val]

def extract_features(raw_window, config=CONFIG):
    x = raw_window * config['scale_axes']
    features = []
    for axis in range(config['axes']):
        feat = extract_spec_features_axis(x[:, axis], config)
        features.extend(feat)
    return np.array(features, dtype=np.float32)

dataset_path = '/home/tuan/workplace/ak-tinyML/application/sources/app/nn/trainning/anomaly-detection-export'

label_map = {}
labels_filepath = os.path.join(dataset_path, 'info.labels')
with open(labels_filepath, 'r') as f:
    labels_info = json.load(f)

unique_labels = sorted(list(set(item['label']['label'] for item in labels_info['files']
                                if 'label' in item and 'label' in item['label'])))
unique_labels = [l for l in unique_labels if l != 'testing']
for i, label_name in enumerate(unique_labels):
    label_map[label_name] = i
print(f"Label mapping: {label_map}")

window_size = CONFIG['raw_samples_per_axis']
stride = window_size // 2

all_raw_windows = []
all_labels = []

subdirs = ['training']
for subdir in subdirs:
    current_dir = os.path.join(dataset_path, subdir)
    if not os.path.isdir(current_dir):
        continue
    for filename in os.listdir(current_dir):
        if not filename.endswith('.json') or filename.startswith('info.'):
            continue
        filepath = os.path.join(current_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            values = data['payload']['values']
            raw_recording = np.array(values, dtype=np.float32)

            label_name = data.get('label', {}).get('label')
            if label_name is None:
                match = re.match(r'([a-zA-Z0-9_\-]+)\.json', filename)
                if match:
                    label_name = match.group(1)

            if label_name not in label_map:
                continue

            n = raw_recording.shape[0]
            if n < window_size:
                continue

            num_windows = (n - window_size) // stride + 1
            for i in range(num_windows):
                start = i * stride
                window = raw_recording[start:start + window_size]
                if window.shape == (window_size, CONFIG['axes']):
                    all_raw_windows.append(window)
                    all_labels.append(label_map[label_name])
        except Exception as e:
            print(f"Error {filepath}: {e}")

print(f"Loaded {len(all_raw_windows)} windows")
raw_data = np.array(all_raw_windows, dtype=np.float32)
labels = np.array(all_labels, dtype=np.int32)

print("Extracting DSP features...")
X = []
for i in range(len(raw_data)):
    feat = extract_features(raw_data[i])
    X.append(feat)
X = np.array(X, dtype=np.float32)
print(f"Features shape: {X.shape}")

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std = np.maximum(std, 1e-10)

print("\n/* Copy to anomal_detect.cpp */")
print(f"static const float NORM_MEAN[FEATURE_LEN] = {{", ", ".join(f"{v:.4f}f" for v in mean), "};")
print(f"static const float NORM_SCALE[FEATURE_LEN] = {{", ", ".join(f"{v:.6f}f" for v in 1.0 / std), "};")
