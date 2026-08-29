import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_basic_features(sample):
    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    magnitude = np.abs(x)
    power = magnitude ** 2

    mean_power = np.mean(power)
    peak_power = np.max(power)

    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    return {
        "i_mean": np.mean(I),
        "i_std": np.std(I),
        "q_mean": np.mean(Q),
        "q_std": np.std(Q),

        "magnitude_std": np.std(magnitude),
        "magnitude_max": np.max(magnitude),

        "power_mean": mean_power,
        "rms": np.sqrt(mean_power),

        "papr": peak_power / (mean_power + 1e-12),

        "phase_diff_mean": np.mean(phase_diff),
        "phase_diff_std": np.std(phase_diff),
    }

def normalized_autocorrelation(x, lag):
    x1 = x[:-lag]
    x2 = x[lag:]

    numerator = np.abs(np.vdot(x1, x2))

    denominator = np.sqrt(
        np.vdot(x1, x1).real
        * np.vdot(x2, x2).real
    )

    return numerator / (denominator + 1e-12)


def extract_temporal_features(sample):
    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    magnitude = np.abs(x)

    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    return {
        "i_skew": skew(I),
        "i_kurtosis": kurtosis(I),

        "q_skew": skew(Q),
        "q_kurtosis": kurtosis(Q),

        "magnitude_skew": skew(magnitude),
        "magnitude_kurtosis": kurtosis(magnitude),

        "phase_diff_skew": skew(phase_diff),
        "phase_diff_kurtosis": kurtosis(phase_diff),

        "autocorr_lag1": normalized_autocorrelation(x, 1),
        "autocorr_lag2": normalized_autocorrelation(x, 2),
        "autocorr_lag4": normalized_autocorrelation(x, 4),
    }
def build_feature_table(
    radioml,
    metadata,
    feature_set="basic"
):
    rows = []

    for row in metadata.itertuples(index=False):

        sample = radioml[
            (row.modulation, int(row.snr))
        ][int(row.array_index)]

        features = extract_features(
            sample,
            feature_set=feature_set
        )

        features["sample_id"] = row.sample_id
        features["modulation"] = row.modulation
        features["snr"] = row.snr

        rows.append(features)

    return pd.DataFrame(rows)
def extract_spectral_features(sample):
    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    window = np.hanning(len(x))

    spectrum = np.fft.fftshift(
        np.fft.fft(x * window)
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(len(x))
    )

    power = np.abs(spectrum) ** 2

    normalized_power = power / (
        np.sum(power) + 1e-12
    )

    spectral_centroid = np.sum(
        frequencies * normalized_power
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            (frequencies - spectral_centroid) ** 2
            * normalized_power
        )
    )

    spectral_entropy = -np.sum(
        normalized_power
        * np.log(normalized_power + 1e-12)
    ) / np.log(len(normalized_power))

    spectral_flatness = (
        np.exp(np.mean(np.log(power + 1e-12)))
        / (np.mean(power) + 1e-12)
    )

    peak_index = np.argmax(power)

    peak_frequency = frequencies[peak_index]

    peak_power_ratio = (
        power[peak_index]
        / (np.sum(power) + 1e-12)
    )

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_entropy": spectral_entropy,
        "spectral_flatness": spectral_flatness,
        "spectral_peak_frequency": peak_frequency,
        "spectral_peak_ratio": peak_power_ratio,
    }
def extract_features(sample, feature_set="basic"):
    features = extract_basic_features(sample)

    if feature_set in {"temporal", "full"}:
        features.update(
            extract_temporal_features(sample)
        )

    if feature_set == "full":
        features.update(
            extract_spectral_features(sample)
        )

    return features