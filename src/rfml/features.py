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

def centered_autocorrelation(x, lag):
    x = np.asarray(x)

    if len(x) <= lag:
        return 0.0

    x1 = x[:-lag]
    x2 = x[lag:]

    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    numerator = np.sum(x1 * x2)

    denominator = np.sqrt(
        np.sum(x1 ** 2)
        * np.sum(x2 ** 2)
    )

    return float(
        numerator
        / (denominator + 1e-12)
    )
def extract_physical_features(sample):

    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    eps = 1e-12

    magnitude = np.abs(x)

    mean_power = np.mean(
        magnitude ** 2
    )

    rms = np.sqrt(
        mean_power + eps
    )

    # ==================================================
    # 1. PSK angular symmetry
    # ==================================================

    unit_x = (
        x
        / (magnitude + eps)
    )

    circular_moment_2 = np.abs(
        np.mean(unit_x ** 2)
    )

    circular_moment_4 = np.abs(
        np.mean(unit_x ** 4)
    )

    circular_moment_8 = np.abs(
        np.mean(unit_x ** 8)
    )

    # ==================================================
    # 2. Rotation-invariant I/Q geometry
    # ==================================================

    second_moment = np.mean(
        x ** 2
    )

    circularity_coeff = (
        np.abs(second_moment)
        / (mean_power + eps)
    )

    I_centered = I - np.mean(I)
    Q_centered = Q - np.mean(Q)

    covariance = np.array([
        [
            np.mean(I_centered ** 2),
            np.mean(I_centered * Q_centered),
        ],
        [
            np.mean(I_centered * Q_centered),
            np.mean(Q_centered ** 2),
        ],
    ])

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    iq_anisotropy = (
        eigenvalues[-1]
        - eigenvalues[0]
    ) / (
        eigenvalues[-1]
        + eigenvalues[0]
        + eps
    )

    # ==================================================
    # 3. QAM radial structure
    # ==================================================

    normalized_radius = (
        magnitude / rms
    )

    amplitude_m4 = np.mean(
        normalized_radius ** 4
    )

    amplitude_m6 = np.mean(
        normalized_radius ** 6
    )

    radial_q25 = np.quantile(
        normalized_radius,
        0.25,
    )

    radial_q50 = np.quantile(
        normalized_radius,
        0.50,
    )

    radial_q75 = np.quantile(
        normalized_radius,
        0.75,
    )

    radial_iqr = (
        radial_q75
        - radial_q25
    )

    # ==================================================
    # 4. Higher-order cumulants
    # ==================================================

    m20 = np.mean(
        x ** 2
    )

    m21 = np.mean(
        np.abs(x) ** 2
    )

    m40 = np.mean(
        x ** 4
    )

    m42 = np.mean(
        np.abs(x) ** 4
    )

    c40 = (
        m40
        - 3 * m20 ** 2
    )

    c42 = (
        m42
        - np.abs(m20) ** 2
        - 2 * m21 ** 2
    )

    c40_normalized = (
        np.abs(c40)
        / (m21 ** 2 + eps)
    )

    c42_normalized = (
        np.real(c42)
        / (m21 ** 2 + eps)
    )

    # ==================================================
    # 5. Envelope dynamics
    # ==================================================

    normalized_envelope = (
        magnitude
        / (np.mean(magnitude) + eps)
    )

    envelope_diff_std = np.std(
        np.diff(
            normalized_envelope
        )
    )

    envelope_autocorr_lag1 = (
        centered_autocorrelation(
            normalized_envelope,
            1,
        )
    )

    # ==================================================
    # 6. Instantaneous-frequency dynamics
    # ==================================================

    phase_diff = np.angle(
        x[1:]
        * np.conj(x[:-1])
    )

    phase_diff_resultant = np.abs(
        np.mean(
            np.exp(
                1j * phase_diff
            )
        )
    )

    unwrapped_phase = np.unwrap(
        np.angle(x)
    )

    instantaneous_frequency = (
        np.diff(unwrapped_phase)
    )

    instantaneous_frequency_diff_std = (
        np.std(
            np.diff(
                instantaneous_frequency
            )
        )
    )

    instantaneous_frequency_autocorr_lag1 = (
        centered_autocorrelation(
            instantaneous_frequency,
            1,
        )
    )

    # ==================================================
    # 7. Spectral sideband asymmetry
    # ==================================================

    window = np.hanning(
        len(x)
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            x * window
        )
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(x)
        )
    )

    spectrum_power = (
        np.abs(spectrum) ** 2
    )

    positive_power = np.sum(
        spectrum_power[
            frequencies > 0
        ]
    )

    negative_power = np.sum(
        spectrum_power[
            frequencies < 0
        ]
    )

    spectral_sideband_asymmetry = (
        np.abs(
            positive_power
            - negative_power
        )
        / (
            positive_power
            + negative_power
            + eps
        )
    )

    return {
        # PSK
        "circular_moment_2": circular_moment_2,
        "circular_moment_4": circular_moment_4,
        "circular_moment_8": circular_moment_8,

        # geometry
        "circularity_coeff": circularity_coeff,
        "iq_anisotropy": iq_anisotropy,

        # QAM / radial
        "amplitude_m4": amplitude_m4,
        "amplitude_m6": amplitude_m6,
        "radial_q25": radial_q25,
        "radial_q50": radial_q50,
        "radial_q75": radial_q75,
        "radial_iqr": radial_iqr,

        # cumulants
        "c40_normalized": c40_normalized,
        "c42_normalized": c42_normalized,

        # AM / FM
        "envelope_diff_std": envelope_diff_std,
        "envelope_autocorr_lag1":
            envelope_autocorr_lag1,

        "phase_diff_resultant":
            phase_diff_resultant,

        "inst_freq_diff_std":
            instantaneous_frequency_diff_std,

        "inst_freq_autocorr_lag1":
            instantaneous_frequency_autocorr_lag1,

        # analog / SSB
        "spectral_sideband_asymmetry":
            spectral_sideband_asymmetry,
    }
def extract_features(
    sample,
    feature_set="basic"
):
    features = extract_basic_features(
        sample
    )

    if feature_set in {
        "temporal",
        "full",
        "full_physical",
        "full_selected_physical",
        "full_final",
    }:
        features.update(
            extract_temporal_features(
                sample
            )
        )

    if feature_set in {
        "full",
        "full_physical",
        "full_selected_physical",
        "full_final",
    }:
        features.update(
            extract_spectral_features(
                sample
            )
        )

    if feature_set == "full_physical":
        features.update(
            extract_physical_features(
                sample
            )
        )

    if feature_set == "full_selected_physical":
        features.update(
            extract_selected_physical_features(
                sample
            )
        )

    if feature_set == "full_final":
        features.update(
            extract_final_physical_features(
                sample
            )
        )

    return features
def extract_selected_physical_features(sample):

    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    eps = 1e-12

    magnitude = np.abs(x)

    # ==================================================
    # 1. PSK angular symmetry
    # ==================================================

    unit_x = (
        x
        / (magnitude + eps)
    )

    circular_moment_4 = np.abs(
        np.mean(
            unit_x ** 4
        )
    )

    # ==================================================
    # 2. Instantaneous-frequency dynamics
    # ==================================================

    unwrapped_phase = np.unwrap(
        np.angle(x)
    )

    instantaneous_frequency = np.diff(
        unwrapped_phase
    )

    inst_freq_autocorr_lag1 = (
        centered_autocorrelation(
            instantaneous_frequency,
            1,
        )
    )

    # ==================================================
    # 3. Spectral sideband asymmetry
    # ==================================================

    window = np.hanning(
        len(x)
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            x * window
        )
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            len(x)
        )
    )

    spectrum_power = (
        np.abs(spectrum) ** 2
    )

    positive_power = np.sum(
        spectrum_power[
            frequencies > 0
        ]
    )

    negative_power = np.sum(
        spectrum_power[
            frequencies < 0
        ]
    )

    spectral_sideband_asymmetry = (
        np.abs(
            positive_power
            - negative_power
        )
        / (
            positive_power
            + negative_power
            + eps
        )
    )

    return {
        "circular_moment_4":
            circular_moment_4,

        "inst_freq_autocorr_lag1":
            inst_freq_autocorr_lag1,

        "spectral_sideband_asymmetry":
            spectral_sideband_asymmetry,
    }
def extract_final_physical_features(sample):

    I = sample[0]
    Q = sample[1]

    x = I + 1j * Q

    eps = 1e-12

    magnitude = np.abs(x)

    unit_x = (
        x
        / (magnitude + eps)
    )

    circular_moment_4 = np.abs(
        np.mean(
            unit_x ** 4
        )
    )

    return {
        "circular_moment_4":
            circular_moment_4,
    }