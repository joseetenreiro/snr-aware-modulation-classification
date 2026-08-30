# SNR-Aware Automatic Modulation Classification

Classical machine learning project for **Automatic Modulation Classification (AMC)** from complex I/Q radio signals under different Signal-to-Noise Ratio (SNR) conditions.

The project uses the **RadioML 2016.10A** dataset and focuses not only on classification performance, but also on understanding **why the model fails at different SNR regimes** and whether physically motivated signal features can improve those limitations.

The final system uses a **Random Forest with 29 handcrafted features**, including a fourth-order circular moment specifically designed to improve PSK-order discrimination.

---

## Final results

The final model was frozen before opening the held-out test set.

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Training OOF | 0.5543 | 0.5765 |
| Validation | 0.5537 | 0.5762 |
| **Held-out test** | **0.5562** | **0.5786** |

The close agreement between out-of-fold training, validation and test performance suggests that the final pipeline generalizes consistently.

### Performance by SNR

Performance strongly depends on the signal-to-noise ratio.

| SNR (dB) | Test accuracy |
|---:|---:|
| -20 | 0.0927 |
| -10 | 0.2109 |
| 0 | 0.7467 |
| 10 | 0.8691 |
| 18 | 0.9030 |

At extremely low SNR, performance approaches random chance for an 11-class problem. As SNR increases, modulation-specific temporal, spectral and angular structure becomes progressively recoverable.

---

## Dataset

This project uses **RadioML 2016.10A**.

The dataset contains:

- 220,000 I/Q signal examples
- 11 modulation classes
- 20 SNR levels from -20 dB to +18 dB
- 1,000 examples per modulation/SNR combination
- 128 complex samples per example
- Input shape: `(2, 128)`, corresponding to I and Q components

The modulation classes are:

`8PSK`, `AM-DSB`, `AM-SSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, and `WBFM`.

The dataset is split using joint stratification by **modulation and SNR**:

```text
Train       70%   154,000 samples
Validation  15%    33,000 samples
Test        15%    33,000 samples
```

The test set remained closed during model development and was evaluated only after the final model and feature set had been frozen.

---

## Methodology

The project follows a signal-processing-driven machine learning workflow:

```text
Raw I/Q signal
      |
      v
Handcrafted feature extraction
      |
      v
28-feature baseline representation
      |
      v
Classical ML model comparison
      |
      v
Random Forest baseline
      |
      v
SNR-dependent error analysis
      |
      v
Pairwise modulation analysis
      |
      v
Physically motivated feature engineering
      |
      v
29-feature final representation
      |
      v
Frozen Random Forest
      |
      v
Independent held-out test
```

SNR is used for stratification and diagnostic analysis, but it is **not provided to the Random Forest as an input feature**.

---

## Feature engineering

The original representation contains 28 handcrafted features extracted directly from each `(2, 128)` I/Q window.

They include time-domain statistics, higher-order statistics, phase-difference features, normalized autocorrelation and frequency-domain descriptors such as spectral entropy, bandwidth, flatness and spectral peak measurements.

During error analysis, one of the most persistent high-SNR problems was confusion between **QPSK and 8PSK**.

The original features could identify the PSK family but represented the modulation order poorly.

A physically motivated fourth-order circular moment was therefore introduced:

```python
circular_moment_4 = abs(
    mean(
        (x / (abs(x) + eps)) ** 4
    )
)
```

This feature captures fourth-order angular symmetry in the normalized complex signal.

The resulting final representation contains:

```text
28 baseline handcrafted features
+
1 circular_moment_4 feature
=
29 final features
```

On validation data, the change improved Macro-F1 from approximately **0.5493 to 0.5762**.

---

## Error analysis

A major goal of this project was to go beyond global accuracy and investigate the structure of model errors.

### Very low SNR

At very low SNR, most handcrafted features lose modulation separability because AWGN dominates the observed waveform.

The Random Forest shows a strong tendency to send uncertain examples toward the **AM-SSB decision region**.

This should not be interpreted as the signals physically becoming AM-SSB. Instead, the feature representation collapses under severe noise and the learned decision boundaries preferentially assign many low-information examples to that region.

At -20 dB, test accuracy is approximately **9.27%**, close to the random-chance level of `1/11 ≈ 9.09%`.

### QPSK vs 8PSK

The baseline model showed substantial QPSK/8PSK confusion even at high SNR.

The `circular_moment_4` feature substantially improves their separation.

At 18 dB on the independent test set:

```text
8PSK recall ≈ 0.93
QPSK recall ≈ 0.97
```

Their previous mutual confusion is largely removed.

### QAM16 vs QAM64

This remains one of the main residual errors.

At 18 dB:

```text
QAM16 recall ≈ 0.64
QAM64 recall ≈ 0.98
```

A significant fraction of QAM16 examples are still classified as QAM64.

Additional radial and cumulant-based features were investigated but did not improve end-to-end model performance and were therefore rejected.

### WBFM vs AM-DSB

This is the main remaining analog-modulation ambiguity.

At 18 dB, approximately 40% of WBFM examples are still classified as AM-DSB.

Instantaneous-frequency and spectral-sideband features were tested, but their improvement was too small to justify increasing model complexity.

---

## Models investigated

Several classical machine learning approaches were compared during development.

| Model | Approx. Macro-F1 |
|---|---:|
| Logistic Regression | 0.474 |
| HistGradientBoosting | 0.502 |
| Random Forest | 0.526 |

The Random Forest was selected as the main model.

Hyperparameter search did not provide a meaningful improvement over the baseline configuration, so the simpler frozen configuration was retained:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
```

A selective mixture-of-experts architecture was also tested for difficult modulation pairs, but its global improvement was negligible compared with its added complexity. It was therefore rejected.

---

## Repository structure

```text
snr-aware-modulation-classification/
|
├── app/
|   └── Interactive demo
|
├── configs/
|   └── Project configuration
|
├── data/
|   ├── raw/
|   └── processed/
|       Local data excluded from Git
|
├── models/
|   ├── final_rf_29_features.json
|   └── final_rf_29_features.joblib
|       Model binary stored locally and excluded from Git
|
├── notebooks/
|   ├── 00_environment_setup.ipynb
|   ├── 01_data_loading.ipynb
|   ├── 02_exploratory_analysis.ipynb
|   ├── 03_baseline_models.ipynb
|   ├── 04_model_tuning.ipynb
|   ├── 05_snr_evaluation.ipynb
|   ├── 06_error_analysis.ipynb
|   ├── 07_physical_feature_engineering.ipynb
|   └── 08_final_test.ipynb
|
├── reports/
|   └── Figures and technical results
|
├── scripts/
|   └── Reproducible execution scripts
|
├── src/
|   └── rfml/
|       ├── data.py
|       ├── features.py
|       ├── inference.py
|       └── splits.py
|
├── tests/
|
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

---

## Inference pipeline

The reusable inference code is located in:

```text
src/rfml/inference.py
```

A raw RadioML signal can be transformed into the final 29-feature representation and classified using the frozen model.

Conceptually:

```python
from rfml.inference import (
    load_model_and_metadata,
    predict_modulation,
)

model, metadata = load_model_and_metadata(
    model_path,
    metadata_path,
)

result = predict_modulation(
    sample,
    model,
    metadata,
)

print(result["predicted_modulation"])
print(result["confidence"])
print(result["probabilities"])
```

The metadata JSON stores the exact feature schema required by the classifier, ensuring that inference uses the same feature ordering as training.

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd snr-aware-modulation-classification

python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For the exact development environment:

```bash
pip install -r requirements-lock.txt
```

---

## Reproducibility

The raw RadioML dataset and large generated artifacts are intentionally excluded from Git.

The trained Random Forest binary is also excluded because its serialized size is approximately **95 MB**.

The repository instead versions the feature extraction code, model metadata, experimental notebooks and inference pipeline required to reproduce the methodology.

The final model metadata is stored in:

```text
models/final_rf_29_features.json
```

---

## Main conclusions

This project shows that AMC performance is highly dependent on SNR and that aggregate classification metrics alone can hide important representation failures.

At very low SNR, the handcrafted feature space becomes almost non-separable. At medium and high SNR, temporal, spectral and phase-related structure becomes increasingly informative.

Error analysis also showed that targeted, physically motivated feature engineering can outperform simply adding more generic features. In particular, the fourth-order circular moment substantially reduced QPSK/8PSK confusion and improved global Macro-F1 while adding only one feature.

The final 29-feature Random Forest reaches:

**Test Accuracy: 0.5562**

**Test Macro-F1: 0.5786**

with more than **90% test accuracy at 18 dB**.

The main remaining challenges are QAM16/QAM64 discrimination and WBFM/AM-DSB separation.

---

## Current status

The experimental machine learning phase is complete.

The final model, feature representation and held-out test results are frozen. Current development focuses on reproducibility, repository documentation and an interactive inference demo.