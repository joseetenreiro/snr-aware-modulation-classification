from pathlib import Path
import pickle
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from rfml.inference import (
    load_model_and_metadata,
    predict_modulation,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_rf_29_features.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_rf_29_features.json"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RML2016.10a"
    / "RML2016.10a_dict.pkl"
)


print("Project root:", PROJECT_ROOT)

print(
    "Model exists:",
    MODEL_PATH.exists(),
)

print(
    "Metadata exists:",
    METADATA_PATH.exists(),
)

print(
    "Raw dataset exists:",
    RAW_DATA_PATH.exists(),
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model, metadata = load_model_and_metadata(
    MODEL_PATH,
    METADATA_PATH,
)

print(
    "\nModel loaded successfully."
)

print(
    "Expected features:",
    len(metadata["feature_columns"]),
)


# --------------------------------------------------
# Load RadioML dataset
# --------------------------------------------------

with open(
    RAW_DATA_PATH,
    "rb",
) as f:
    radioml = pickle.load(
        f,
        encoding="latin1",
    )


# --------------------------------------------------
# Select one known example
# --------------------------------------------------

true_modulation = "QPSK"
snr = 18
sample_index = 0

sample = radioml[
    (
        true_modulation,
        snr,
    )
][sample_index]


print(
    "\nSample shape:",
    sample.shape,
)

print(
    "True modulation:",
    true_modulation,
)

print(
    "SNR:",
    snr,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

result = predict_modulation(
    sample,
    model,
    metadata,
)


print("\n--- Prediction ---")

print(
    "Predicted modulation:",
    result["predicted_modulation"],
)

print(
    "Confidence:",
    f"{result['confidence']:.2%}",
)


print("\nTop probabilities:")

top_probabilities = sorted(
    result["probabilities"].items(),
    key=lambda item: item[1],
    reverse=True,
)

for modulation, probability in top_probabilities[:5]:

    print(
        f"{modulation:8s}: "
        f"{probability:.2%}"
    )