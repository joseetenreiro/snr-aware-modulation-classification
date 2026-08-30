from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from rfml.features import extract_features


ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_rf_29_features.joblib"
)

COMPRESSED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_rf_29_features_compressed.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_rf_29_features.json"
)

DEMO_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "radioml_demo_samples.npz"
)


# --------------------------------------------------
# Load original model and metadata
# --------------------------------------------------

print("Loading original model...")

original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


feature_columns = metadata[
    "feature_columns"
]


# --------------------------------------------------
# Save with maximum joblib compression
# --------------------------------------------------

print("Saving compressed model...")

joblib.dump(
    original_model,
    COMPRESSED_MODEL_PATH,
    compress=9,
)


# --------------------------------------------------
# Compare sizes
# --------------------------------------------------

original_size_mb = (
    ORIGINAL_MODEL_PATH.stat().st_size
    / (1024 ** 2)
)

compressed_size_mb = (
    COMPRESSED_MODEL_PATH.stat().st_size
    / (1024 ** 2)
)

print()
print(
    f"Original size:   "
    f"{original_size_mb:.2f} MB"
)

print(
    f"Compressed size: "
    f"{compressed_size_mb:.2f} MB"
)


# --------------------------------------------------
# Load compressed model
# --------------------------------------------------

compressed_model = joblib.load(
    COMPRESSED_MODEL_PATH
)


# --------------------------------------------------
# Load lightweight demo samples
# --------------------------------------------------

demo = np.load(
    DEMO_DATA_PATH,
    allow_pickle=True,
)

samples = demo["samples"]

print()
print(
    "Demo samples:",
    samples.shape,
)


# --------------------------------------------------
# Extract the same 29 features
# --------------------------------------------------

feature_rows = []

for sample in samples:

    features = extract_features(
        sample,
        feature_set="full_final",
    )

    feature_rows.append(
        features
    )


X_demo = pd.DataFrame(
    feature_rows
)

X_demo = X_demo[
    feature_columns
]


# --------------------------------------------------
# Compare predictions
# --------------------------------------------------

original_predictions = (
    original_model.predict(
        X_demo
    )
)

compressed_predictions = (
    compressed_model.predict(
        X_demo
    )
)


same_predictions = np.array_equal(
    original_predictions,
    compressed_predictions,
)


print()
print(
    "Predictions identical:",
    same_predictions,
)