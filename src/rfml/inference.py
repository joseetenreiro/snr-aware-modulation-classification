import json

import joblib
import numpy as np
import pandas as pd

from rfml.features import extract_features


def load_model_and_metadata(
    model_path,
    metadata_path,
):
    model = joblib.load(model_path)

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return model, metadata


def sample_to_feature_vector(
    sample,
    feature_columns,
):
    sample = np.asarray(sample)

    if sample.shape != (2, 128):
        raise ValueError(
            "Expected sample shape (2, 128), "
            f"received {sample.shape}."
        )

    features = extract_features(
        sample,
        feature_set="full_final",
    )

    feature_df = pd.DataFrame(
        [features]
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{missing_features}"
        )

    return feature_df[
        feature_columns
    ]


def predict_modulation(
    sample,
    model,
    metadata,
):
    feature_columns = metadata[
        "feature_columns"
    ]

    X = sample_to_feature_vector(
        sample,
        feature_columns,
    )

    predicted_modulation = model.predict(
        X
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    probability_by_class = dict(
        zip(
            model.classes_,
            probabilities,
        )
    )

    return {
        "predicted_modulation":
            predicted_modulation,

        "confidence":
            float(
                np.max(probabilities)
            ),

        "probabilities":
            {
                str(label): float(prob)
                for label, prob
                in probability_by_class.items()
            },
    }