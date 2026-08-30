from pathlib import Path
from urllib.request import urlretrieve
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# Project paths
# ==================================================

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
    / "final_rf_29_features_compressed.joblib"
)

MODEL_URL = (
    "https://github.com/joseetenreiro/"
    "snr-aware-modulation-classification/"
    "releases/download/v1.0.0/"
    "final_rf_29_features_compressed.joblib"
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


# ==================================================
# Streamlit configuration
# ==================================================

st.set_page_config(
    page_title="Automatic Modulation Classification",
    page_icon="📡",
    layout="wide",
)


# ==================================================
# Model download
# ==================================================

def ensure_model_exists():
    """
    Download the frozen Random Forest model from the
    GitHub Release when it is not available locally.
    """

    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        MODEL_PATH.parent
        / f"{MODEL_PATH.name}.part"
    )

    try:
        with st.spinner(
            "Downloading trained model "
            "(~84 MB, first launch only)..."
        ):
            urlretrieve(
                MODEL_URL,
                temporary_path,
            )

        temporary_path.replace(
            MODEL_PATH
        )

    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()

        st.error(
            "The trained model could not be downloaded."
        )

        st.exception(exc)
        st.stop()


# ==================================================
# Cached resource loaders
# ==================================================

@st.cache_resource
def load_classifier():
    """
    Load the frozen classifier and its metadata.
    """

    ensure_model_exists()

    return load_model_and_metadata(
        MODEL_PATH,
        METADATA_PATH,
    )


@st.cache_data
def load_demo_dataset():
    """
    Load the lightweight RadioML demonstration subset.
    """

    demo = np.load(
        DEMO_DATA_PATH,
        allow_pickle=True,
    )

    return {
        "samples": demo["samples"],
        "modulation": demo["modulation"],
        "snr": demo["snr"],
        "original_index": demo["original_index"],
    }


# ==================================================
# Header
# ==================================================

st.title(
    "📡 Automatic Modulation Classification"
)

st.caption(
    "RadioML 2016.10A · "
    "Random Forest · "
    "29 handcrafted signal features"
)

st.write(
    """
    Explore how an Automatic Modulation Classification
    system behaves as signal quality changes.

    Select a modulation, SNR and I/Q example from the
    sidebar and inspect the model prediction, confidence,
    signal representation and class probabilities.
    """
)


# ==================================================
# Required repository files
# ==================================================

if not METADATA_PATH.exists():
    st.error(
        f"Model metadata not found:\n\n{METADATA_PATH}"
    )
    st.stop()


if not DEMO_DATA_PATH.exists():
    st.error(
        f"Demo dataset not found:\n\n{DEMO_DATA_PATH}"
    )
    st.stop()


# ==================================================
# Load resources
# ==================================================

with st.spinner("Loading classifier..."):
    model, metadata = load_classifier()


demo = load_demo_dataset()


# ==================================================
# Sidebar — signal selection
# ==================================================

st.sidebar.header(
    "Signal selection"
)


modulations = sorted(
    np.unique(
        demo["modulation"]
    ).tolist()
)


snr_values = sorted(
    np.unique(
        demo["snr"]
    ).tolist()
)


selected_modulation = st.sidebar.selectbox(
    "True modulation",
    modulations,
    index=(
        modulations.index("QPSK")
        if "QPSK" in modulations
        else 0
    ),
)


selected_snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=snr_values,
    value=18,
)


mask = (
    (
        demo["modulation"]
        == selected_modulation
    )
    &
    (
        demo["snr"]
        == selected_snr
    )
)


available_samples = (
    demo["samples"][mask]
)

available_indices = (
    demo["original_index"][mask]
)


sample_position = st.sidebar.slider(
    "Example index",
    min_value=0,
    max_value=len(available_samples) - 1,
    value=0,
)


sample = available_samples[
    sample_position
]

original_index = available_indices[
    sample_position
]


st.sidebar.caption(
    f"Original RadioML index: "
    f"{int(original_index)}"
)

st.sidebar.caption(
    "Public demo subset: "
    f"{len(modulations)} modulations · "
    f"{len(snr_values)} SNR levels"
)


# ==================================================
# Prediction
# ==================================================

result = predict_modulation(
    sample,
    model,
    metadata,
)


prediction = result[
    "predicted_modulation"
]

confidence = result[
    "confidence"
]

probabilities = result[
    "probabilities"
]


# ==================================================
# Main classification result
# ==================================================

st.subheader(
    f"Classification result — {selected_snr} dB"
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "True modulation",
    selected_modulation,
)


col2.metric(
    "Predicted modulation",
    prediction,
)


col3.metric(
    "Confidence",
    f"{confidence:.1%}",
)


if prediction == selected_modulation:
    st.success(
        "Correct classification"
    )

else:
    st.warning(
        f"Misclassification: "
        f"{selected_modulation} → {prediction}"
    )


# ==================================================
# Signal visualization
# ==================================================

i_signal = sample[0]
q_signal = sample[1]


st.subheader(
    "Signal visualization"
)


signal_col, iq_col = st.columns(2)


# --------------------------------------------------
# Temporal I/Q signal
# --------------------------------------------------

with signal_col:

    st.markdown(
        "#### Temporal I/Q"
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        i_signal,
        label="I",
    )

    ax.plot(
        q_signal,
        label="Q",
    )

    ax.set_xlabel(
        "Sample"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        f"{selected_modulation} — "
        f"{selected_snr} dB"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# --------------------------------------------------
# I/Q plane
# --------------------------------------------------

with iq_col:

    st.markdown(
        "#### I/Q plane"
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.scatter(
        i_signal,
        q_signal,
        alpha=0.7,
        s=25,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        "In-phase (I)"
    )

    ax.set_ylabel(
        "Quadrature (Q)"
    )

    ax.set_title(
        "Observed I/Q samples"
    )

    ax.grid(
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ==================================================
# Model output
# ==================================================

st.subheader(
    "Model output"
)


probability_df = pd.DataFrame(
    {
        "Modulation":
            list(
                probabilities.keys()
            ),

        "Probability":
            list(
                probabilities.values()
            ),
    }
)


probability_df = (
    probability_df
    .sort_values(
        "Probability",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


prob_col, table_col = st.columns(
    [1.5, 1]
)


# --------------------------------------------------
# Probability chart
# --------------------------------------------------

with prob_col:

    st.markdown(
        "#### Class probabilities"
    )

    probability_chart_df = (
        probability_df
        .set_index(
            "Modulation"
        )
    )

    st.bar_chart(
        probability_chart_df,
        height=350,
    )


# --------------------------------------------------
# Prediction ranking
# --------------------------------------------------

with table_col:

    st.markdown(
        "#### Prediction ranking"
    )

    display_df = (
        probability_df.copy()
    )

    display_df[
        "Probability"
    ] = display_df[
        "Probability"
    ].map(
        lambda value:
            f"{value:.2%}"
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=350,
    )


# ==================================================
# Prediction interpretation
# ==================================================

with st.expander(
    "How to interpret this prediction"
):

    st.write(
        f"""
        The classifier predicts **{prediction}**
        with a confidence of **{confidence:.1%}**.

        The selected signal has an SNR of
        **{selected_snr} dB**.

        SNR is used only to select and analyze the
        RadioML example. It is **not provided as an
        input feature to the Random Forest**.
        """
    )


    if selected_snr < 0:

        st.info(
            """
            At negative SNR, additive noise increasingly
            dominates the observed I/Q samples.

            Modulation-specific structure becomes less
            separable, so classification uncertainty and
            error rates are expected to increase.
            """
        )


# ==================================================
# Model information
# ==================================================

with st.expander(
    "Model information"
):

    model_col1, model_col2 = (
        st.columns(2)
    )


    model_col1.write(
        "**Classifier:** Random Forest"
    )

    model_col1.write(
        f"**Features:** "
        f"{metadata['n_features']}"
    )

    model_col1.write(
        "**Feature set:** "
        f"`{metadata['feature_set']}`"
    )

    model_col1.write(
        "**Dataset:** RadioML 2016.10A"
    )


    model_col2.write(
        "**Held-out test accuracy:** "
        f"{metadata['final_test']['accuracy']:.2%}"
    )

    model_col2.write(
        "**Held-out test Macro-F1:** "
        f"{metadata['final_test']['macro_f1']:.2%}"
    )

    model_col2.write(
        "**Mean accuracy for SNR ≥ 0 dB:** "
        "85.35%"
    )

    model_col2.write(
        "**Accuracy at 18 dB:** "
        "90.30%"
    )


    st.divider()

    st.write(
        """
        The final model uses 28 baseline handcrafted
        signal features plus the physically motivated
        `circular_moment_4` feature.

        The model configuration and held-out test
        evaluation were frozen before deployment.
        """
    )


# ==================================================
# Demo information
# ==================================================

with st.expander(
    "About the public demo"
):

    st.write(
        """
        To keep the application lightweight, this demo
        uses a small representative subset of RadioML
        2016.10A rather than the complete dataset.

        The public subset contains all 11 modulation
        classes at four representative SNR levels:

        - -10 dB
        - 0 dB
        - 10 dB
        - 18 dB

        Ten I/Q examples are included for each
        modulation/SNR combination.

        The classifier itself is the frozen final model
        used for the reported project results.
        """
    )