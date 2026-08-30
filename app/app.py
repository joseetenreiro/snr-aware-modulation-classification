from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project paths
# --------------------------------------------------

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


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Automatic Modulation Classification",
    page_icon="📡",
    layout="wide",
)


# --------------------------------------------------
# Cached loaders
# --------------------------------------------------

@st.cache_resource
def load_classifier():
    return load_model_and_metadata(
        MODEL_PATH,
        METADATA_PATH,
    )


@st.cache_resource
def load_radioml():
    with open(
        RAW_DATA_PATH,
        "rb",
    ) as f:
        return pickle.load(
            f,
            encoding="latin1",
        )


# --------------------------------------------------
# Header
# --------------------------------------------------

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
    system behaves as signal quality changes. Select a
    modulation, SNR and I/Q example from the sidebar.
    """
)


# --------------------------------------------------
# Check local files
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        f"Model not found: {MODEL_PATH}"
    )
    st.stop()

if not METADATA_PATH.exists():
    st.error(
        f"Metadata not found: {METADATA_PATH}"
    )
    st.stop()

if not RAW_DATA_PATH.exists():
    st.error(
        f"RadioML dataset not found: {RAW_DATA_PATH}"
    )
    st.stop()


# --------------------------------------------------
# Load resources
# --------------------------------------------------

with st.spinner("Loading classifier..."):
    model, metadata = load_classifier()

with st.spinner("Loading RadioML dataset..."):
    radioml = load_radioml()


# --------------------------------------------------
# Dataset controls
# --------------------------------------------------

st.sidebar.header("Signal selection")

modulations = sorted(
    {
        modulation
        for modulation, snr in radioml.keys()
    }
)

snr_values = sorted(
    {
        snr
        for modulation, snr in radioml.keys()
    }
)


selected_modulation = st.sidebar.selectbox(
    "True modulation",
    modulations,
    index=modulations.index("QPSK")
    if "QPSK" in modulations
    else 0,
)

selected_snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=snr_values,
    value=18,
)

signals = radioml[
    (
        selected_modulation,
        selected_snr,
    )
]

sample_index = st.sidebar.slider(
    "Example index",
    min_value=0,
    max_value=len(signals) - 1,
    value=0,
)

sample = signals[sample_index]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

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


# --------------------------------------------------
# --------------------------------------------------
# Main result
# --------------------------------------------------

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
    st.success("Correct classification")
else:
    st.warning(
        f"Misclassification: "
        f"{selected_modulation} → {prediction}"
    )


# --------------------------------------------------
# Signal visualizations
# --------------------------------------------------

i_signal = sample[0]
q_signal = sample[1]

st.subheader("Signal visualization")

signal_col, iq_col = st.columns(2)


# Temporal I/Q signal
with signal_col:

    st.markdown("#### Temporal I/Q")

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

    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

    ax.set_title(
        f"{selected_modulation} — {selected_snr} dB"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# I/Q plane
with iq_col:

    st.markdown("#### I/Q plane")

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

    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")

    ax.set_title(
        "Observed I/Q samples"
    )

    ax.grid(alpha=0.25)

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


# --------------------------------------------------
# Probabilities
# --------------------------------------------------

st.subheader("Model output")

probability_df = pd.DataFrame(
    {
        "Modulation":
            list(probabilities.keys()),

        "Probability":
            list(probabilities.values()),
    }
)

probability_df = (
    probability_df
    .sort_values(
        "Probability",
        ascending=False,
    )
    .reset_index(drop=True)
)


prob_col, table_col = st.columns(
    [1.5, 1]
)


# Probability chart
with prob_col:

    st.markdown(
        "#### Class probabilities"
    )

    probability_chart_df = (
        probability_df
        .set_index("Modulation")
    )

    st.bar_chart(
        probability_chart_df,
        height=350,
    )


# Probability table
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


# --------------------------------------------------
# Interpretation
# --------------------------------------------------

with st.expander(
    "How to interpret this prediction"
):

    st.write(
        f"""
        The classifier predicts **{prediction}**
        with a confidence of **{confidence:.1%}**.

        The selected example has an SNR of
        **{selected_snr} dB**.

        SNR is used here only to select and analyze
        the RadioML sample. It is **not provided as
        an input feature to the classifier**.
        """
    )

    if selected_snr < 0:

        st.info(
            """
            At negative SNR, noise increasingly
            dominates the received I/Q signal.
            Classification uncertainty is therefore
            expected to increase.
            """
        )


# --------------------------------------------------
# Model information
# --------------------------------------------------

with st.expander(
    "Model information"
):

    model_col1, model_col2 = st.columns(2)

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

    model_col2.write(
        "**Held-out test accuracy:** "
        f"{metadata['final_test']['accuracy']:.2%}"
    )

    model_col2.write(
        "**Held-out test Macro-F1:** "
        f"{metadata['final_test']['macro_f1']:.2%}"
    )

    model_col2.write(
        "**Accuracy at 18 dB:** 90.30%"
    )