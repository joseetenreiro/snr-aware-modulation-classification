from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RML2016.10a"
    / "RML2016.10a_dict.pkl"
)

DEMO_DIR = (
    PROJECT_ROOT
    / "data"
    / "demo"
)

DEMO_PATH = (
    DEMO_DIR
    / "radioml_demo_samples.npz"
)


SELECTED_SNRS = [
    -10,
    0,
    10,
    18,
]

SAMPLES_PER_COMBINATION = 10


DEMO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("Loading RadioML dataset...")

with open(
    RAW_DATA_PATH,
    "rb",
) as f:
    radioml = pickle.load(
        f,
        encoding="latin1",
    )


modulations = sorted(
    {
        modulation
        for modulation, snr
        in radioml.keys()
    }
)


samples = []
labels = []
snrs = []
indices = []


for modulation in modulations:

    for snr in SELECTED_SNRS:

        signals = radioml[
            (
                modulation,
                snr,
            )
        ]

        for index in range(
            SAMPLES_PER_COMBINATION
        ):

            samples.append(
                signals[index]
            )

            labels.append(
                modulation
            )

            snrs.append(
                snr
            )

            indices.append(
                index
            )


samples = np.asarray(
    samples,
    dtype=np.float32,
)

labels = np.asarray(labels)

snrs = np.asarray(snrs)

indices = np.asarray(indices)


np.savez_compressed(
    DEMO_PATH,
    samples=samples,
    modulation=labels,
    snr=snrs,
    original_index=indices,
)


print("\nDemo dataset created.")
print("Path:", DEMO_PATH)
print("Samples:", len(samples))
print("Shape:", samples.shape)
print("Modulations:", len(modulations))
print("SNR levels:", SELECTED_SNRS)