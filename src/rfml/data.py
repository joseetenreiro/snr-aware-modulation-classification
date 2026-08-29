from pathlib import Path
import pickle

import pandas as pd


def load_radioml(data_path: str | Path):
    """
    Load the RadioML 2016.10A dataset from a pickle file.

    Parameters
    ----------
    data_path : str or Path
        Path to RML2016.10a_dict.pkl.

    Returns
    -------
    dict
        Dictionary whose keys are (modulation, snr) tuples and whose
        values are NumPy arrays with shape (n_samples, 2, 128).
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "rb") as f:
        radioml = pickle.load(f, encoding="latin1")

    return radioml


def build_metadata(radioml):
    """
    Build a lightweight metadata table for the RadioML dataset.

    The signal arrays are not copied into the DataFrame.
    """
    records = []
    sample_id = 0

    for modulation, snr in sorted(radioml.keys()):
        n_samples = radioml[(modulation, snr)].shape[0]

        for array_index in range(n_samples):
            records.append(
                {
                    "sample_id": sample_id,
                    "modulation": modulation,
                    "snr": snr,
                    "array_index": array_index,
                }
            )

            sample_id += 1

    return pd.DataFrame(records)