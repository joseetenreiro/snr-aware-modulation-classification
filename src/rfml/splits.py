from sklearn.model_selection import train_test_split


def create_stratified_split(
    metadata,
    train_size=0.70,
    validation_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Split RadioML metadata into train, validation and test sets.

    Stratification is performed jointly by modulation and SNR.
    """

    if not abs(train_size + validation_size + test_size - 1.0) < 1e-9:
        raise ValueError("Split proportions must sum to 1.")

    metadata = metadata.copy()

    # Example:
    # QPSK_18
    # BPSK_-20
    metadata["stratum"] = (
        metadata["modulation"].astype(str)
        + "_"
        + metadata["snr"].astype(str)
    )

    train, temp = train_test_split(
        metadata,
        test_size=(validation_size + test_size),
        random_state=random_state,
        stratify=metadata["stratum"],
    )

    relative_test_size = test_size / (validation_size + test_size)

    validation, test = train_test_split(
        temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp["stratum"],
    )

    train = train.drop(columns="stratum").reset_index(drop=True)
    validation = validation.drop(columns="stratum").reset_index(drop=True)
    test = test.drop(columns="stratum").reset_index(drop=True)

    return train, validation, test