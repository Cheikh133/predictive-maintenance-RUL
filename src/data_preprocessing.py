# src/data_preprocessing.py

from pathlib import Path
import pandas as pd


# Project directory constants
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_raw_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw training, test and RUL datasets from FD001 files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing 'train_FD001.txt', 'test_FD001.txt' and 'RUL_FD001.txt'.

    Returns
    -------
    train_df : DataFrame
        Raw training data.
    test_df : DataFrame
        Raw test data.
    rul_df : DataFrame
        Raw RUL data.
    """
    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    train_df = pd.read_csv(train_path, sep=" ", header=None).dropna(axis=1, how="all")
    test_df = pd.read_csv(test_path, sep=" ", header=None).dropna(axis=1, how="all")
    rul_df = pd.read_csv(rul_path, sep=" ", header=None)
    return train_df, test_df, rul_df


def assign_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to match FD001 documentation.

    Parameters
    ----------
    df : DataFrame
        DataFrame with unnamed columns.

    Returns
    -------
    DataFrame
        With columns: unit_number, time_in_cycles, operational_settings, sensors...
    """
    base_cols = [
        "unit_number", "time_in_cycles",
        "operational_setting_1", "operational_setting_2", "operational_setting_3",
    ]
    sensor_cols = [f"sensor_measurement_{i}" for i in range(1, 22)]
    df.columns = base_cols + sensor_cols
    return df


def drop_non_informative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop constant or non-informative features identified in EDA.

    Parameters
    ----------
    df : DataFrame
        Dataset with sensor and operational_setting columns.

    Returns
    -------
    DataFrame
        With non-informative columns removed.
    """
    to_drop = [
        "operational_setting_3",
        "sensor_measurement_1",
        "sensor_measurement_5",
        "sensor_measurement_10",
        "sensor_measurement_16",
        "sensor_measurement_18",
        "sensor_measurement_19",
    ]
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing)


def clean_rul(rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the first column (RUL) and rename it.

    Parameters
    ----------
    rul_df : DataFrame
        Raw RUL DataFrame (possibly with empty trailing columns).

    Returns
    -------
    DataFrame
        Single-column ['RUL'] dataset.
    """
    if rul_df.shape[1] > 1:
        # drop any extra empty columns
        rul_df = rul_df.loc[:, [0]]
    rul_df.columns = ["RUL"]
    return rul_df


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unit, compute RUL = max_cycle - time_in_cycles.

    Parameters
    ----------
    df : DataFrame
        Training data after renaming and dropping.

    Returns
    -------
    DataFrame
        With new 'RUL' column.
    """
    max_cycle = (
        df.groupby("unit_number")["time_in_cycles"]
        .max()
        .reset_index(name="max_cycle")
    )
    df = df.merge(max_cycle, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    return df.drop(columns=["max_cycle"])


def compute_test_true_rul(
    test_df: pd.DataFrame, rul_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build final test set: select last cycle per unit and attach true_RUL.

    Parameters
    ----------
    test_df : DataFrame
        Test data after renaming and dropping.
    rul_df : DataFrame
        Cleaned RUL DataFrame.

    Returns
    -------
    DataFrame
        Test set with 'true_RUL' as target.
    """
    last = test_df.groupby("unit_number").last().reset_index()
    merged = pd.concat(
        [last.reset_index(drop=True), rul_df.reset_index(drop=True)],
        axis=1,
    )
    merged = merged.rename(columns={"RUL": "true_RUL"})
    return merged


def save_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Save processed train and test DataFrames to CSV.

    Parameters
    ----------
    train_df : DataFrame
        Final training set with RUL.
    test_df : DataFrame
        Final test set with true_RUL.
    out_dir : Path
        Directory where processed CSVs are saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_processed.csv", index=False)
    test_df.to_csv(out_dir / "test_processed.csv", index=False)


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full preprocessing pipeline correspondng to Notebook 01_EDA.

    Steps
    -----
    1. load_raw_data
    2. assign_column_names
    3. drop_non_informative
    4. clean_rul
    5. compute_train_rul
    6. compute_test_true_rul
    7. save_processed

    Returns
    -------
    train_processed : DataFrame
    test_processed : DataFrame
    """
    train_raw, test_raw, rul_raw = load_raw_data()
    train = assign_column_names(train_raw)
    test = assign_column_names(test_raw)
    train = drop_non_informative(train)
    test = drop_non_informative(test)
    rul = clean_rul(rul_raw)
    train_proc = compute_train_rul(train)
    test_proc = compute_test_true_rul(test, rul)
    save_processed(train_proc, test_proc)
    return train_proc, test_proc