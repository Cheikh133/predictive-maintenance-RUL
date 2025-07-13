# src/feature_engineering.py

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_processed_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned train and test datasets from processed folder.

    Parameters
    ----------
    data_dir : Path
        Path to 'data/processed' directory.

    Returns
    -------
    train_df : DataFrame
        Processed training data.
    test_df : DataFrame
        Processed test data.
    """
    train_df = pd.read_csv(data_dir / "train_processed.csv")
    test_df = pd.read_csv(data_dir / "test_processed.csv")
    return train_df, test_df


def generate_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling mean and std for each sensor per engine.

    Parameters
    ----------
    df : DataFrame
        Input data with 'unit_number' and sensor columns.
    window : int
        Rolling window size.

    Returns
    -------
    DataFrame
        With added '<sensor>_rolling_mean' and '<sensor>_rolling_std' columns.
    """
    result = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_measurement_")]
    for col in sensor_cols:
        grp = result.groupby("unit_number")[col]
        result[f"{col}_rolling_mean"] = (
            grp.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        result[f"{col}_rolling_std"] = (
            grp.rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        )
    return result


def fill_rolling_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in all rolling_std features with 0.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing '<sensor>_rolling_std' columns.

    Returns
    -------
    DataFrame
        With NaNs replaced by 0 in rolling_std columns.
    """
    result = df.copy()
    std_cols = [c for c in result.columns if c.endswith("_rolling_std")]
    for col in std_cols:
        result[col] = result[col].fillna(0)
    return result


def compute_cycle_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized cycle ratio per engine: time_in_cycles / max_cycle.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'unit_number' and 'time_in_cycles'.

    Returns
    -------
    DataFrame
        With added 'cycle_ratio' column.
    """
    result = df.copy()
    max_cycle = result.groupby("unit_number")["time_in_cycles"].transform("max")
    result["cycle_ratio"] = result["time_in_cycles"] / max_cycle
    return result


def generate_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference of each sensor measurement to its previous cycle.

    Parameters
    ----------
    df : DataFrame
        DataFrame with original sensor columns (no rolling or delta).

    Returns
    -------
    DataFrame
        With added '<sensor>_delta' columns.
    """
    result = df.copy()
    base_sensors = [
        c
        for c in df.columns
        if c.startswith("sensor_measurement_")
        and "_rolling" not in c
        and "_delta" not in c
    ]
    for col in base_sensors:
        result[f"{col}_delta"] = result.groupby("unit_number")[col].diff().fillna(0)
    return result


def drop_constant_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop features that have zero variance in the test set.

    Parameters
    ----------
    train_df : DataFrame
        Training DataFrame.
    test_df : DataFrame
        Test DataFrame.

    Returns
    -------
    train_df_clean, test_df_clean : Tuple[DataFrame, DataFrame]
        DataFrames with constant columns removed.
    """
    constant_cols = [c for c in test_df.columns if test_df[c].nunique() <= 1]
    return train_df.drop(columns=constant_cols), test_df.drop(columns=constant_cols)


def scale_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standard scale specified feature columns on train and apply to test.

    Parameters
    ----------
    train_df : DataFrame
    test_df : DataFrame
    feature_cols : List[str]
        Columns to scale.

    Returns
    -------
    train_scaled, test_scaled, scaler : Tuple
        Scaled DataFrames and the fitted scaler.
    """
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def drop_correlated_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, drop_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop a pre-specified list of correlated features from both DataFrames.

    Parameters
    ----------
    train_df : DataFrame
    test_df : DataFrame
    drop_cols : List[str]
        Columns to remove.

    Returns
    -------
    train_df_clean, test_df_clean : Tuple[DataFrame, DataFrame]
    """
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols)


def save_engineered_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path
) -> None:
    """
    Save the final engineered train/test sets to CSV.

    Parameters
    ----------
    train_df : DataFrame
    test_df : DataFrame
    out_dir : Path
        Directory to save 'train_features.csv' and 'test_features.csv'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_features.csv", index=False)
    test_df.to_csv(out_dir / "test_features.csv", index=False)


def run_feature_engineering(
    train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full feature engineering pipeline.

    Steps
    -----
    1. Rolling mean & std
    2. Fill rolling std NaNs
    3. Cycle ratio
    4. Delta features
    5. Drop constant features
    6. Scale features
    7. Drop highly correlated features
    8. Save engineered CSVs

    Parameters
    ----------
    train_df : DataFrame
        Output from data_preprocessing.run_preprocessing().
    test_df : DataFrame
    out_dir : Path, optional
        Where to save outputs (defaults to project/data/processed).

    Returns
    -------
    train_feat, test_feat : Tuple[DataFrame, DataFrame]
        Engineered training and testing sets.
    """
    # Infer out_dir if not provided
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    # 1. Rolling + fill std
    train_feat = generate_rolling_features(train_df)
    train_feat = fill_rolling_std(train_feat)
    test_feat = generate_rolling_features(test_df)
    test_feat = fill_rolling_std(test_feat)

    # 2. Cycle ratio
    train_feat = compute_cycle_ratio(train_feat)
    test_feat = compute_cycle_ratio(test_feat)

    # 3. Delta features
    train_feat = generate_delta_features(train_feat)
    test_feat = generate_delta_features(test_feat)

    # 4. Drop constant features
    train_feat, test_feat = drop_constant_features(train_feat, test_feat)

    # 5. Scale
    feat_cols = [
        c
        for c in train_feat.columns
        if c not in ["unit_number", "time_in_cycles", "RUL"]
    ]
    train_feat, test_feat, _ = scale_features(train_feat, test_feat, feat_cols)

    # 6. Drop correlated
    corr_drop = [
        "sensor_measurement_14",
        "sensor_measurement_13",
        "sensor_measurement_11",
        "sensor_measurement_12",
    ]
    train_feat, test_feat = drop_correlated_features(train_feat, test_feat, corr_drop)

    # 7. Save to CSV
    save_engineered_data(train_feat, test_feat, out_dir)

    return train_feat, test_feat
