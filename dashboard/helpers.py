# dashboard/helpers.py

import pandas as pd

from src.model_utils import get_processed_path, load_model


def load_rul_model():
    """
    Load and return the trained LightGBM model.
    """
    return load_model("final_lgb.joblib")


def load_test_data() -> pd.DataFrame:
    """
    Load and return the test features DataFrame.
    """
    return pd.read_csv(get_processed_path("test_features.csv"))


def get_unit_ids(df: pd.DataFrame) -> list[int]:
    """
    Extract sorted list of engine unit IDs from the test DataFrame.
    """
    return sorted(df["unit_number"].unique().tolist())


def get_unit_row(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    """
    Return the single-row DataFrame corresponding to the selected engine unit.
    """
    return df[df["unit_number"] == unit].reset_index(drop=True)


def extract_features(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier and target columns, returning feature matrix for prediction.
    """
    return row_df.drop(
        columns=["unit_number", "time_in_cycles", "true_RUL"], errors="ignore"
    )
