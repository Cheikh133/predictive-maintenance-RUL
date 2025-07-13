import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from joblib import load

from src import (
    data_preprocessing,
    feature_engineering,
    model_evaluation,
    model_training,
    model_utils,
)


def test_preprocessing_creates_processed_files(tmp_path, monkeypatch):
    """
    Ensure preprocessing pipeline runs and outputs valid DataFrames
    with expected columns.
    """
    train_df, test_df = data_preprocessing.run_preprocessing()
    # Check types
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    # Check required columns
    for col in ("unit_number", "time_in_cycles", "RUL"):
        assert col in train_df.columns
    assert "true_RUL" in test_df.columns
    # At least one row exists
    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0


def test_feature_engineering_outputs(tmp_path):
    """
    Validate that feature engineering adds new columns to the data.
    """
    train_raw = pd.read_csv(os.path.join("data", "processed", "train_processed.csv"))
    test_raw = pd.read_csv(os.path.join("data", "processed", "test_processed.csv"))
    train_feat, test_feat = feature_engineering.run_feature_engineering(
        train_raw, test_raw
    )
    # Check types
    assert isinstance(train_feat, pd.DataFrame)
    assert isinstance(test_feat, pd.DataFrame)
    # New features should increase column count
    assert train_feat.shape[1] > train_raw.shape[1]
    assert test_feat.shape[1] > test_raw.shape[1]


def test_model_utils_save_and_load(tmp_path, monkeypatch):
    """
    Test save_model and load_model with a dummy object.
    """
    dummy = {"foo": 123}
    filename = "test_model.joblib"
    # Redirect model directory to temporary path
    monkey_dir = tmp_path / "models"
    monkeypatch.setattr(model_utils, "MODELS_DIR", monkey_dir)
    model_utils.save_model(dummy, filename)
    # Verify file exists
    path = model_utils.get_model_path(filename)
    assert path.exists()
    # Load and compare
    loaded = model_utils.load_model(filename)
    assert loaded == dummy


def test_train_lightgbm_returns_fitted_model():
    """
    Ensure train_lightgbm returns a fitted LGBMRegressor.
    """
    # Generate dummy data
    X = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.rand(20)
    groups = np.arange(20) // 4  # simulate 5 groups
    model = model_training.train_lightgbm(
        X, y, groups, random_state=0, n_trials=2, n_estimators=10
    )
    from lightgbm import LGBMRegressor

    assert isinstance(model, LGBMRegressor)
    # The model should have the n_estimators attribute set
    assert hasattr(model, "n_estimators")


def test_compute_metrics_and_plot(tmp_path):
    """
    Verify compute_metrics returns correct MAE/RMSE and
    plot_learning_curve returns a matplotlib Figure.
    """
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.5, 0.8, 2.2])
    mae, rmse = model_evaluation.compute_metrics(y_true, y_pred)
    # Check numeric accuracy
    assert pytest.approx(mae, rel=1e-3) == np.mean(np.abs(y_true - y_pred))
    assert pytest.approx(rmse, rel=1e-3) == np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Generate a learning curve plot
    import matplotlib.pyplot as plt

    dummy_estimator = model_training.train_lightgbm(
        pd.DataFrame(np.random.rand(20, 5)),
        np.random.rand(20),
        np.arange(20) // 4,
        random_state=0,
        n_trials=1,
        n_estimators=5,
    )
    fig = model_evaluation.plot_learning_curve(
        estimator=dummy_estimator,
        X=pd.DataFrame(np.random.rand(20, 5)),
        y=np.random.rand(20),
        groups=np.arange(20) // 4,
    )
    # Confirm we got a Figure
    assert isinstance(fig, plt.Figure)
    # That it can be saved without error
    fig.savefig(tmp_path / "curve.png")
