import os

import numpy as np
import pandas as pd
import pytest

from src import (
    data_preprocessing,
    feature_engineering,
    model_evaluation,
    model_training,
    model_utils,
)


# 1. Data preprocessing
def test_preprocessing_creates_processed_files():
    train_df, test_df = data_preprocessing.run_preprocessing()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    for col in ("unit_number", "time_in_cycles", "RUL"):
        assert col in train_df.columns
    assert "true_RUL" in test_df.columns
    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0


# 2. Feature engineering
def test_feature_engineering_outputs():
    train_raw = pd.read_csv(os.path.join("data", "processed", "train_processed.csv"))
    test_raw = pd.read_csv(os.path.join("data", "processed", "test_processed.csv"))
    train_feat, test_feat = feature_engineering.run_feature_engineering(
        train_raw, test_raw
    )
    assert isinstance(train_feat, pd.DataFrame)
    assert isinstance(test_feat, pd.DataFrame)
    assert train_feat.shape[1] > train_raw.shape[1]
    assert test_feat.shape[1] > test_raw.shape[1]


# 3. Model utils
def test_model_utils_save_and_load(tmp_path, monkeypatch):
    dummy = {"foo": 123}
    filename = "test_model.joblib"
    monkey_dir = tmp_path / "models"
    monkeypatch.setattr(model_utils, "MODELS_DIR", monkey_dir)
    model_utils.save_model(dummy, filename)
    path = model_utils.get_model_path(filename)
    assert path.exists()
    loaded = model_utils.load_model(filename)
    assert loaded == dummy


# 4. Model training
def test_train_lightgbm_returns_fitted_model():
    X = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.rand(20)
    groups = np.arange(20) // 4
    model = model_training.train_lightgbm(
        X, y, groups, random_state=0, n_trials=2, n_estimators=10
    )
    from lightgbm import LGBMRegressor

    assert isinstance(model, LGBMRegressor)
    assert hasattr(model, "n_estimators")


# 5. Model evaluation
def test_compute_metrics_and_plot(tmp_path):
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.5, 0.8, 2.2])
    mae, rmse = model_evaluation.compute_metrics(y_true, y_pred)
    assert pytest.approx(mae, rel=1e-3) == np.mean(np.abs(y_true - y_pred))
    assert pytest.approx(rmse, rel=1e-3) == np.sqrt(np.mean((y_true - y_pred) ** 2))
    import matplotlib.pyplot as plt

    fig = model_evaluation.plot_learning_curve(
        estimator=model_training.train_lightgbm(
            pd.DataFrame(np.random.rand(20, 5)),
            np.random.rand(20),
            np.arange(20) // 4,
            random_state=0,
            n_trials=1,
            n_estimators=5,
        ),
        X=pd.DataFrame(np.random.rand(20, 5)),
        y=np.random.rand(20),
        groups=np.arange(20) // 4,
    )
    assert isinstance(fig, plt.Figure)
    fig.savefig(tmp_path / "curve.png")
