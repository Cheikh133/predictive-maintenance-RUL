# src/model_evaluation.py
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, learning_curve


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute MAE and RMSE between true and predicted values.

    Args:
        y_true: array-like of shape (n_samples,) with true target values.
        y_pred: array-like of shape (n_samples,) with predicted values.

    Returns:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    mae = mean_absolute_error(y_true, y_pred)
    # avoid sklearn's 'squared' argument for compatibility
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, rmse


def plot_learning_curve(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    title: str = "Learning Curve",
    cv_splits: int = 3,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    scoring: str = "neg_mean_absolute_error",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot a learning curve for the given estimator using GroupKFold.

    Args:
        estimator: model implementing 'fit' and 'predict'.
        X: Training features.
        y: Target vector.
        groups: Group labels for CV splitting.
        title: Figure title.
        cv_splits: Number of folds.
        train_sizes: Training set proportions.
        scoring: Scoring metric.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    cv = GroupKFold(n_splits=cv_splits)
    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )
    train_mae = -np.mean(train_scores, axis=1)
    valid_mae = -np.mean(valid_scores, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes_abs, train_mae, marker="o", label="Train MAE")
    ax.plot(train_sizes_abs, valid_mae, marker="o", label="CV MAE")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
