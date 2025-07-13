from typing import Any, Tuple  # on retire Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, learning_curve


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute MAE and RMSE between true and predicted values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    (mae, rmse) : tuple of floats
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
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

    Returns
    -------
    fig : matplotlib.figure.Figure
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

    fig = plt.figure(figsize=figsize)
    plt.plot(train_sizes_abs, train_mae, marker="o", label="Train MAE")
    plt.plot(train_sizes_abs, valid_mae, marker="o", label="CV MAE")
    plt.xlabel("Training Set Size")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig
