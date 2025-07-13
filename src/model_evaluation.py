# src/model_evaluation.py

from typing import Any, Dict, Tuple

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
    mae : float
        Mean Absolute Error
    rmse : float
        Root Mean Squared Error
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
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

    Parameters
    ----------
    estimator : estimator object
        Model implementing 'fit' and 'predict'.
    X : array-like of shape (n_samples, n_features)
        Training vector.
    y : array-like of shape (n_samples,)
        Target vector.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset.
    title : str
        Title for the chart.
    cv_splits : int
        Number of folds for GroupKFold cross-validation.
    train_sizes : array-like
        Relative or absolute numbers of training examples that will be used.
    scoring : str
        Scoring metric for learning curve.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plotted figure.
    """
    cv = GroupKFold(n_splits=cv_splits)
    sizes, train_scores, valid_scores = learning_curve(
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
    ax.plot(sizes, train_mae, marker="o", label="Train MAE")
    ax.plot(sizes, valid_mae, marker="o", label="CV MAE")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
