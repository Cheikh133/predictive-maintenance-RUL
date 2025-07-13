from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score


def load_feature_data(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train_features.csv and test_features.csv into X/y sets.
    """
    train = pd.read_csv(data_dir / "train_features.csv")
    test = pd.read_csv(data_dir / "test_features.csv")
    feats = [
        c for c in train.columns if c not in ["unit_number", "time_in_cycles", "RUL"]
    ]
    X_train, y_train = train[feats], train["RUL"]
    X_test, y_test = test[feats], test["true_RUL"]
    return X_train, X_test, y_train, y_test


def evaluate_baselines(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """
    Cross-validate simple linear and tree ensemble baselines.
    """
    cv = GroupKFold(n_splits=5)
    mae_s = make_scorer(mean_absolute_error, greater_is_better=False)
    rmse_s = make_scorer(
        lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False
    )
    results = []
    # Linear models
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    for name, model in [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
        ("Lasso", Lasso(alpha=0.1, random_state=random_state)),
    ]:
        mae = -cross_val_score(model, X, y, scoring=mae_s, cv=cv, groups=groups).mean()
        rmse = -cross_val_score(
            model, X, y, scoring=rmse_s, cv=cv, groups=groups
        ).mean()
        results.append({"model": name, "MAE": mae, "RMSE": rmse})
    # Ensembles
    for name, model in [
        (
            "RandomForest",
            RandomForestRegressor(n_estimators=100, random_state=random_state),
        ),
        ("LightGBM", LGBMRegressor(random_state=random_state)),
    ]:
        mae = -cross_val_score(model, X, y, scoring=mae_s, cv=cv, groups=groups).mean()
        rmse = -cross_val_score(
            model, X, y, scoring=rmse_s, cv=cv, groups=groups
        ).mean()
        results.append({"model": name, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(results)


def tune_random_forest(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = 42
) -> Dict[str, Any]:
    """
    RandomizedSearchCV for RandomForest hyperparameters.
    """
    cv = GroupKFold(n_splits=3)
    mae_s = make_scorer(mean_absolute_error, greater_is_better=False)
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    dist = {
        "n_estimators": randint(50, 200),
        "max_depth": [None] + list(range(5, 21, 5)),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
    }
    rs = RandomizedSearchCV(
        rf,
        dist,
        n_iter=20,
        scoring=mae_s,
        cv=cv,
        groups=groups,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(X, y)
    return rs.best_params_


def tune_lightgbm(
    X: pd.DataFrame,
    y: Any,
    groups: Any,
    random_state: int = 42,
    n_trials: int = 30,
) -> Dict[str, Any]:
    """
    Optuna tuning for LightGBM with GroupKFold and early stopping.
    """

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "l1",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": random_state,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        }
        cv = GroupKFold(n_splits=3)
        scores = []
        for tr, val in cv.split(X, y, groups):
            X_tr, X_val = X.iloc[tr], X.iloc[val]
            y_tr = y.iloc[tr] if hasattr(y, "iloc") else y[tr]
            y_val = y.iloc[val] if hasattr(y, "iloc") else y[val]
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_val, label=y_val)
            gbm = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(stopping_rounds=20)],
            )
            preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
            scores.append(mean_absolute_error(y_val, preds))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_lightgbm(
    X: pd.DataFrame,
    y: Any,
    groups: Any,
    random_state: int = 42,
    n_trials: int = 30,
    n_estimators: int = 100,
) -> LGBMRegressor:
    """
    Full LightGBM pipeline: tune + train final.
    """
    best = tune_lightgbm(X, y, groups, random_state, n_trials)
    model = LGBMRegressor(
        objective="regression",
        metric="l1",
        boosting_type="gbdt",
        random_state=random_state,
        n_estimators=n_estimators,
        **best,
    )
    model.fit(X, y)
    return model


def train_and_save_final(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    model_path: Path,
    random_state: int = 42,
) -> LGBMRegressor:
    """
    Train final model with given params and dump to disk.
    """
    model = LGBMRegressor(
        objective="regression",
        metric="l1",
        boosting_type="gbdt",
        random_state=random_state,
        n_estimators=100,
        **best_params,
    )
    model.fit(X, y)
    dump(model, model_path)
    return model


if __name__ == "__main__":
    # Example CLI usage
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    X_tr, X_te, y_tr, y_te = load_feature_data(data_dir)
    groups = pd.read_csv(data_dir / "train_features.csv")["unit_number"]
    print("Baselines:\n", evaluate_baselines(X_tr, y_tr, groups))
    rf_p = tune_random_forest(X_tr, y_tr, groups)
    print("RF params:", rf_p)
    lgb_p = tune_lightgbm(X_tr, y_tr, groups)
    print("LGBM params:", lgb_p)
    path = Path(__file__).resolve().parent.parent / "models" / "final_lgb.joblib"
    train_and_save_final(X_tr, y_tr, lgb_p, path)
    print("Saved model to", path)
