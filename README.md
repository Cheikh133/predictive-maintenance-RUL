# predictive-maintenance-RUL

[![CI](https://github.com/Cheikh133/predictive-maintenance-RUL/actions/workflows/ci.yml/badge.svg)](https://github.com/Cheikh133/predictive-maintenance-RUL/actions/workflows/ci.yml)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?style=flat-square&logo=streamlit&logoColor=white&label=Live%20Demo&labelColor=555555&color=007ACC&cacheSeconds=3600)](https://predictive-maintenance-rul-wth7dklhnwcvyzgzoyp2ry.streamlit.app/)

## Table of Contents

1. [Project Description](#project-description)  
2. [Detailed Steps](#detailed-steps)  
3. [Data Description](#data-description)  
4. [Installation](#installation)  
5. [Development](#development)  
6. [Simplified Pipeline](#simplified-data--model--dashboard-pipeline)  


## Project Description

An end‑to‑end pipeline for Remaining Useful Life (RUL) prediction on turbofan engine data (NASA CMAPSS FD001), covering:

| Stage                   | Script / Notebook                       | Inputs                                                                 | Outputs                                                           |
|-------------------------|-----------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| **Data Preprocessing**  | `src/data_preprocessing.py`<br>`01_EDA.ipynb`   | `data/raw/train_FD001.txt`<br>`data/raw/test_FD001.txt`<br>`data/raw/RUL_FD001.txt` | `data/processed/train_processed.csv`<br>`data/processed/test_processed.csv` |
| **Feature Engineering** | `src/feature_engineering.py`<br>`02_Feature_Engineering.ipynb` | Processed CSVs (`*_processed.csv`)                                     | `data/processed/train_features.csv`<br>`data/processed/test_features.csv`      |
| **Model Training**      | `src/model_training.py`<br>`03_Modeling.ipynb`       | Engine features (`*_features.csv`)                                     | `models/final_lgb.joblib`                                          |
| **Model Evaluation**    | `src/model_evaluation.py`                | `y_true`, `y_pred` arrays                                               | MAE, RMSE, learning‑curve Figure                                   |
| **Dashboard**           | `dashboard/app.py`, plus modules        | `data/processed/test_features.csv`<br>`models/final_lgb.joblib`         | Interactive Streamlit dashboard (Overview, Signals, Diagnostics, Explainability) |


## Detailed Steps


- **Data preprocessing** (`src/data_preprocessing.py`, inspired by `notebooks/01_EDA.ipynb`)
  Load and clean the raw CMAPSS FD001 data, drop empty and non-informative columns, compute RUL for every cycle in the training set, select only the last cycle per engine with its true RUL for testing, and save processed CSVs.
  - **Input** → `data/raw/train_FD001.txt`, `data/raw/test_FD001.txt`, `data/raw/RUL_FD001.txt`
  - **Output** → `data/processed/train_processed.csv`, `data/processed/test_processed.csv`

- **Feature engineering** (`src/feature_engineering.py`, inspired by `notebooks/02_Feature_Engineering.ipynb`)
  - Rolling-window stats (mean, std over 5 cycles) to capture short-term trends and variability
  - Delta features (difference from previous cycle) to encode cycle-to-cycle changes
  - Cycle ratio (current cycle / max cycle) to normalize the engine’s life stage
  - Drop zero-variance features (constant signals) and drop highly correlated sensors to reduce redundancy
  - Standard scaling (zero mean, unit variance) for all features
  - Save engineered datasets
  - **Input** → `data/processed/train_processed.csv`, `data/processed/test_processed.csv`
  - **Output** → `data/processed/train_features.csv`, `data/processed/test_features.csv`

- **Model training** (`src/model_training.py`, inspired by `notebooks/03_Modeling.ipynb`)
  Compare baselines (Linear Regression, Ridge, Lasso, Random Forest, LightGBM), tune Random Forest hyperparameters with `RandomizedSearchCV`, tune LightGBM with Optuna, train the final LightGBM model, and serialize it.
  - **Input** → `data/processed/train_features.csv`, `data/processed/test_features.csv`
  - **Output** → `models/final_lgb.joblib`

- **Model evaluation** (`src/model_evaluation.py`)
  - `compute_metrics(y_true, y_pred)` : returns MAE and RMSE
  - `plot_learning_curve(estimator, X, y, groups, …)` : returns a `matplotlib.figure.Figure` showing train vs CV MAE as training set size grows

- **Streamlit Dashboard** (`dashboard/app.py`)  
  An interactive dashboard organized into four tabs:

  - **Overview (Test Set)**  
    - **Select Engine Unit** → dropdown selector  
    - **Cycle at Measurement** & **Life Stage** (cycle_ratio)  
    - **Predicted vs True RUL**  
    - **Global MAE / RMSE** (test set)  
  - **Raw Sensor & Settings** → display raw feature table for chosen engine  
  - **Diagnostics** → error distribution, bias, RMSE, scatter True vs Predicted  
  - **Explainability (SHAP)** → global & local SHAP plots

  - **Input**  → Processed test features: `data/processed/test_features.csv` & Trained model artifact:   `models/final_lgb.joblib`
  - **Output** → interactive metrics & charts  
  
- **CI/CD** (GitHub Actions)
  - Install dependencies via `pip install -r requirements.txt`
  - Run linters to enforce code quality and consistency:
    - **Black** – opinionated code formatter that automatically reformats Python source to a canonical style.
    - **isort** – sorts and groups `import` statements according to configurable rules, keeping them tidy.
    - **Flake8** – static code analysis to catch bugs, style violations (PEP8), unused variables/imports, etc.
  - Execute unit tests with `pytest` to ensure each component behaves as expected

## Data Description: CMAPSS FD001

The FD001 subset of the NASA CMAPSS dataset contains run-to-failure trajectories for turbofan engines under a single operating condition (sea level) and a single fault mode (HPC degradation).

- **Training set**
  - Trajectories: 100 engines
  - Cycles per trajectory: variable, until failure
- **Test set**
  - Trajectories: 100 engines
  - Cycles per trajectory: ends before failure
  - True RUL provided in `RUL_FD001.txt`

| Column                                     | Description                                                        |
|--------------------------------------------|--------------------------------------------------------------------|
| `unit_number`                              | Engine (unit) identifier                                           |
| `time_in_cycles`                           | Elapsed cycles since start of run                                  |
| `operational_setting_1`–`operational_setting_3` | Three operational parameters affecting engine performance     |
| `sensor_measurement_1`–`sensor_measurement_21` | 21 distinct sensor readings (temperatures, pressures, speeds, vibrations, flow rates, etc.) |

**File formats**
- `train_FD001.txt` / `test_FD001.txt`: space-delimited, no header
- `RUL_FD001.txt`: one RUL value per test unit

**Processed outputs**
- `data/processed/train_processed.csv` (adds `RUL` for each cycle)
- `data/processed/test_processed.csv`  (last cycle per unit + `true_RUL`)

## Installation

```bash
git clone https://github.com/Cheikh133/predictive-maintenance-RUL.git
cd predictive-maintenance-RUL
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Development

To contribute, run linters, tests and pre-commit hooks locally:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows PowerShell

# 2. Install development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install \
  pre-commit \
  black \
  isort \
  flake8 \
  pytest

# 3. Install Git hooks
pre-commit install

# 4. Format & sort imports
pre-commit run --all-files

# 5. Lint
flake8 .

# 6. Run tests
pytest -q --disable-warnings --maxfail=1
```


## Simplified Pipeline

| Stage                   | Input                        | Script                          | Output                                   |
|-------------------------|------------------------------|---------------------------------|------------------------------------------|
| **1. Preprocessing**    | `data/raw/*.txt`             | `src/data_preprocessing.py`     | `data/processed/*_processed.csv`         |
| **2. Feature Eng.**     | `*_processed.csv`            | `src/feature_engineering.py`    | `data/processed/*_features.csv`          |
| **3. Training**         | `*_features.csv`             | `src/model_training.py`         | `models/final_lgb.joblib`                |
| **4. Evaluation**       | Model + test features        | `src/model_evaluation.py`       | MAE, RMSE, learning-curve figures        |
| **5. Dashboard**        | Model + test features        | `dashboard/app.py`              | Interactive Streamlit web UI             |
