# predictive-maintenance-RUL

[![CI](https://github.com/Cheikh133/predictive-maintenance-RUL/actions/workflows/ci.yml/badge.svg)](https://github.com/Cheikh133/predictive-maintenance-RUL/actions/workflows/ci.yml)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-blue?style=flat-square&logo=streamlit&logoColor=white&label=Live%20Demo&labelColor=555555&color=007ACC&cacheSeconds=3600)](https://predictive-maintenance-rul-wth7dklhnwcvyzgzoyp2ry.streamlit.app/)

**Objective:** Build and deploy an end‑to‑end pipeline to predict Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS data, with automated preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit dashboard.

## Table of Contents

1. [Key Technologies](#key-technologies)
2. [Pipeline Overview](#pipeline-overview)  
3. [Dataset: NASA CMAPSS FD001](#dataset-nasa-cmapss-fd001)  
4. [Installation](#installation)  
5. [Development](#development)  
6. [Author](#author)  

## Key Technologies

- **Python 3.11**  
- **Pandas & NumPy** — data manipulation and numerical computing  
- **scikit‑learn** —  
  - *Preprocessing*: `StandardScaler`, `GroupKFold`  
  - *Linear models*: `LinearRegression`, `Ridge`, `Lasso`  
  - *Metrics*: `mean_absolute_error`, `mean_squared_error`  
- **LightGBM**, **XGBoost** — gradient boosting frameworks  
- **Optuna** — hyperparameter optimization  
- **Streamlit** — interactive dashboard  
- **Altair** — declarative visualization  
- **SHAP** — model explainability via Shapley values 


## Pipeline Overview

Below is a concise view of the end‑to‑end flow, from raw data ingestion to interactive dashboard.


| Stage             | Script                            | Inputs                            | Outputs                             |
|-------------------|-----------------------------------|-----------------------------------|-------------------------------------|
| **Preprocess** | `src/data_preprocessing.py`       | `data/raw/*.txt`                  | `data/processed/*_processed.csv`    |
| **Feature Eng.** | `src/feature_engineering.py`    | `data/processed/*_processed.csv`  | `data/processed/*_features.csv`     |
| **Train**      | `src/model_training.py`           | `data/processed/*_features.csv`   | `models/final_lgb.joblib`           |
| **Evaluate**   | `src/model_evaluation.py`         | model + test features             | MAE, RMSE & learning‑curve figures  |
| **Dashboard**  | `dashboard/app.py`                | model + `test_features.csv`       | Interactive Streamlit UI            |

## Data Description: CMAPSS FD001

The FD001 subset of the NASA CMAPSS dataset contains run-to-failure trajectories for turbofan engines under a single operating condition (sea level) and a single fault mode (HPC degradation).

- **Training set**
  - Trajectories: 100 engines
  - Cycles per trajectory: variable, until failure
- **Test set**
  - Trajectories: 100 engines
  - Cycles per trajectory: ends before failure
  - True RUL provided in `RUL_FD001.txt`

| Column                       | Description                                                |
|:-----------------------------|:-----------------------------------------------------------|
| `unit_number`                | Engine identifier                                          |
| `time_in_cycles`             | Cycle index                                                |
| `operational_setting_1–3`    | Three engine operating parameters (e.g. temp, pressure)   |
| `sensor_measurement_1–21`    | 21 multivariate sensor readings (temp, pressure, vibration, flow) |

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

## Author

**Cheikh LO**  
*Data Scientist & Statistical Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Cheikh%20LO-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/cheikh-lo-531701193/)  
[![GitHub](https://img.shields.io/badge/GitHub-cheikh133-black?logo=github&style=flat-square)](https://github.com/cheikh133)