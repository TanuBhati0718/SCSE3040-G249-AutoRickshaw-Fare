# AutoRickshaw Fare Estimation (SCSE3040-G249)

## Project Overview

This repository contains a Python-based machine learning project that estimates auto-rickshaw fares using features such as distance, waiting time, passengers, hour of day, luggage, fuel price, booking channel, and zone.

The project follows the multi-stage MLOps workflow defined in the SCSE3040 mini-project assignment.

## Team Members

- **Member 1:** Kanishk Dhiman
- **Member 2:** Tanu Bhati

## Data

- Raw training data: `data/raw/train.csv`
- Later data for exploratory analysis: `data/raw/later.csv`
- Hold-out features for final evaluation: `data/raw/holdout_features.csv`

DVC is used to version-control the large CSV files.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd SCSE3040-G249-AutoRickshaw-Fare

# Install dependencies
pip install -r requirements.txt

# Download DVC-tracked data
dvc pull
```

---

## Stage 1: Data Investigation

### Stage 1 Checklist

- [x] Repository created and pushed to GitHub.
- [x] Both participating members listed in this README.
- [x] DVC initialized.
- [x] `train.csv` and `later.csv` tracked using DVC.
- [x] Data investigation script created.
- [x] Seven required data checks implemented.
- [x] Findings recorded in `reports/findings.yaml` under `data_issues`.

### Run Data Investigation

```bash
python -m src.data_investigation
```

The script investigates seven data problems, including leaking columns, sentinel values, and duplicate rows, and updates `reports/findings.yaml`.

---

## Stage 2: Data Cleaning and MLflow Tracking

### Task 1 — Cleaning Code

Implemented configuration-driven data cleaning using an in-memory copy, preserving the original raw CSV files.

Cleaning operations include:
- Removing duplicate rows.
- Replacing the `9999` distance sentinel.
- Correcting fares entered ten times too large.

**Files:** `params.yaml`, `src/fare_model/data.py`

**Result:** 1051 raw rows were reduced to 1000 cleaned rows. Four pytest tests passed.

```bash
python -m pytest -q
```

### Task 2 — Train v1 with MLflow Tracking

Implemented the first tracked model training run.

The run logs:
- `data_sha256`
- `mae`
- Model artifacts

The trained model is saved as `models/model.joblib`, and experiment metadata is stored in `mlflow.db`.

**Files:** `src/fare_model/model.py`, `.gitignore`, `mlflow.db`, `models/model.joblib`

**Result:**
- Run name: `RandomForest_v1`
- Run ID: `b4fed6fe016140c5a0411a3da97b74b7`
- MAE: `17.5171`
- Four pytest tests passed.

```bash
python -m src.fare_model.model
python -m pytest -q
```

View the MLflow experiment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Task 3 — Try 3+ Settings

Added `model_config` and `run_name` parameters to `train_model()` to support multiple hyperparameter configurations.

Three training runs were completed.

**File:** `src/fare_model/model.py`

**Results:**

| Run | MAE |
|---|---:|
| RandomForest_v1 | 17.5171 |
| RandomForest_v2 | 17.9935 |
| RandomForest_v3 | 17.1684 |

Four pytest tests passed.

### Task 4 — Register Best Model and Champion Alias

Registered the best-performing run, RandomForest_v3, in the MLflow Model Registry.

**Registered model:** `MP072_AutoRickshaw_Fare_Model`

**Version:** 1

**Alias:** `champion`

**Run ID:** `45028a3dcaab4fe38e0c288408a6fa59`

**Best recorded MAE:** `17.1684`

Updated the tracking fields in `reports/findings.yaml`.

### Task 5 — Write Tests

Created automated tests using FastAPI TestClient.

**File:** `tests/test_pipeline.py`

Four tests cover:
1. `GET /health` endpoint.
