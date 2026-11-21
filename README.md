# AutoRickshaw Fare Estimation (SCSE3040‑G249)

## Project Overview
This repository contains a Python project that estimates auto‑rickshaw fares based on a set of features (distance, wait time, passengers, hour of day, luggage, fuel price, booking channel, and zone). The work follows the multi‑stage workflow defined in the course assignment.

## Team Members
- **Member 1:** Kanishk Dhiman
- **Member 2:** Tanu Bhati



## Data
- Raw training data: `data/raw/train.csv`
- Later data for exploratory analysis: `data/raw/later.csv`
- Hold‑out features for final evaluation: `data/raw/holdout_features.csv`

DVC is used to version‑control these large CSV files.

## Setup
```bash
# Clone the repo
git clone <repo‑url>
cd SCSE3040-G249-AutoRickshaw-Fare

# Install dependencies
pip install -r requirements.txt

# Initialize DVC (already done in the repo)
dvc pull   # fetch data files tracked by DVC
```

## Stage 1 Checklist (completed)
- Repository created and pushed to GitHub.
- All four members listed in this README.
- DVC initialized; `train.csv` and `later.csv` are tracked (`*.dvc` files present).
- Data investigation script `src/data_investigation.py` runs the seven required checks.
- Findings recorded in `reports/findings.yaml` under the `data_issues` section.

## How to Run the Data Investigation
```bash
python -m src.data_investigation
```
The script prints a summary of the seven data problems (leaking column, sentinel values, duplicate rows, etc.) and updates `reports/findings.yaml`.

## Stage 2: Cleaning and MLflow tracking

### Task 1 — Cleaning code

Cleaning is configuration-driven and works on an in-memory copy, so the raw CSV is unchanged. It removes duplicate rows, replaces the `9999` distance sentinel, and corrects fares entered ten times too large.

Files: `params.yaml`, `src/fare_model/data.py`.

Result: 1051 raw rows become 1000 cleaned rows; `python -m pytest -q` passes 4 tests.

```bash
source .venv312/bin/activate
python -m pytest -q
```

Service: `uvicorn app.main:app --host 127.0.0.1 --port 8000`

### Task 2 — Train v1 with MLflow tracking

Added the first tracked training run. The run logs `data_sha256` and `mae`, saves `models/model.joblib`, and stores its metadata in `mlflow.db`. Model registration is added in a later task.

Files: `src/fare_model/model.py`, `.gitignore`, `mlflow.db`, `models/model.joblib`.

Result: run `b4fed6fe016140c5a0411a3da97b74b7` (`RandomForest_v1`) finished with MAE `17.5171`; `python -m pytest -q` passes 4 tests.

```bash
source .venv312/bin/activate
python -m src.fare_model.model
python -m pytest -q
```

View the experiment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Service:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use it:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km":5.2,"wait_min":4,"passengers":2,"hour":14,"luggage_bags":1,"fuel_price_rs":105,"booking":"app","zone":"Market"}'
```

### Task 3 — Try 3+ settings

Added `model_config` and `run_name` parameters to `train_model()` to support multiple hyperparameter configurations. Ran two additional training runs with different settings.

Files: `src/fare_model/model.py`.

Result: 3 runs total — RandomForest_v1 (MAE 17.5171), RandomForest_v2 (MAE 17.9935), RandomForest_v3 (MAE 17.1684); `python -m pytest -q` passes 4 tests.

```bash
source .venv312/bin/activate
python -m src.fare_model.model --help  # shows new params
python -m pytest -q
```

View all runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Service:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Task 4 — Register best model + champion alias

Registered the best run (RandomForest_v3, MAE 17.1684) as model version 1 and set the `champion` alias. Updated `reports/findings.yaml` tracking fields.

Files: `src/fare_model/model.py`, `reports/findings.yaml`, `mlflow.db`, `models/model.joblib`.

Result: Registered model `MP072_AutoRickshaw_Fare_Model` version 1 with `@champion` alias pointing to run `45028a3dcaab4fe38e0c288408a6fa59` (MAE 17.1684); `python -m pytest -q` passes 4 tests.

```bash
source .venv312/bin/activate
python -m src.fare_model.model  # baseline
python -m pytest -q
```

View registered model:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Service (uses champion model):

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Task 5 — Write tests

Tests satisfy handbook requirements: 3+ pytest tests with at least one TestClient test.

Files: `tests/test_pipeline.py`.

Result: 4 tests using TestClient — `GET /health`, `POST /predict` valid input, rejects negative distance, rejects high fuel price; all pass.

```bash
source .venv312/bin/activate
python -m pytest -q
```

## Next Stages
- Stage 2: Cleaning code, MLflow tracking, model registration.
- Stage 3: FastAPI service, Docker, CI/CD.
- Stage 4: PSI analysis, retraining, final submission.

---
*Generated by Antigravity assistant to provide a clean, informative README.*