
from pathlib import Path
import hashlib

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.fare_model.data import (
    build_preprocessor,
    load_and_prepare_training_data,
)


def get_file_sha256(file_path):
    """Calculate SHA256 fingerprint of the original dataset."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def train_model(
    config_path="params.yaml",
    model_config=None,
    run_name="RandomForest_v1",
):
    """Train, evaluate, save and track the fare prediction model."""

    from src.fare_model.data import load_params

    config = load_params(config_path)

    # Load prepared training data.
    X, y, cleaned_data = load_and_prepare_training_data(config_path)

    data_path = config["data"]["train_path"]
    data_sha256 = get_file_sha256(data_path)

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Read model settings, allowing caller overrides.
    resolved_model_config = {
        **config["model"],
        **(model_config or {}),
    }

    regressor = RandomForestRegressor(
        n_estimators=resolved_model_config["n_estimators"],
        max_depth=resolved_model_config["max_depth"],
        min_samples_split=resolved_model_config["min_samples_split"],
        random_state=resolved_model_config["random_state"],
        n_jobs=-1,
    )

    # Combine preprocessing and model into one pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )

    # Configure MLflow with a local SQLite database.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(config["tracking"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:

        # Train the model.
        pipeline.fit(X_train, y_train)

        # Evaluate on unseen test data.
        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        # Log parameters and evaluation results.
        mlflow.log_params(resolved_model_config)
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))

        mlflow.log_metric("mae", mae)

        # Save the model locally.
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "model.joblib"
        joblib.dump(pipeline, model_path)

        # Log the model artifact without registering a version yet.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print("\nModel training completed successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"MAE: {mae:.4f}")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print(f"Model saved at: {model_path}")
        print(f"Dataset SHA256: {data_sha256}")

        return pipeline, mae, run.info.run_id


def register_best_model(
    config_path="params.yaml",
    experiment_name=None,
    alias="champion",
):
    """Register the best run as model version 1 and set the champion alias."""
    from src.fare_model.data import load_params

    config = load_params(config_path)
    tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    resolved_experiment_name = (
        experiment_name or config["tracking"]["experiment_name"]
    )
    experiment = mlflow.get_experiment_by_name(resolved_experiment_name)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {resolved_experiment_name}"
        )

    best_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and metrics.mae >= 0"
        ),
        order_by=["metrics.mae ASC"],
        max_results=1,
    )

    if best_runs.empty:
        raise ValueError(
            "No finished MLflow runs with an mae metric found."
        )

    best_run = best_runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_mae = float(best_run["metrics.mae"])

    model_name = config["tracking"]["registered_model_name"]
    model_version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name,
    )

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    client.set_registered_model_alias(
        model_name,
        alias,
        model_version.version,
    )

    champion = mlflow.sklearn.load_model(
        f"models:/{model_name}@{alias}"
    )
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, models_dir / "model.joblib")

    print("\nChampion model registered successfully!")
    print(f"Best run ID: {best_run_id}")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Registered model: {model_name}")
    print(f"Version: {model_version.version}")
    print(f"Alias: {alias}")
    print(f"Model saved at: {models_dir / 'model.joblib'}")

    return model_version.version, best_run_id, best_mae


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train fare prediction model with MLflow tracking")
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="JSON string of model hyperparameter overrides, e.g. '{\"n_estimators\": 100, \"max_depth\": 8}'"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="RandomForest_v1",
        help="MLflow run name"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="params.yaml",
        help="Path to params.yaml"
    )
    args = parser.parse_args()

    model_config = json.loads(args.model_config) if args.model_config else None
    train_model(config_path=args.config_path, model_config=model_config, run_name=args.run_name)
