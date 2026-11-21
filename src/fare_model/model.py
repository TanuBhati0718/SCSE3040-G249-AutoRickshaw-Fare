
from pathlib import Path
import hashlib

import joblib
import mlflow
import mlflow.sklearn

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


if __name__ == "__main__":
    train_model()
