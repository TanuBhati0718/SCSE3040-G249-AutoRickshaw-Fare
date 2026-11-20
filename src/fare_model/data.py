
from pathlib import Path

import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# These are the eight input features required by the project API.
NUMERIC_FEATURES = [
    "distance_km",
    "wait_min",
    "passengers",
    "hour",
    "luggage_bags",
    "fuel_price_rs",
]

CATEGORICAL_FEATURES = [
    "booking",
    "zone",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "fare_rs"


def load_params(config_path="params.yaml"):
    """Load project settings from the YAML configuration file."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_csv(file_path):
    """Read a CSV file without modifying the original file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def clean_training_data(df):
    """
    Apply confirmed basic cleaning to a copy of the training data.

    The original CSV remains unchanged.
    """
    data = df.copy()

    # Remove exact duplicate rows.
    data = data.drop_duplicates()

    # Convert the known distance sentinel into a missing value.
    data["distance_km"] = pd.to_numeric(
        data["distance_km"], errors="coerce"
    )
    data.loc[data["distance_km"] == 9999, "distance_km"] = float("nan")

    # Ensure the target is numeric and remove rows without a target.
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN], errors="coerce"
    )
    data = data.dropna(subset=[TARGET_COLUMN])

    # Keep only rows with a valid target.
    data = data.reset_index(drop=True)

    return data


def prepare_features_and_target(df):
    """Separate the eight model features from the fare target."""
    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_preprocessor():
    """Build preprocessing for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def load_and_prepare_training_data(config_path="params.yaml"):
    """Load, clean and prepare the training dataset."""
    config = load_params(config_path)
    train_path = config["data"]["train_path"]

    raw_data = load_csv(train_path)
    cleaned_data = clean_training_data(raw_data)

    X, y = prepare_features_and_target(cleaned_data)

    return X, y, cleaned_data
