from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "current",
    "humidity",
    "coolant_temp",
    "power_draw",
    "device_age_hours",
    "maintenance_gap_days",
]


def load_and_prepare_data(data_path="data/industrial_iot_failure_data.csv"):
    """Load the dataset, clean it, engineer features, and split for training."""
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    df = df.drop_duplicates().copy()
    df["failure"] = df["failure"].fillna(0).astype(int)

    df["temperature_pressure_ratio"] = df["temperature"] / df["pressure"].replace(0, 1)
    df["vibration_load_index"] = df["vibration"] * df["current"]
    df["rpm_current_ratio"] = df["rpm"] / df["current"].replace(0, 1)
    df["stress_index"] = (
        df["temperature"]
        + 2 * df["vibration"]
        + 0.5 * df["pressure"]
        + 0.8 * df["current"]
        + 0.04 * df["rpm"]
    )

    feature_columns = [
        col for col in df.columns if col not in {"device_id", "timestamp", "failure"}
    ]

    X = df[feature_columns]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cleaned_path = data_path.with_name("cleaned_iot_data.csv")
    df.to_csv(cleaned_path, index=False)

    return X_train, X_test, y_train, y_test, feature_columns
