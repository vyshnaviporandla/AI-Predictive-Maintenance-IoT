import json
from pathlib import Path

import joblib


MODEL_PATH = Path("models") / "predictive_maintenance_model.pkl"


def predict_failure(model, sample):
    """Predict whether a device is likely to fail from a single dictionary of sensor values."""
    input_df = __import__("pandas").DataFrame([sample])
    input_df["temperature_pressure_ratio"] = input_df["temperature"] / input_df["pressure"].replace(0, 1)
    input_df["vibration_load_index"] = input_df["vibration"] * input_df["current"]
    input_df["rpm_current_ratio"] = input_df["rpm"] / input_df["current"].replace(0, 1)
    input_df["stress_index"] = (
        input_df["temperature"]
        + 2 * input_df["vibration"]
        + 0.5 * input_df["pressure"]
        + 0.8 * input_df["current"]
        + 0.04 * input_df["rpm"]
    )

    final_columns = [
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
        "temperature_pressure_ratio",
        "vibration_load_index",
        "rpm_current_ratio",
        "stress_index",
    ]

    result = model.predict(input_df[final_columns])[0]
    probability = model.predict_proba(input_df[final_columns])[0, 1]
    return int(result), float(probability)


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def build_alert(prediction, probability):
    """Create a simple alert message based on risk score."""
    if prediction == 1 and probability >= 0.6:
        return "CRITICAL ALERT: Machine likely to fail soon. Schedule maintenance immediately."
    if prediction == 1 and probability >= 0.4:
        return "WARNING: Increased risk detected. Inspect the machine soon."
    return "NORMAL: Machine operating within safe range."


def save_prediction_report(sample, prediction, probability):
    report = {
        "sample": sample,
        "prediction": int(prediction),
        "probability_of_failure": float(probability),
        "alert": build_alert(prediction, probability),
    }
    output = Path("outputs") / "latest_prediction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
