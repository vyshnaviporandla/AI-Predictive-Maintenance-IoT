import numpy as np
import pandas as pd
from pathlib import Path


def generate_iot_dataset(output_path="data/industrial_iot_failure_data.csv", n_samples=3000, seed=42):
    """Create a synthetic IoT sensor dataset for predictive maintenance."""
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_id = np.arange(1, n_samples + 1)
    timestamp = pd.date_range("2024-01-01 00:00:00", periods=n_samples, freq="15min")

    temperature = rng.normal(72, 12, n_samples)
    vibration = rng.normal(1.5, 0.75, n_samples)
    pressure = rng.normal(110, 18, n_samples)
    rpm = rng.normal(1250, 260, n_samples)
    current = rng.normal(28, 10, n_samples)
    humidity = rng.normal(45, 12, n_samples)
    coolant_temp = rng.normal(72, 16, n_samples)
    power_draw = rng.normal(40, 9, n_samples)
    device_age_hours = rng.integers(100, 6000, size=n_samples)
    maintenance_gap_days = rng.integers(2, 180, size=n_samples)

    # Add realistic system stress based on multiple sensor readings.
    stress_index = (
        (temperature - 72) / 12
        + (vibration - 1.5) / 0.75
        + (pressure - 110) / 18
        + (current - 28) / 10
        + (rpm - 1250) / 260
        + (coolant_temp - 72) / 16
    )

    failure_probability = 1 / (1 + np.exp(-(stress_index - 2.7)))
    failure = (rng.random(n_samples) < failure_probability).astype(int)

    # Force some obvious high-risk patterns to make learning realistic.
    failure = np.where(temperature > 95, 1, failure)
    failure = np.where((pressure > 145) & (current > 55), 1, failure)
    failure = np.where((vibration > 3.4) & (temperature > 90), 1, failure)
    failure = np.where((rpm > 1700) & (coolant_temp > 92), 1, failure)
    failure = np.where((temperature < 40) & (pressure < 80), 0, failure)

    dataset = pd.DataFrame(
        {
            "device_id": device_id,
            "timestamp": timestamp,
            "temperature": np.clip(temperature, 20, 140),
            "vibration": np.clip(vibration, 0.2, 6.0),
            "pressure": np.clip(pressure, 40, 170),
            "rpm": np.clip(rpm, 500, 2000),
            "current": np.clip(current, 5, 90),
            "humidity": np.clip(humidity, 10, 90),
            "coolant_temp": np.clip(coolant_temp, 20, 130),
            "power_draw": np.clip(power_draw, 10, 90),
            "device_age_hours": device_age_hours,
            "maintenance_gap_days": maintenance_gap_days,
            "failure": failure,
        }
    )

    dataset.to_csv(output_path, index=False)
    return dataset
