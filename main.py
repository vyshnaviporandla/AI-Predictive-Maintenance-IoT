from pathlib import Path

from src.data_generator import generate_iot_dataset
from src.model_building import evaluate_model, train_model
from src.preprocessing import load_and_prepare_data
from src.prediction import build_alert, predict_failure, save_prediction_report
from src.visualization import plot_confusion_matrix, plot_feature_importance, plot_failure_probability


def main():
    """Run the full predictive maintenance workflow for an IoT dataset."""
    print("Step 1: Generating synthetic sensor data...")
    data_path = Path("data/industrial_iot_failure_data.csv")
    generate_iot_dataset(output_path=data_path, n_samples=3000, seed=42)
    print(f"Dataset saved to {data_path}")

    print("Step 2: Cleaning and preparing the data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(data_path)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    print("Step 3: Training the predictive maintenance model...")
    model = train_model(X_train, y_train)

    print("Step 4: Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test, model_dir="models")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")

    print("Step 5: Saving charts and visuals...")
    plot_confusion_matrix(metrics["confusion_matrix"], [0, 1], save_path="images/confusion_matrix.png")
    plot_feature_importance(model, feature_names, save_path="images/feature_importance.png")
    plot_failure_probability(model, X_test, y_test, save_path="images/failure_probability.png")

    print("Step 6: Running a sample prediction and alert check...")
    sample_device = {
        "temperature": 98,
        "vibration": 3.8,
        "pressure": 155,
        "rpm": 1765,
        "current": 62,
        "humidity": 54,
        "coolant_temp": 96,
        "power_draw": 73,
        "device_age_hours": 4800,
        "maintenance_gap_days": 42,
    }

    prediction, probability = predict_failure(model, sample_device)
    alert = build_alert(prediction, probability)
    save_prediction_report(sample_device, prediction, probability)
    print(f"Prediction: {prediction} | Failure probability: {probability:.2f}")
    print(alert)

    summary_path = Path("outputs/maintenance_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "AI-Powered Predictive Maintenance for IoT Devices",
                "==============================================",
                f"Accuracy: {metrics['accuracy']}",
                f"Precision: {metrics['precision']}",
                f"Recall: {metrics['recall']}",
                f"F1 Score: {metrics['f1_score']}",
                f"Sample alert: {alert}",
            ]
        ),
        encoding="utf-8",
    )

    print("Project completed successfully. Results are saved in the outputs and images folders.")


if __name__ == "__main__":
    main()
