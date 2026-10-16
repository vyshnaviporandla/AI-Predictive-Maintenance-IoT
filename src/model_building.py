import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train):
    """Train a beginner-friendly classification model for failure prediction."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test, model_dir="models"):
    """Evaluate the trained model and save metrics."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "average_failure_probability": round(float(y_prob.mean()), 4),
    }

    joblib.dump(model, model_dir / "predictive_maintenance_model.pkl")
    (model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
