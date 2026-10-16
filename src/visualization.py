from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix, labels, save_path="images/confusion_matrix.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path="images/feature_importance.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.named_steps["model"].feature_importances_
    sorted_idx = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar([feature_names[i] for i in sorted_idx[:10]], importances[sorted_idx[:10]], color="steelblue")
    plt.title("Top 10 Features by Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_failure_probability(model, X_test, y_test, save_path="images/failure_probability.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    risk_scores = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 5))
    plt.hist(risk_scores[y_test == 0], bins=20, alpha=0.7, label="Healthy", color="green")
    plt.hist(risk_scores[y_test == 1], bins=20, alpha=0.7, label="Failure", color="red")
    plt.title("Failure Probability Distribution")
    plt.xlabel("Predicted Probability of Failure")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
