from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def build_calibration_data(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )

    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "bins": [
            {
                "mean_predicted_probability": float(predicted),
                "fraction_positive": float(observed),
            }
            for predicted, observed in zip(mean_predicted, fraction_positive, strict=True)
        ],
    }


def analyze_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        _, false_positive, false_negative, _ = matrix.ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "business_cost": float(
                    false_positive * false_positive_cost + false_negative * false_negative_cost
                ),
            }
        )

    f1_best = max(rows, key=lambda item: (float(item["f1"]), float(item["threshold"])))
    cost_best = min(
        rows,
        key=lambda item: (float(item["business_cost"]), -float(item["threshold"])),
    )

    default_threshold = 0.5
    f1_threshold = float(f1_best["threshold"])
    business_threshold = float(cost_best["threshold"])
    strategy_thresholds = {
        "default": default_threshold,
        "f1": f1_threshold,
        "business_cost": business_threshold,
    }
    strategies = []
    for strategy, threshold in strategy_thresholds.items():
        row = next(item for item in rows if float(item["threshold"]) == threshold)
        strategies.append(
            {
                "strategy": strategy,
                "threshold": threshold,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "false_positive": int(row["false_positive"]),
                "false_negative": int(row["false_negative"]),
                "business_cost": float(row["business_cost"]),
            }
        )

    return {
        "rows": rows,
        "default_threshold": default_threshold,
        "f1_optimal_threshold": f1_threshold,
        "business_cost_threshold": business_threshold,
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "strategies": strategies,
    }


def select_threshold(threshold_analysis: dict[str, Any], strategy: str) -> float:
    if strategy == "default":
        return float(threshold_analysis["default_threshold"])
    if strategy == "f1":
        return float(threshold_analysis["f1_optimal_threshold"])
    if strategy == "business_cost":
        return float(threshold_analysis["business_cost_threshold"])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def save_confusion_matrix(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Not converted", "Converted"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Lead Conversion Confusion Matrix (threshold={threshold:.2f})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_calibration_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(mean_predicted, fraction_positive, marker="o", label="Lead model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Probability Calibration Curve")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_threshold_analysis(
    threshold_analysis: dict[str, Any],
    selected_threshold: float,
    output_path: Path,
) -> None:
    rows = threshold_analysis["rows"]
    thresholds = [float(row["threshold"]) for row in rows]
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]
    costs = np.array([float(row["business_cost"]) for row in rows], dtype=float)
    normalized_costs = costs / costs.max() if costs.max() else costs

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.plot(thresholds, f1_values, label="F1")
    plt.plot(thresholds, normalized_costs, label="Normalized business cost")
    plt.axvline(selected_threshold, linestyle="--", label=f"Selected {selected_threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric / normalized cost")
    plt.title("Decision Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
