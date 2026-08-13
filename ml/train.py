import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.hashing import sha256_file
from ml.evaluation import (
    analyze_thresholds,
    build_calibration_data,
    calculate_metrics,
    save_calibration_curve,
    save_confusion_matrix,
    save_threshold_analysis,
    select_threshold,
)
from ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

RANDOM_SEED = 42
MODEL_VERSION = "2.0.0"
SCHEMA_VERSION = "1.0"
SELECTION_METRIC = "brier_score"
DEFAULT_FALSE_POSITIVE_COST = 1.0
DEFAULT_FALSE_NEGATIVE_COST = 5.0
ThresholdStrategy = Literal["default", "f1", "business_cost"]


def build_pipeline(class_weight: str | None) -> Pipeline:
    """Build one preprocessing + logistic-regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def extract_feature_weights(
    pipeline: Pipeline,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    feature_weights = sorted(
        zip(feature_names, coefficients, strict=True),
        key=lambda item: item[1],
    )

    negative = [
        {"feature": str(name), "coefficient": float(weight)} for name, weight in feature_weights[:5]
    ]
    positive = [
        {"feature": str(name), "coefficient": float(weight)}
        for name, weight in reversed(feature_weights[-5:])
    ]

    return positive, negative


def _validate_dataset(dataframe: pd.DataFrame) -> None:
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    invalid_targets = set(dataframe[TARGET_COLUMN].dropna().unique()) - {0, 1}
    if invalid_targets:
        raise ValueError("Target column must contain only binary values 0 and 1")


def train(
    data_path: Path,
    artifact_dir: Path,
    threshold_strategy: ThresholdStrategy = "business_cost",
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python -m ml.generate_data` first."
        )

    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Business error costs must be non-negative")

    dataframe = pd.read_csv(data_path)
    _validate_dataset(dataframe)

    features = dataframe[FEATURE_COLUMNS]
    target = dataframe[TARGET_COLUMN]

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_validation,
    )

    candidate_specs = {
        "logistic_default": None,
        "logistic_balanced": "balanced",
    }
    candidate_results: list[dict[str, Any]] = []
    validation_models: dict[str, Pipeline] = {}

    for model_name, class_weight in candidate_specs.items():
        pipeline = build_pipeline(class_weight=class_weight)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        metrics = calculate_metrics(y_validation, probabilities, threshold=0.5)
        validation_models[model_name] = pipeline
        candidate_results.append(
            {
                "model_name": model_name,
                **{
                    key: metrics[key]
                    for key in [
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "brier_score",
                        "log_loss",
                    ]
                },
            }
        )

    selected_candidate = min(candidate_results, key=lambda item: item[SELECTION_METRIC])
    selected_model_name = str(selected_candidate["model_name"])
    validation_pipeline = validation_models[selected_model_name]
    validation_probabilities = validation_pipeline.predict_proba(x_validation)[:, 1]

    threshold_analysis = analyze_thresholds(
        y_true=y_validation,
        probabilities=validation_probabilities,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    selected_threshold = select_threshold(threshold_analysis, threshold_strategy)

    selected_pipeline = build_pipeline(class_weight=candidate_specs[selected_model_name])
    selected_pipeline.fit(x_train_validation, y_train_validation)
    test_probabilities = selected_pipeline.predict_proba(x_test)[:, 1]
    selected_metrics = calculate_metrics(y_test, test_probabilities, selected_threshold)
    calibration_data = build_calibration_data(y_test, test_probabilities)
    top_positive, top_negative = extract_feature_weights(selected_pipeline)

    trained_at = datetime.now(UTC).isoformat()
    metadata = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "algorithm": "LogisticRegression",
        "selected_model": selected_model_name,
        "trained_at_utc": trained_at,
        "dataset_rows": int(len(dataframe)),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "target": TARGET_COLUMN,
        "threshold": selected_threshold,
        "threshold_strategy": threshold_strategy,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    metrics_payload = {
        "selected_model": selected_model_name,
        "selection_metric": SELECTION_METRIC,
        "selection_split": "validation",
        "dataset_rows": int(len(dataframe)),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "positive_rate": float(target.mean()),
        "threshold": selected_threshold,
        "metrics": {
            key: selected_metrics[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "log_loss",
            ]
        },
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "calibration": calibration_data,
        "threshold_analysis": {
            "default_threshold": threshold_analysis["default_threshold"],
            "f1_optimal_threshold": threshold_analysis["f1_optimal_threshold"],
            "business_cost_threshold": threshold_analysis["business_cost_threshold"],
            "selected_threshold": selected_threshold,
            "selected_strategy": threshold_strategy,
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
            "strategies": threshold_analysis["strategies"],
        },
        "candidates": candidate_results,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump({"pipeline": selected_pipeline, "metadata": metadata}, model_path)

    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "algorithm": "LogisticRegression",
        "selected_model": selected_model_name,
        "trained_at_utc": trained_at,
        "model_sha256": sha256_file(model_path),
        "dataset_sha256": sha256_file(data_path),
        "feature_contract": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "threshold": selected_threshold,
        "threshold_strategy": threshold_strategy,
        "library_versions": metadata["library_versions"],
    }

    with (artifact_dir / "model_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    save_confusion_matrix(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=selected_threshold,
        output_path=artifact_dir / "confusion_matrix.png",
    )
    save_calibration_curve(
        y_true=y_test,
        probabilities=test_probabilities,
        output_path=artifact_dir / "calibration_curve.png",
    )
    save_threshold_analysis(
        threshold_analysis=threshold_analysis,
        selected_threshold=selected_threshold,
        output_path=artifact_dir / "threshold_analysis.png",
    )

    return {
        "metadata": metadata,
        "metrics": metrics_payload,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lead-scoring model.")
    parser.add_argument("--data", type=Path, default=Path("data/leads.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--threshold-strategy",
        choices=["default", "f1", "business_cost"],
        default="business_cost",
    )
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=DEFAULT_FALSE_POSITIVE_COST,
    )
    parser.add_argument(
        "--false-negative-cost",
        type=float,
        default=DEFAULT_FALSE_NEGATIVE_COST,
    )
    args = parser.parse_args()

    if args.false_positive_cost < 0 or args.false_negative_cost < 0:
        parser.error("business error costs must be non-negative")

    return args


def main() -> None:
    args = parse_args()
    result = train(
        data_path=args.data,
        artifact_dir=args.artifacts,
        threshold_strategy=args.threshold_strategy,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )

    metrics = result["metrics"]["metrics"]
    metadata = result["metadata"]

    print(f"Selected model: {metadata['selected_model']}")
    print(f"Threshold strategy: {metadata['threshold_strategy']}")
    print(f"Selected threshold: {metadata['threshold']:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"Brier:     {metrics['brier_score']:.3f}")
    print(f"Log loss:  {metrics['log_loss']:.3f}")
    print(f"Saved artifacts to {args.artifacts}")


if __name__ == "__main__":
    main()
