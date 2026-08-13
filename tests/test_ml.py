import json
from pathlib import Path

import numpy as np
import pytest

from common.hashing import sha256_file
from ml.evaluation import analyze_thresholds, select_threshold
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from ml.generate_data import generate_synthetic_leads, save_dataset
from ml.train import train


def test_synthetic_data_is_reproducible() -> None:
    first = generate_synthetic_leads(rows=500, seed=42)
    second = generate_synthetic_leads(rows=500, seed=42)

    assert first.equals(second)
    assert list(first.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert 0.15 < first[TARGET_COLUMN].mean() < 0.40


def test_synthetic_data_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="rows must be at least 100"):
        generate_synthetic_leads(rows=50)


def test_threshold_analysis_separates_business_policy_from_probability() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.55, 0.35, 0.6, 0.9])

    analysis = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        false_positive_cost=1.0,
        false_negative_cost=5.0,
    )

    assert 0 < analysis["f1_optimal_threshold"] < 1
    assert 0 < analysis["business_cost_threshold"] < 1
    assert select_threshold(analysis, "default") == 0.5
    assert select_threshold(analysis, "business_cost") == analysis["business_cost_threshold"]


def test_training_creates_expected_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "leads.csv"
    artifact_dir = tmp_path / "artifacts"
    dataframe = generate_synthetic_leads(rows=1200, seed=42)
    save_dataset(dataframe, data_path)

    result = train(data_path=data_path, artifact_dir=artifact_dir)

    expected_files = {
        "model.joblib",
        "metrics.json",
        "model_manifest.json",
        "confusion_matrix.png",
        "calibration_curve.png",
        "threshold_analysis.png",
    }
    assert expected_files <= {path.name for path in artifact_dir.iterdir()}
    assert result["metrics"]["selection_metric"] == "brier_score"
    assert result["metrics"]["selection_split"] == "validation"
    assert result["metrics"]["selected_model"] in {
        "logistic_default",
        "logistic_balanced",
    }

    saved_metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
    assert len(saved_metrics["candidates"]) == 2
    assert saved_metrics["test_rows"] == 240
    assert saved_metrics["validation_rows"] == 192
    assert saved_metrics["threshold_analysis"]["selected_strategy"] == "business_cost"
    assert len(saved_metrics["calibration"]["bins"]) > 0
    assert saved_manifest["model_sha256"] == sha256_file(artifact_dir / "model.joblib")
    assert saved_manifest["dataset_sha256"] == sha256_file(data_path)
    assert saved_manifest["feature_contract"] == FEATURE_COLUMNS
    assert saved_manifest["feature_count"] == len(FEATURE_COLUMNS)
    assert {"python", "scikit_learn", "pandas", "numpy", "joblib"} <= set(
        saved_manifest["library_versions"]
    )


def test_training_rejects_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        train(
            data_path=tmp_path / "missing.csv",
            artifact_dir=tmp_path / "artifacts",
        )


def test_training_rejects_negative_business_cost(tmp_path: Path) -> None:
    data_path = tmp_path / "leads.csv"
    dataframe = generate_synthetic_leads(rows=500, seed=42)
    save_dataset(dataframe, data_path)

    with pytest.raises(ValueError, match="non-negative"):
        train(
            data_path=data_path,
            artifact_dir=tmp_path / "artifacts",
            false_negative_cost=-1,
        )
