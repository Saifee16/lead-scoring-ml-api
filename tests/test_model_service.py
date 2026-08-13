import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.services.model_service as model_service_module
from app.core.exceptions import AppError
from app.schemas.lead import LeadFeatures
from app.services.model_service import ModelService
from ml.generate_data import generate_synthetic_leads, save_dataset
from ml.train import train


def build_trained_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_path = tmp_path / "leads.csv"
    artifact_dir = tmp_path / "artifacts"
    dataframe = generate_synthetic_leads(rows=1200, seed=42)
    save_dataset(dataframe, data_path)
    train(data_path=data_path, artifact_dir=artifact_dir)
    return (
        artifact_dir / "model.joblib",
        artifact_dir / "metrics.json",
        artifact_dir / "model_manifest.json",
    )


def build_lead() -> LeadFeatures:
    return LeadFeatures(
        lead_source="referral",
        industry="saas",
        company_size=120,
        annual_revenue_k=2500,
        website_visits_30d=28,
        email_opens_30d=8,
        demo_requests_30d=2,
        form_submissions_30d=3,
        days_since_last_contact=4,
        decision_maker_engaged=True,
    )


def settings_for(model_path: Path, metrics_path: Path, manifest_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=model_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        max_batch_size=1000,
    )


def test_real_model_service_predicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, metrics_path, manifest_path = build_trained_artifacts(tmp_path)
    monkeypatch.setattr(
        model_service_module,
        "get_settings",
        lambda: settings_for(model_path, metrics_path, manifest_path),
    )

    service = ModelService()
    prediction = service.predict(build_lead())

    assert 0 <= prediction.conversion_probability <= 1
    assert prediction.lead_score == round(prediction.conversion_probability * 100, 2)
    assert prediction.prediction in {"qualified", "not_qualified"}
    assert prediction.model_version == "2.0.0"
    assert prediction.top_positive_factors
    assert all(factor.direction == "positive" for factor in prediction.top_positive_factors)
    assert all(factor.direction == "negative" for factor in prediction.top_negative_factors)
    assert "local linear-model contributions" in prediction.explanation
    assert service.get_model_info().algorithm == "LogisticRegression"
    assert service.get_metrics().selection_split == "validation"
    assert len(service.get_manifest().model_sha256) == 64


def test_real_model_service_scores_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, metrics_path, manifest_path = build_trained_artifacts(tmp_path)
    monkeypatch.setattr(
        model_service_module,
        "get_settings",
        lambda: settings_for(model_path, metrics_path, manifest_path),
    )

    service = ModelService()
    predictions = service.predict_many([build_lead(), build_lead()])

    assert len(predictions) == 2
    assert predictions[0].conversion_probability == predictions[1].conversion_probability


def test_model_service_enforces_configured_batch_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, metrics_path, manifest_path = build_trained_artifacts(tmp_path)
    settings = settings_for(model_path, metrics_path, manifest_path)
    settings.max_batch_size = 1
    monkeypatch.setattr(model_service_module, "get_settings", lambda: settings)

    service = ModelService()

    with pytest.raises(AppError) as error:
        service.predict_many([build_lead(), build_lead()])

    assert error.value.code == "LEAD_BATCH_TOO_LARGE"
    assert error.value.status_code == 413


def test_model_service_rejects_runtime_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, metrics_path, manifest_path = build_trained_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["library_versions"]["scikit_learn"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        model_service_module,
        "get_settings",
        lambda: settings_for(model_path, metrics_path, manifest_path),
    )

    with pytest.raises(AppError) as error:
        ModelService()

    assert error.value.code == "MODEL_RUNTIME_VERSION_MISMATCH"


def test_model_service_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, metrics_path, manifest_path = build_trained_artifacts(tmp_path)
    with model_path.open("ab") as file:
        file.write(b"tampered")

    monkeypatch.setattr(
        model_service_module,
        "get_settings",
        lambda: settings_for(model_path, metrics_path, manifest_path),
    )

    with pytest.raises(AppError) as error:
        ModelService()

    assert error.value.code == "MODEL_ARTIFACT_HASH_MISMATCH"


def test_model_service_reports_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_settings = settings_for(
        tmp_path / "missing.joblib",
        tmp_path / "missing.json",
        tmp_path / "missing-manifest.json",
    )
    monkeypatch.setattr(model_service_module, "get_settings", lambda: fake_settings)

    with pytest.raises(AppError) as error:
        ModelService()

    assert error.value.code == "MODEL_ARTIFACT_MISSING"
    assert error.value.status_code == 503


def test_confidence_uses_selected_decision_threshold() -> None:
    assert ModelService._decision_confidence(0.4, 0.4) == 0
    assert ModelService._decision_confidence(1.0, 0.4) == 1
    assert 0 < ModelService._decision_confidence(0.7, 0.4) < 1
    assert ModelService._confidence_level(0.80) == "high"
    assert ModelService._confidence_level(0.50) == "medium"
    assert ModelService._confidence_level(0.10) == "low"
