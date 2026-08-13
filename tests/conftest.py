from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import AppError
from app.main import app
from app.schemas.lead import FeatureContribution, LeadFeatures, LeadScoreData
from app.schemas.model import MetricsData, ModelInfoData, ModelManifestData
from app.services.model_service import get_model_service


class FakeModelService:
    def predict(self, lead: LeadFeatures) -> LeadScoreData:
        return self.predict_many([lead])[0]

    def predict_many(self, leads: list[LeadFeatures]) -> list[LeadScoreData]:
        if len(leads) > 1000:
            raise AppError(
                message="Batch size cannot exceed 1000 leads.",
                code="LEAD_BATCH_TOO_LARGE",
                status_code=413,
            )

        return [
            LeadScoreData(
                prediction="qualified",
                lead_score=82.5,
                conversion_probability=0.825,
                confidence=0.65,
                confidence_level="medium",
                threshold=0.38,
                top_positive_factors=[
                    FeatureContribution(
                        feature="demo_requests_30d",
                        log_odds_contribution=1.2,
                        direction="positive",
                    )
                ],
                top_negative_factors=[
                    FeatureContribution(
                        feature="days_since_last_contact",
                        log_odds_contribution=-0.4,
                        direction="negative",
                    )
                ],
                explanation=f"Fake explanation for {lead.industry} lead.",
                model_version="test-2.0.0",
            )
            for lead in leads
        ]

    def get_model_info(self) -> ModelInfoData:
        return ModelInfoData(
            model_version="test-2.0.0",
            schema_version="1.0",
            algorithm="LogisticRegression",
            selected_model="logistic_balanced",
            trained_at_utc="2026-01-01T00:00:00+00:00",
            dataset_rows=5000,
            training_rows=3200,
            validation_rows=800,
            test_rows=1000,
            target="converted",
            threshold=0.38,
            threshold_strategy="business_cost",
            numeric_features=["company_size"],
            categorical_features=["lead_source"],
            top_positive_features=[{"feature": "demo", "coefficient": 1.0}],
            top_negative_features=[{"feature": "stale", "coefficient": -1.0}],
            library_versions={
                "python": "test",
                "scikit_learn": "test",
                "pandas": "test",
                "numpy": "test",
                "joblib": "test",
            },
        )

    def get_metrics(self) -> MetricsData:
        return MetricsData.model_validate(
            {
                "selected_model": "logistic_balanced",
                "selection_metric": "brier_score",
                "selection_split": "validation",
                "dataset_rows": 5000,
                "training_rows": 3200,
                "validation_rows": 800,
                "test_rows": 1000,
                "positive_rate": 0.27,
                "threshold": 0.38,
                "metrics": {
                    "accuracy": 0.68,
                    "precision": 0.43,
                    "recall": 0.72,
                    "f1": 0.54,
                    "roc_auc": 0.73,
                    "brier_score": 0.22,
                    "log_loss": 0.64,
                },
                "confusion_matrix": {
                    "true_negative": 470,
                    "false_positive": 260,
                    "false_negative": 75,
                    "true_positive": 195,
                },
                "calibration": {
                    "brier_score": 0.22,
                    "log_loss": 0.64,
                    "bins": [
                        {
                            "mean_predicted_probability": 0.2,
                            "fraction_positive": 0.15,
                        }
                    ],
                },
                "threshold_analysis": {
                    "default_threshold": 0.5,
                    "f1_optimal_threshold": 0.42,
                    "business_cost_threshold": 0.38,
                    "selected_threshold": 0.38,
                    "selected_strategy": "business_cost",
                    "false_positive_cost": 1.0,
                    "false_negative_cost": 5.0,
                    "strategies": [
                        {
                            "strategy": "default",
                            "threshold": 0.5,
                            "precision": 0.58,
                            "recall": 0.25,
                            "f1": 0.35,
                            "false_positive": 50,
                            "false_negative": 199,
                            "business_cost": 1045.0,
                        },
                        {
                            "strategy": "f1",
                            "threshold": 0.26,
                            "precision": 0.43,
                            "recall": 0.65,
                            "f1": 0.52,
                            "false_positive": 227,
                            "false_negative": 94,
                            "business_cost": 697.0,
                        },
                        {
                            "strategy": "business_cost",
                            "threshold": 0.38,
                            "precision": 0.43,
                            "recall": 0.72,
                            "f1": 0.54,
                            "false_positive": 260,
                            "false_negative": 75,
                            "business_cost": 635.0,
                        },
                    ],
                },
                "candidates": [
                    {
                        "model_name": "logistic_balanced",
                        "accuracy": 0.70,
                        "precision": 0.45,
                        "recall": 0.64,
                        "f1": 0.53,
                        "roc_auc": 0.73,
                        "brier_score": 0.16,
                        "log_loss": 0.48,
                    }
                ],
            }
        )

    def get_manifest(self) -> ModelManifestData:
        return ModelManifestData(
            schema_version="1.0",
            model_version="test-2.0.0",
            algorithm="LogisticRegression",
            selected_model="logistic_balanced",
            trained_at_utc="2026-01-01T00:00:00+00:00",
            model_sha256="a" * 64,
            dataset_sha256="b" * 64,
            feature_contract=["company_size", "lead_source"],
            feature_count=2,
            target="converted",
            threshold=0.38,
            threshold_strategy="business_cost",
            library_versions={
                "python": "test",
                "scikit_learn": "test",
                "pandas": "test",
                "numpy": "test",
                "joblib": "test",
            },
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def override_model_service() -> Iterator[None]:
    app.dependency_overrides[get_model_service] = lambda: FakeModelService()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        yield async_client
