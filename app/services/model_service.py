import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import status

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.lead import FeatureContribution, LeadFeatures, LeadScoreData
from app.schemas.model import MetricsData, ModelInfoData, ModelManifestData
from common.hashing import sha256_file

logger = get_logger(__name__)


class ModelService:
    """Validate and load trained artifacts, then serve vectorized inference."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._require_artifacts()

        try:
            self.manifest = self._read_json(self.settings.manifest_path)
            self._verify_runtime_versions()
            self._verify_model_hash()
            bundle = joblib.load(self.settings.model_path)
            self.pipeline = bundle["pipeline"]
            self.metadata = bundle["metadata"]
            self.metrics = self._read_json(self.settings.metrics_path)
            self._verify_bundle_contract()
            self.feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
            self.classifier = self.pipeline.named_steps["classifier"]
        except AppError:
            raise
        except Exception as exc:
            logger.exception("Failed to load model artifacts")
            raise AppError(
                message="Model artifacts could not be loaded.",
                code="MODEL_ARTIFACT_LOAD_FAILED",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def predict(self, lead: LeadFeatures) -> LeadScoreData:
        return self.predict_many([lead])[0]

    def predict_many(self, leads: list[LeadFeatures]) -> list[LeadScoreData]:
        if not leads:
            raise AppError(
                message="At least one lead is required.",
                code="EMPTY_LEAD_BATCH",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if len(leads) > self.settings.max_batch_size:
            raise AppError(
                message=f"Batch size cannot exceed {self.settings.max_batch_size} leads.",
                code="LEAD_BATCH_TOO_LARGE",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )

        frame = pd.DataFrame(
            [lead.model_dump() for lead in leads],
            columns=self.manifest["feature_contract"],
        )

        try:
            probabilities = self.pipeline.predict_proba(frame)[:, 1]
            transformed = self.pipeline.named_steps["preprocessor"].transform(frame)
            transformed_array = (
                transformed.toarray()
                if hasattr(transformed, "toarray")
                else np.asarray(transformed)
            )
            contributions = transformed_array * self.classifier.coef_[0]
        except Exception as exc:
            logger.exception("Lead score prediction failed")
            raise AppError(
                message="Lead score prediction failed.",
                code="LEAD_PREDICTION_FAILED",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        return [
            self._build_prediction(lead, float(probability), contribution_row)
            for lead, probability, contribution_row in zip(
                leads,
                probabilities,
                contributions,
                strict=True,
            )
        ]

    def get_model_info(self) -> ModelInfoData:
        return ModelInfoData.model_validate(self.metadata)

    def get_metrics(self) -> MetricsData:
        return MetricsData.model_validate(self.metrics)

    def get_manifest(self) -> ModelManifestData:
        return ModelManifestData.model_validate(self.manifest)

    def _require_artifacts(self) -> None:
        required = {
            "MODEL_ARTIFACT_MISSING": self.settings.model_path,
            "METRICS_ARTIFACT_MISSING": self.settings.metrics_path,
            "MANIFEST_ARTIFACT_MISSING": self.settings.manifest_path,
        }

        for code, path in required.items():
            if not path.exists():
                raise AppError(
                    message=f"Required artifact is missing: {path}",
                    code=code,
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

    @staticmethod
    def _read_json(path: Any) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _verify_runtime_versions(self) -> None:
        expected = self.manifest.get("library_versions", {})
        actual = {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        }
        mismatches = [
            f"{name}: expected {expected.get(name)}, found {version}"
            for name, version in actual.items()
            if expected.get(name) != version
        ]

        if mismatches:
            raise AppError(
                message=(
                    "Runtime ML library versions do not match the trusted model manifest: "
                    + "; ".join(mismatches)
                ),
                code="MODEL_RUNTIME_VERSION_MISMATCH",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _verify_model_hash(self) -> None:
        expected_hash = str(self.manifest.get("model_sha256", ""))
        actual_hash = sha256_file(self.settings.model_path)

        if not expected_hash or actual_hash != expected_hash:
            raise AppError(
                message="Model artifact hash does not match the trusted manifest.",
                code="MODEL_ARTIFACT_HASH_MISMATCH",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _verify_bundle_contract(self) -> None:
        if self.metadata.get("model_version") != self.manifest.get("model_version"):
            raise AppError(
                message="Model bundle version does not match the artifact manifest.",
                code="MODEL_VERSION_MISMATCH",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if self.metadata.get("threshold") != self.manifest.get("threshold"):
            raise AppError(
                message="Model threshold does not match the artifact manifest.",
                code="MODEL_THRESHOLD_MISMATCH",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _build_prediction(
        self,
        lead: LeadFeatures,
        probability: float,
        contribution_row: np.ndarray,
    ) -> LeadScoreData:
        threshold = float(self.metadata["threshold"])
        prediction = "qualified" if probability >= threshold else "not_qualified"
        confidence = self._decision_confidence(probability, threshold)
        positive_factors, negative_factors = self._top_contributions(contribution_row)

        return LeadScoreData(
            prediction=prediction,
            lead_score=round(probability * 100, 2),
            conversion_probability=round(probability, 4),
            confidence=round(confidence, 4),
            confidence_level=self._confidence_level(confidence),
            threshold=threshold,
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors,
            explanation=self._build_explanation(
                probability,
                positive_factors,
                negative_factors,
            ),
            model_version=str(self.metadata["model_version"]),
        )

    def _top_contributions(
        self,
        contribution_row: np.ndarray,
        limit: int = 3,
    ) -> tuple[list[FeatureContribution], list[FeatureContribution]]:
        pairs = [
            (self._humanize_feature_name(str(name)), float(value))
            for name, value in zip(self.feature_names, contribution_row, strict=True)
            if abs(float(value)) > 1e-9
        ]
        positive = sorted(
            (pair for pair in pairs if pair[1] > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )[:limit]
        negative = sorted(
            (pair for pair in pairs if pair[1] < 0),
            key=lambda pair: pair[1],
        )[:limit]

        return (
            [
                FeatureContribution(
                    feature=feature,
                    log_odds_contribution=round(value, 4),
                    direction="positive",
                )
                for feature, value in positive
            ],
            [
                FeatureContribution(
                    feature=feature,
                    log_odds_contribution=round(value, 4),
                    direction="negative",
                )
                for feature, value in negative
            ],
        )

    def _humanize_feature_name(self, transformed_name: str) -> str:
        _, _, feature_name = transformed_name.partition("__")

        for original_feature in sorted(
            self.metadata["categorical_features"],
            key=len,
            reverse=True,
        ):
            prefix = f"{original_feature}_"
            if feature_name.startswith(prefix):
                category = feature_name[len(prefix) :]
                return f"{original_feature}={category}"

        return feature_name

    @staticmethod
    def _decision_confidence(probability: float, threshold: float) -> float:
        max_distance = max(threshold, 1 - threshold)
        if max_distance == 0:
            return 0.0
        return min(abs(probability - threshold) / max_distance, 1.0)

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        if confidence >= 0.70:
            return "high"
        if confidence >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _build_explanation(
        probability: float,
        positive_factors: list[FeatureContribution],
        negative_factors: list[FeatureContribution],
    ) -> str:
        likelihood = "high" if probability >= 0.70 else "moderate" if probability >= 0.40 else "low"

        def summarize(factors: list[FeatureContribution]) -> str:
            if not factors:
                return "none among the strongest transformed features"
            return ", ".join(
                f"{factor.feature} ({factor.log_odds_contribution:+.3f} log-odds)"
                for factor in factors
            )

        return (
            f"The model estimates a {likelihood} conversion probability. "
            f"Largest positive log-odds contributions: {summarize(positive_factors)}. "
            f"Largest negative log-odds contributions: {summarize(negative_factors)}. "
            "These are local linear-model contributions after preprocessing, not causal effects."
        )


@lru_cache
def get_model_service() -> ModelService:
    """Return one cached model service so artifacts load once per process."""

    return ModelService()
