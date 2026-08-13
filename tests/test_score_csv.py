from pathlib import Path

import pandas as pd
import pytest

from app.schemas.lead import FeatureContribution, LeadFeatures, LeadScoreData
from ml.generate_data import generate_synthetic_leads
from scripts.score_csv import score_csv


class FakeBatchPredictor:
    def predict_many(self, leads: list[LeadFeatures]) -> list[LeadScoreData]:
        return [
            LeadScoreData(
                prediction="qualified",
                lead_score=75.0,
                conversion_probability=0.75,
                confidence=0.5,
                confidence_level="medium",
                threshold=0.4,
                top_positive_factors=[
                    FeatureContribution(
                        feature="demo_requests_30d",
                        log_odds_contribution=1.0,
                        direction="positive",
                    )
                ],
                top_negative_factors=[],
                explanation="test",
                model_version="test-2.0.0",
            )
            for _ in leads
        ]


def test_score_csv_writes_business_friendly_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "leads.csv"
    output_path = tmp_path / "scored_leads.csv"
    dataframe = generate_synthetic_leads(rows=100, seed=42).drop(columns=["converted"]).head(3)
    dataframe.to_csv(input_path, index=False)

    output = score_csv(input_path, output_path, service=FakeBatchPredictor())

    assert output_path.exists()
    assert len(output) == 3
    assert output["lead_score"].tolist() == [75.0, 75.0, 75.0]
    assert "top_positive_factors" in output.columns
    reloaded = pd.read_csv(output_path)
    assert reloaded["model_version"].iloc[0] == "test-2.0.0"


def test_score_csv_rejects_missing_feature_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.csv"
    pd.DataFrame([{"company_size": 10}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        score_csv(input_path, tmp_path / "out.csv", service=FakeBatchPredictor())
