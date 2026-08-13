import argparse
import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from app.schemas.lead import LeadFeatures, LeadScoreData
from app.services.model_service import ModelService
from ml.features import FEATURE_COLUMNS


class BatchPredictor(Protocol):
    def predict_many(self, leads: list[LeadFeatures]) -> list[LeadScoreData]: ...


def score_csv(
    input_path: Path,
    output_path: Path,
    service: BatchPredictor | None = None,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    missing_columns = set(FEATURE_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    leads = [
        LeadFeatures.model_validate(record)
        for record in dataframe[FEATURE_COLUMNS].to_dict(orient="records")
    ]
    predictor = service or ModelService()
    predictions = predictor.predict_many(leads)

    output = dataframe.copy()
    output["prediction"] = [result.prediction for result in predictions]
    output["lead_score"] = [result.lead_score for result in predictions]
    output["conversion_probability"] = [result.conversion_probability for result in predictions]
    output["confidence"] = [result.confidence for result in predictions]
    output["confidence_level"] = [result.confidence_level for result in predictions]
    output["threshold"] = [result.threshold for result in predictions]
    output["top_positive_factors"] = [
        json.dumps([factor.model_dump() for factor in result.top_positive_factors])
        for result in predictions
    ]
    output["top_negative_factors"] = [
        json.dumps([factor.model_dump() for factor in result.top_negative_factors])
        for result in predictions
    ]
    output["model_version"] = [result.model_version for result in predictions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a CSV of lead rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = score_csv(args.input, args.output)
    print(f"Scored {len(output)} leads")
    print(f"Saved scored CSV to {args.output}")


if __name__ == "__main__":
    main()
