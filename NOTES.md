# NOTES.md — Project 2 Deep Learning Manual


This document is the teaching companion for **Lead Scoring ML API**.

It assumes you completed Project 1, **AI Engineering Starter Kit**. I will not re-teach basic FastAPI concepts such as `APIRouter`, `BaseSettings`, generic `SuccessResponse`, global exception handlers, async HTTPX tests, Docker basics, or GitHub Actions basics unless Project 2 changes how they are used.

Instead, this file focuses on **everything new in Project 2**:

- supervised machine learning lifecycle
- binary classification
- synthetic data generation
- feature contracts
- train / validation / test separation
- scikit-learn preprocessing pipelines
- model selection for probability quality
- Brier score and log loss
- probability calibration diagnostics
- threshold tuning
- business-cost policies
- confusion matrices
- model persistence
- model manifests and SHA-256
- vectorized inference
- local logistic-regression contributions
- batch API scoring
- CSV scoring
- release assets
- GHCR image publishing
- CodeQL

The goal is not syntax memorization. The goal is to understand **why the system is shaped this way and where to debug it**.

# 1. Project 1 vs Project 2


## What we reuse from the AI Engineering Starter Kit

Project 1 already taught this backbone:

```text
request
  ↓
Pydantic schema
  ↓
FastAPI endpoint
  ↓
service layer
  ↓
external capability
  ↓
typed response
```

Project 2 reuses:

| Project 1 concept | Project 2 use |
|---|---|
| `app/main.py` | creates the FastAPI application |
| `api/v1/router.py` | combines versioned route groups |
| `core/config.py` | loads environment configuration |
| `core/logging.py` | centralized logging |
| `core/exceptions.py` | controlled API errors |
| `schemas/common.py` | generic typed success wrapper |
| service layer | now wraps an ML model instead of Gemini |
| dependency injection | supplies `ModelService` to endpoints |
| async HTTPX tests | tests the API without starting Uvicorn |
| Docker | packages runtime behavior |
| CI | runs quality and test gates |

## The important mental shift

Project 1 external capability:

```text
Gemini API
```

Project 2 external capability:

```text
trained local ML artifact
```

The route should still **not know how the capability works**.

The route says:

```python
prediction = service.predict(lead)
```

It does not know:

- how categories are encoded
- how numeric values are scaled
- which model was selected
- which threshold is used
- how probabilities are calculated
- how contributions are computed
- where the model file lives
- how the model hash is checked

That separation is exactly why the service layer exists.

# 2. Full file structure


```text
lead-scoring-ml-api/
|
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── config.yml
│   │   └── feature_request.yml
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── codeql.yml
│   │   ├── publish-ghcr.yml
│   │   └── release.yml
│   ├── dependabot.yml
│   └── PULL_REQUEST_TEMPLATE.md
|
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── endpoints/
│   │       │   ├── __init__.py
│   │       │   ├── health.py
│   │       │   ├── model.py
│   │       │   └── prediction.py
│   │       └── router.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── health.py
│   │   ├── lead.py
│   │   └── model.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── model_service.py
│   └── main.py
|
├── common/
│   ├── __init__.py
│   └── hashing.py
|
├── ml/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── features.py
│   ├── generate_data.py
│   └── train.py
|
├── scripts/
│   ├── __init__.py
│   └── score_csv.py
|
├── data/
│   ├── README.md
│   └── leads.csv
|
├── artifacts/
│   ├── README.md
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── model_manifest.json
│   ├── threshold_analysis.png
│   └── model.joblib          # generated locally; ignored by Git
|
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_ml.py
│   ├── test_model.py
│   ├── test_model_service.py
│   ├── test_prediction.py
│   └── test_score_csv.py
|
├── .dockerignore
├── .editorconfig
├── .env.example
├── .gitignore
├── .python-version
├── compose.yaml
├── CONTRIBUTING.md
├── DATA_CARD.md
├── Dockerfile
├── LICENSE
├── MODEL_CARD.md
├── NOTES.md
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── requirements-ml.txt
├── requirements.txt
└── SECURITY.md
```

## Why `__init__.py` is present

Files such as:

```text
app/__init__.py
ml/__init__.py
scripts/__init__.py
common/__init__.py
```

are intentionally empty.

They explicitly mark normal Python packages.

That lets us write imports such as:

```python
from ml.train import train
from app.services.model_service import ModelService
from common.hashing import sha256_file
```

The file is named `__init__.py`: two underscores before and after `init`.

# 3. The whole system in one diagram


## Offline path: training

```text
ml.generate_data
      ↓
data/leads.csv
      ↓
ml.train
      ↓
feature contract validation
      ↓
train / validation / test
      ↓
fit candidate pipelines on TRAIN
      ↓
compare candidate probabilities on VALIDATION
      ↓
select lower Brier score
      ↓
tune threshold on VALIDATION
      ↓
refit selected pipeline on TRAIN + VALIDATION
      ↓
evaluate once on TEST
      ↓
generate model + metrics + plots + manifest
```

## Online path: inference

```text
POST /predict-lead-score
      ↓
LeadFeatures validation
      ↓
ModelService
      ↓
manifest exists?
      ↓
SHA-256 matches model?
      ↓
trusted joblib load
      ↓
DataFrame in manifest feature order
      ↓
pipeline.predict_proba
      ↓
probability
      ↓
selected threshold
      ↓
qualified / not_qualified
      ↓
transformed values × coefficients
      ↓
top positive / negative log-odds contributions
      ↓
LeadScoreData
```

# 4. ML concepts before code


## 4.1 Supervised learning

We have examples where the answer is already known.

One row:

```text
lead features + converted label
```

Example:

```text
referral
saas
2 demo requests
28 website visits
converted = 1
```

The model studies many rows and learns a function:

```text
features → probability of converted = 1
```

This is **supervised learning** because the target label is present during training.

## 4.2 Binary classification

The target has two classes:

```text
0 = not converted
1 = converted
```

That is binary classification.

The model produces a probability:

```text
P(converted = 1)
```

Example:

```text
0.73
```

The probability itself is not yet the business action.

We need a threshold:

```text
probability >= threshold → qualified
probability < threshold  → not_qualified
```

## 4.3 Why probability and threshold are separate

Suppose:

```text
probability = 0.35
```

At threshold `0.50`:

```text
not qualified
```

At threshold `0.24`:

```text
qualified
```

The model did not change.

The probability did not change.

Only the **decision policy** changed.

This is one of the most important lessons in Project 2.

## 4.4 Train, validation, test

We use three groups.

### Training set

Used to fit coefficients.

```text
model learns here
```

### Validation set

Used to make engineering/model decisions.

We use it to:

```text
select candidate model
select threshold
```

### Test set

Used only after decisions are made.

```text
final honest evaluation
```

If we repeatedly choose models based on the test score, the test set becomes part of the decision process and is no longer truly untouched.

Actual split:

```text
5000 total
3200 training
800 validation
1000 test
```

## 4.5 Logistic regression mental model

Despite the name "regression", logistic regression is commonly used for classification.

It first computes a linear score:

```text
z = b + w1*x1 + w2*x2 + ... + wn*xn
```

Then applies sigmoid:

```text
p = 1 / (1 + exp(-z))
```

The sigmoid maps any real number into:

```text
0 to 1
```

That becomes the positive-class probability estimate.

## 4.6 Log-odds contributions

For a transformed feature `x'_j` and coefficient `w_j`:

```text
contribution_j = x'_j * w_j
```

The full model is:

```text
log_odds = intercept + sum(contributions)
```

Positive contribution:

```text
pushes model log-odds upward
```

Negative contribution:

```text
pushes model log-odds downward
```

This is why our explanation is a **real model contribution explanation** for a linear model.

But:

```text
model contribution != causal effect
```

The model can learn correlation and synthetic assumptions.

## 4.7 Brier score

Brier score measures squared error of probability predictions.

For binary targets:

```text
mean((predicted_probability - actual_label)^2)
```

Lower is better.

Example:

```text
actual = 1
prediction A = 0.90
prediction B = 0.55
```

Errors:

```text
A: (0.90 - 1)^2 = 0.01
B: (0.55 - 1)^2 = 0.2025
```

A was the better probability for that example.

Important nuance:

Brier score combines probability reliability and discrimination effects. It is useful, but the calibration curve is still needed to inspect reliability visually/by bins.

## 4.8 Log loss

Log loss heavily penalizes confident wrong predictions.

A model saying:

```text
99% positive
```

when the actual answer is negative receives a large penalty.

Lower is better.

## 4.9 Calibration curve

A well-calibrated binary model should behave roughly like:

```text
among predictions near 0.80
about 80% are actually positive
```

The calibration curve compares:

```text
mean predicted probability
vs
observed positive fraction
```

Our synthetic test benchmark is **not perfectly calibrated**.

That is documented honestly.

## 4.10 Threshold costs

The project uses educational relative costs:

```text
false positive cost = 1
false negative cost = 5
```

Meaning:

```text
missing a converter is treated as 5x more costly
than reviewing a weak lead
```

These are not dollars.

For each threshold:

```text
business_cost = FP * false_positive_cost + FN * false_negative_cost
```

Then we choose the validation threshold with the lowest cost.

## `common/hashing.py`

`common/` is new. It contains tiny code used by both offline training and online inference. We do not put this helper inside `ml/` because the API also needs it. We do not put it inside `app/` because training also needs it.

```python
import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it fully into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()
```

### How to read this file


### `hashlib.sha256()`

Creates a SHA-256 digest calculator.

Think of it as a fingerprint machine.

```python
digest = hashlib.sha256()
```

### Why read in chunks?

```python
while chunk := file.read(CHUNK_SIZE):
```

A model can become large.

Bad approach:

```python
data = path.read_bytes()
```

That loads the entire model into memory.

Our approach reads one MiB at a time.

The `:=` operator is called the **walrus operator**. It assigns and tests the value in one expression.

Equivalent longer logic:

```python
chunk = file.read(CHUNK_SIZE)
while chunk:
    digest.update(chunk)
    chunk = file.read(CHUNK_SIZE)
```

### `hexdigest()`

Returns a readable hexadecimal fingerprint such as:

```text
fabea5b5d2c063...
```

Training saves this value in the manifest.

Inference calculates it again and compares.

## `ml/features.py`

This file is the **feature contract**. Project 1 had request schemas, but it did not have an ML training feature contract shared across offline code.

```python
TARGET_COLUMN = "converted"

NUMERIC_FEATURES = [
    "company_size",
    "annual_revenue_k",
    "website_visits_30d",
    "email_opens_30d",
    "demo_requests_30d",
    "form_submissions_30d",
    "days_since_last_contact",
]

CATEGORICAL_FEATURES = [
    "lead_source",
    "industry",
    "decision_maker_engaged",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAD_SOURCES = [
    "organic",
    "paid_search",
    "referral",
    "social",
    "outbound",
]

INDUSTRIES = [
    "saas",
    "ecommerce",
    "healthcare",
    "finance",
    "services",
]
```

### How to read this file


### `TARGET_COLUMN`

```python
TARGET_COLUMN = "converted"
```

This is the answer the model learns to predict.

### Numeric vs categorical features

Numeric values can be scaled:

```text
company_size
website_visits_30d
```

Categorical values must be encoded:

```text
lead_source
industry
```

`decision_maker_engaged` is boolean, but we intentionally treat it as categorical so the one-hot encoder creates explicit category indicators.

### `FEATURE_COLUMNS`

```python
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
```

This produces one ordered list.

Order matters for:

- training DataFrame selection
- CSV validation
- model manifest
- inference DataFrame construction

The manifest becomes the persisted contract for the deployed model.

## `ml/generate_data.py`

Project 1 called an external AI API. Project 2 needs training data. This module creates a deterministic synthetic dataset so anyone cloning the repo can reproduce the ML workflow without private CRM data.

```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import INDUSTRIES, LEAD_SOURCES

RANDOM_SEED = 42


def generate_synthetic_leads(rows: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a reproducible synthetic B2B lead-conversion dataset."""

    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)

    lead_source = rng.choice(
        LEAD_SOURCES,
        rows,
        p=[0.25, 0.22, 0.18, 0.15, 0.20],
    )
    industry = rng.choice(
        INDUSTRIES,
        rows,
        p=[0.25, 0.22, 0.18, 0.15, 0.20],
    )
    company_size = np.clip(
        np.rint(rng.lognormal(mean=4.0, sigma=0.8, size=rows)),
        1,
        1000,
    ).astype(int)
    annual_revenue_k = np.clip(
        rng.lognormal(mean=6.2, sigma=1.0, size=rows),
        25,
        50000,
    ).round(2)
    website_visits_30d = rng.poisson(12, size=rows)
    email_opens_30d = rng.poisson(4, size=rows)
    demo_requests_30d = np.clip(rng.poisson(0.55, size=rows), 0, 5)
    form_submissions_30d = np.clip(rng.poisson(1.1, size=rows), 0, 8)
    days_since_last_contact = np.clip(
        np.rint(rng.exponential(scale=25, size=rows)),
        0,
        180,
    ).astype(int)
    decision_maker_engaged = rng.binomial(1, 0.28, size=rows).astype(bool)

    source_effect = (
        pd.Series(lead_source)
        .map(
            {
                "organic": 0.15,
                "paid_search": 0.25,
                "referral": 0.90,
                "social": -0.15,
                "outbound": -0.45,
            }
        )
        .to_numpy()
    )
    industry_effect = (
        pd.Series(industry)
        .map(
            {
                "saas": 0.35,
                "ecommerce": 0.15,
                "healthcare": 0.10,
                "finance": 0.20,
                "services": -0.05,
            }
        )
        .to_numpy()
    )

    latent_score = (
        -3.2
        + (0.035 * website_visits_30d)
        + (0.09 * email_opens_30d)
        + (0.80 * demo_requests_30d)
        + (0.32 * form_submissions_30d)
        - (0.018 * days_since_last_contact)
        + (1.10 * decision_maker_engaged.astype(int))
        + (0.00008 * annual_revenue_k)
        + (0.0007 * company_size)
        + source_effect
        + industry_effect
        + rng.normal(0, 0.7, rows)
    )

    conversion_probability = 1 / (1 + np.exp(-latent_score))
    converted = rng.binomial(1, conversion_probability)

    return pd.DataFrame(
        {
            "company_size": company_size,
            "annual_revenue_k": annual_revenue_k,
            "website_visits_30d": website_visits_30d,
            "email_opens_30d": email_opens_30d,
            "demo_requests_30d": demo_requests_30d,
            "form_submissions_30d": form_submissions_30d,
            "days_since_last_contact": days_since_last_contact,
            "lead_source": lead_source,
            "industry": industry,
            "decision_maker_engaged": decision_maker_engaged,
            "converted": converted,
        }
    )


def save_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic lead data.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=Path("data/leads.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = generate_synthetic_leads(rows=args.rows, seed=args.seed)
    save_dataset(dataframe, args.output)

    positive_rate = dataframe["converted"].mean()
    print(f"Saved {len(dataframe)} leads to {args.output}")
    print(f"Positive conversion rate: {positive_rate:.2%}")


if __name__ == "__main__":
    main()
```

### How to read this file


### `np.random.default_rng(seed)`

```python
rng = np.random.default_rng(seed)
```

A seeded random-number generator gives reproducible pseudo-random values.

Same seed + same generator logic + compatible numerical environment gives the same deterministic dataset.

### `rng.choice`

```python
lead_source = rng.choice(
    LEAD_SOURCES,
    rows,
    p=[0.25, 0.22, 0.18, 0.15, 0.20],
)
```

This samples categories with configured probabilities.

The `p=` list must sum to 1.

### Log-normal generation

```python
rng.lognormal(...)
```

Used for company size and revenue.

This creates long-tailed positive values: many smaller observations and fewer large observations.

We then:

```text
round
clip
convert type
```

### Poisson count features

```python
rng.poisson(...)
```

Used for engagement event counts.

This is a synthetic approximation for count data.

It is **not a claim** that real website or email activity follows a perfect Poisson process.

### Latent synthetic score

The big expression assigned to `latent_score` is the hidden rule used to create a learnable synthetic target.

Example:

```python
+ (0.80 * demo_requests_30d)
- (0.018 * days_since_last_contact)
```

Meaning inside the synthetic world:

```text
more demo requests → higher latent conversion score
older contact       → lower latent conversion score
```

These are author-designed assumptions.

### Sigmoid

```python
conversion_probability = 1 / (1 + np.exp(-latent_score))
```

Maps the latent real-valued score into `[0, 1]`.

### Bernoulli/binomial target draw

```python
converted = rng.binomial(1, conversion_probability)
```

We do not simply say:

```text
probability > 0.5 = converted
```

Instead, the probability controls a random binary draw.

That adds outcome noise.

A lead with synthetic probability `0.80` can still receive target `0`.

This makes the problem more realistic than a perfectly deterministic rule.

## `ml/evaluation.py`

This entire file is new ML engineering logic. It centralizes evaluation so `train.py` does not become one giant function containing metrics and plotting code.

```python
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
```

### How to read this file


### `calculate_metrics`

Input:

```text
y_true
probabilities
threshold
```

First:

```python
predictions = (probabilities >= threshold).astype(int)
```

NumPy compares the full array at once.

Example:

```text
probabilities = [0.20, 0.70, 0.55]
threshold = 0.50
```

Comparison:

```text
[False, True, True]
```

`astype(int)`:

```text
[0, 1, 1]
```

This is **vectorization**.

We do not write a Python loop for every probability.

### Confusion matrix unpacking

```python
true_negative, false_positive, false_negative, true_positive = matrix.ravel()
```

For binary labels `[0, 1]`, scikit-learn's matrix layout is:

```text
[[TN, FP],
 [FN, TP]]
```

`.ravel()` flattens it.

### Brier and log loss

Notice these use probabilities directly:

```python
brier_score_loss(y_true, probabilities)
log_loss(y_true, probabilities)
```

Precision/recall/F1 require class decisions.

Brier/log loss evaluate probability predictions.

That difference is why threshold policy and probability model can be separated.

### `build_calibration_data`

```python
fraction_positive, mean_predicted = calibration_curve(...)
```

The test predictions are split into probability bins.

For each bin we store:

```text
mean predicted probability
observed positive fraction
```

These bins are returned by `/model/metrics`.

### `analyze_thresholds`

```python
np.arange(0.05, 0.951, 0.01)
```

Creates threshold candidates:

```text
0.05
0.06
0.07
...
0.95
```

For every threshold we calculate:

```text
precision
recall
F1
false positives
false negatives
business cost
```

### Why thresholds stop at 0.05 and 0.95

The educational analysis avoids extreme `0` and `1` boundaries where nearly every row collapses to one class.

For a real business, threshold search range and resolution should be chosen based on the application.

### F1 best

```python
f1_best = max(...)
```

Higher F1 wins.

### Cost best

```python
cost_best = min(...)
```

Lower business cost wins.

### Tie-breaking

The `key=` tuples make selection deterministic when values tie.

For F1:

```python
(f1, threshold)
```

Higher threshold wins a perfect F1 tie.

For cost:

```python
(business_cost, -threshold)
```

Lower cost wins; then higher threshold wins a cost tie.

### Three strategies

The function stores:

```text
default
f1
business_cost
```

This lets the README and metrics endpoint compare policies rather than show only the winner.

### Plot functions

All plot functions save files and close the figure.

```python
plt.savefig(...)
plt.close()
```

Why close?

Training processes can create multiple plots. Leaving figures open wastes memory.

### `matplotlib.use("Agg")`

This selects a non-interactive backend.

CI and Docker trainer stages do not have a desktop window.

`Agg` lets Matplotlib render PNGs headlessly.

## `ml/train.py`

This is the core offline ML lifecycle. Read it from top to bottom as a pipeline of decisions. It is the most important new file in Project 2.

```python
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
```

### How to read this file


### Constants

```python
RANDOM_SEED = 42
MODEL_VERSION = "2.0.0"
SCHEMA_VERSION = "1.0"
SELECTION_METRIC = "brier_score"
```

`MODEL_VERSION` describes the trained model release.

`SCHEMA_VERSION` describes the artifact/feature contract generation.

They are not automatically the same concept.

### `ThresholdStrategy = Literal[...]`

```python
ThresholdStrategy = Literal["default", "f1", "business_cost"]
```

This gives type checkers and readers an explicit set of supported policy names.

### `build_pipeline`

This creates:

```text
ColumnTransformer
      ↓
LogisticRegression
```

Inside `ColumnTransformer`:

```text
numeric columns     → StandardScaler
categorical columns → OneHotEncoder
```

### Why `Pipeline` matters

Bad manual flow:

```text
fit scaler
transform train
forget exact steps
serve API
manually recreate transform
```

Our flow:

```python
pipeline.predict_proba(frame)
```

The saved pipeline already contains fitted preprocessing and model coefficients.

### `StandardScaler`

Conceptually:

```text
z = (x - training_mean) / training_standard_deviation
```

A company size of `120` is not sent raw to the classifier.

It is transformed using statistics learned from training data.

### `OneHotEncoder`

A category such as:

```text
lead_source = referral
```

becomes indicator columns such as:

```text
lead_source_organic  = 0
lead_source_referral = 1
lead_source_social   = 0
```

`handle_unknown="ignore"` prevents inference from crashing at the encoder level for an unseen category.

Our Pydantic API schema is stricter and currently rejects unknown category strings before they reach the model.

### Why `sparse_output=False`

One-hot encoding often produces sparse matrices.

For this small educational feature space, dense arrays simplify local contribution calculation:

```python
transformed_values * coefficients
```

For very high-cardinality production features, sparse representations may be preferable.

### `_validate_dataset`

Checks required columns before training.

This catches:

```text
wrong CSV
renamed feature
missing target
```

before scikit-learn fails deeper in the pipeline.

It also checks target values are only `0` and `1`.

### First split

```python
x_train_validation, x_test, ... = train_test_split(... test_size=0.20)
```

Twenty percent is sealed as final test data.

### Second split

```python
x_train, x_validation, ... = train_test_split(x_train_validation, ... test_size=0.20)
```

Twenty percent of the remaining 80% becomes validation.

Math:

```text
80% × 80% = 64% train
80% × 20% = 16% validation
20%       = 20% test
```

For 5,000 rows:

```text
3200 train
800 validation
1000 test
```

### `stratify=target`

This tries to preserve class proportions across splits.

Without stratification, an imbalanced target can produce uneven positive rates by chance.

### Candidate specifications

```python
candidate_specs = {
    "logistic_default": None,
    "logistic_balanced": "balanced",
}
```

The model algorithm stays the same.

Only `class_weight` changes.

This lets you study class weighting without jumping to a complex estimator.

### Candidate fitting

Each candidate is fit only on `x_train`.

Then probabilities are generated on `x_validation`.

The candidate result includes:

```text
accuracy
precision
recall
F1
ROC-AUC
Brier score
log loss
```

### Why select by minimum Brier score?

```python
selected_candidate = min(
    candidate_results,
    key=lambda item: item[SELECTION_METRIC],
)
```

Our API exposes `conversion_probability`.

The default logistic candidate produced better validation Brier score:

```text
logistic_default  ≈ 0.157
logistic_balanced ≈ 0.196
```

Lower is better.

The balanced candidate had higher validation F1 at threshold 0.50.

But threshold 0.50 is a policy choice.

We choose the probability model first.

Then tune the policy threshold separately.

This is a major design improvement over:

```text
highest accuracy wins
```

or:

```text
highest F1 at 0.50 wins
```

when your product exposes probabilities.

### Threshold analysis uses validation probabilities

```python
threshold_analysis = analyze_thresholds(...)
```

We do **not** inspect test labels to choose the threshold.

Default project policy:

```text
FP cost = 1
FN cost = 5
```

Selected validation threshold:

```text
0.24
```

### Refit after decisions

```python
selected_pipeline.fit(x_train_validation, y_train_validation)
```

After candidate and threshold decisions are finished, we fit the chosen model specification on:

```text
train + validation
```

Why?

We can now use more labeled data to estimate final coefficients.

The threshold was already chosen from validation predictions produced by the train-only model.

### Final test evaluation

```python
test_probabilities = selected_pipeline.predict_proba(x_test)[:, 1]
```

Now we evaluate on the untouched test set.

Actual final metrics:

```text
accuracy  = 0.652
precision = 0.409
recall    = 0.685
F1        = 0.513
ROC-AUC   = 0.729
Brier     = 0.169
log loss  = 0.512
```

Notice validation and test scores differ.

That is normal.

The test set is a different sample.

### Metadata vs metrics vs manifest

Three concepts:

#### Metadata

Lives inside `model.joblib`.

Describes the trained model:

```text
version
algorithm
selected model
threshold
features
library versions
```

#### Metrics

Lives in `metrics.json`.

Describes evaluation:

```text
test metrics
confusion matrix
calibration bins
threshold analysis
candidate comparison
```

#### Manifest

Lives in `model_manifest.json`.

Describes the artifact contract:

```text
hash
feature order
schema version
model version
threshold
library versions
```

### Why save model before creating manifest?

We need the final model file bytes before calculating:

```python
sha256_file(model_path)
```

Flow:

```text
joblib.dump
   ↓
model file exists
   ↓
calculate SHA-256
   ↓
write manifest
```

### Training CLI

`argparse` supports:

```text
--data
--artifacts
--threshold-strategy
--false-positive-cost
--false-negative-cost
```

This makes training reusable from CI, Docker, and a developer terminal.

# 5. Actual benchmark interpretation


## Dataset

```text
rows          = 5000
positive rate = 26.70%
```

## Candidate comparison on validation

```text
logistic_default
  Brier   = 0.157
  log loss= 0.480
  F1@0.50 = 0.448

logistic_balanced
  Brier   = 0.196
  log loss= 0.579
  F1@0.50 = 0.578
```

The balanced model recovers more positives at the default threshold.

But it also shifts probabilities upward and has worse probability losses.

Because we expose a probability, `logistic_default` is selected.

## Threshold strategy comparison on validation

### `default`

```text
threshold = 0.50
precision = 0.608
recall    = 0.355
F1        = 0.448
FP        = 49
FN        = 138
cost      = 739
```

### `f1`

```text
threshold = 0.26
precision = 0.479
recall    = 0.757
F1        = 0.587
FP        = 176
FN        = 52
cost      = 436
```

### `business_cost`

```text
threshold = 0.24
precision = 0.458
recall    = 0.780
F1        = 0.577
FP        = 198
FN        = 47
cost      = 433
```


Business-cost threshold wins on validation:

```text
threshold = 0.24
```

## Final test confusion matrix

```text
                  actual 0    actual 1
predicted 0          469          84
predicted 1          264          183
```

Recall:

```text
TP / (TP + FN)
= 183 / (183 + 84)
≈ 0.685
```

Precision:

```text
TP / (TP + FP)
= 183 / (183 + 264)
≈ 0.409
```

The selected business policy accepts more false positives to reduce false negatives.

That is not "model good" or "model bad" by itself.

It is a trade-off caused by the chosen operational policy.

## `app/core/config.py`

You already know Pydantic Settings from Project 1. The new part is that configuration now points to ML artifacts and sets a batch limit.

```python
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="Lead Scoring ML API", alias="APP_NAME")
    app_env: Literal["local", "development", "staging", "production"] = Field(
        default="local",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    model_path: Path = Field(default=Path("artifacts/model.joblib"), alias="MODEL_PATH")
    metrics_path: Path = Field(default=Path("artifacts/metrics.json"), alias="METRICS_PATH")
    manifest_path: Path = Field(
        default=Path("artifacts/model_manifest.json"),
        alias="MANIFEST_PATH",
    )
    dataset_path: Path = Field(default=Path("data/leads.csv"), alias="DATASET_PATH")
    max_batch_size: int = Field(default=1000, alias="MAX_BATCH_SIZE", ge=1, le=10000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
```

### How to read this file


New fields:

```python
model_path
metrics_path
manifest_path
dataset_path
max_batch_size
```

### Why paths are configuration

Local development:

```text
artifacts/model.joblib
```

A deployment may mount artifacts elsewhere.

Environment configuration means application code does not hard-code one machine layout.

### `max_batch_size`

```python
Field(default=1000, ge=1, le=10000)
```

Two different boundaries exist:

```text
Pydantic Settings validates MAX_BATCH_SIZE itself
ModelService enforces the configured limit at runtime
```

The request schema only guarantees that the batch is non-empty:

```python
leads: list[LeadFeatures] = Field(min_length=1)
```

Why not hard-code `max_length=1000` in the schema?

Because `MAX_BATCH_SIZE` is configurable. A fixed schema limit and a configurable service limit can contradict each other. The service is the single source of truth for the maximum batch size.

## `app/schemas/lead.py`

Project 1 had simple chat request/response schemas. Project 2 introduces a typed ML feature contract, local contribution schema, and batch request/response schemas.

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LeadSource = Literal["organic", "paid_search", "referral", "social", "outbound"]
Industry = Literal["saas", "ecommerce", "healthcare", "finance", "services"]
PredictionLabel = Literal["qualified", "not_qualified"]
ConfidenceLevel = Literal["low", "medium", "high"]
ContributionDirection = Literal["positive", "negative"]


class LeadFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lead_source": "referral",
                "industry": "saas",
                "company_size": 120,
                "annual_revenue_k": 2500,
                "website_visits_30d": 28,
                "email_opens_30d": 8,
                "demo_requests_30d": 2,
                "form_submissions_30d": 3,
                "days_since_last_contact": 4,
                "decision_maker_engaged": True,
            }
        }
    )

    lead_source: LeadSource
    industry: Industry
    company_size: int = Field(ge=1, le=1000)
    annual_revenue_k: float = Field(ge=0, le=50000)
    website_visits_30d: int = Field(ge=0, le=500)
    email_opens_30d: int = Field(ge=0, le=200)
    demo_requests_30d: int = Field(ge=0, le=20)
    form_submissions_30d: int = Field(ge=0, le=50)
    days_since_last_contact: int = Field(ge=0, le=365)
    decision_maker_engaged: bool


class FeatureContribution(BaseModel):
    feature: str
    log_odds_contribution: float
    direction: ContributionDirection


class LeadScoreData(BaseModel):
    prediction: PredictionLabel
    lead_score: float = Field(ge=0, le=100)
    conversion_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    threshold: float = Field(ge=0, le=1)
    top_positive_factors: list[FeatureContribution]
    top_negative_factors: list[FeatureContribution]
    explanation: str
    model_version: str


class BatchLeadScoreRequest(BaseModel):
    leads: list[LeadFeatures] = Field(min_length=1)


class BatchLeadScoreData(BaseModel):
    count: int = Field(ge=1)
    predictions: list[LeadScoreData]
```

### How to read this file


### `Literal`

```python
LeadSource = Literal["organic", ...]
```

Only listed strings are valid.

Swagger also displays the allowed enum-like values.

### Numeric constraints

```python
company_size: int = Field(ge=1, le=1000)
```

`ge` means greater than or equal.

`le` means less than or equal.

This is input-range validation before ML inference.

### `ConfigDict(json_schema_extra=...)`

This adds an example to generated OpenAPI/Swagger docs.

It does not change validation.

It improves developer usability.

### `FeatureContribution`

We explicitly name the number:

```python
log_odds_contribution
```

Not:

```text
importance
confidence
causal_effect
```

The precise name reduces misleading interpretation.

### Batch request

```python
leads: list[LeadFeatures] = Field(min_length=1, max_length=1000)
```

The whole list is validated.

Every item is also validated as `LeadFeatures`.

This is nested Pydantic validation.

## `app/schemas/model.py`

This file turns evaluation JSON into typed API contracts. The new idea is that model observability endpoints have schemas just like business endpoints.

```python
from pydantic import BaseModel, Field


class ConfusionMatrixData(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class MetricData(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0)
    log_loss: float = Field(ge=0)


class CandidateMetricData(MetricData):
    model_name: str


class CalibrationBinData(BaseModel):
    mean_predicted_probability: float = Field(ge=0, le=1)
    fraction_positive: float = Field(ge=0, le=1)


class CalibrationData(BaseModel):
    brier_score: float = Field(ge=0)
    log_loss: float = Field(ge=0)
    bins: list[CalibrationBinData]


class ThresholdStrategyMetricData(BaseModel):
    strategy: str
    threshold: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_positive: int
    false_negative: int
    business_cost: float = Field(ge=0)


class ThresholdAnalysisData(BaseModel):
    default_threshold: float = Field(ge=0, le=1)
    f1_optimal_threshold: float = Field(ge=0, le=1)
    business_cost_threshold: float = Field(ge=0, le=1)
    selected_threshold: float = Field(ge=0, le=1)
    selected_strategy: str
    false_positive_cost: float = Field(ge=0)
    false_negative_cost: float = Field(ge=0)
    strategies: list[ThresholdStrategyMetricData]


class MetricsData(BaseModel):
    selected_model: str
    selection_metric: str
    selection_split: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    test_rows: int
    positive_rate: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    metrics: MetricData
    confusion_matrix: ConfusionMatrixData
    calibration: CalibrationData
    threshold_analysis: ThresholdAnalysisData
    candidates: list[CandidateMetricData]


class FeatureWeight(BaseModel):
    feature: str
    coefficient: float


class ModelInfoData(BaseModel):
    model_version: str
    schema_version: str
    algorithm: str
    selected_model: str
    trained_at_utc: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    test_rows: int
    target: str
    threshold: float
    threshold_strategy: str
    numeric_features: list[str]
    categorical_features: list[str]
    top_positive_features: list[FeatureWeight]
    top_negative_features: list[FeatureWeight]
    library_versions: dict[str, str]


class ModelManifestData(BaseModel):
    schema_version: str
    model_version: str
    algorithm: str
    selected_model: str
    trained_at_utc: str
    model_sha256: str
    dataset_sha256: str
    feature_contract: list[str]
    feature_count: int = Field(ge=1)
    target: str
    threshold: float = Field(ge=0, le=1)
    threshold_strategy: str
    library_versions: dict[str, str]
```

### How to read this file


### Why separate schema classes?

`MetricData`:

```text
final scalar test metrics
```

`CalibrationData`:

```text
probability reliability diagnostics
```

`ThresholdAnalysisData`:

```text
decision policy comparison
```

`ModelManifestData`:

```text
artifact contract and hashes
```

One giant `dict[str, Any]` would be easier to write but worse for:

- Swagger
- validation
- refactoring
- readers

### `CandidateMetricData(MetricData)`

This is inheritance.

It means:

```text
CandidateMetricData has every MetricData field
+ model_name
```

We reuse the scalar metric contract.

## `app/api/v1/endpoints/health.py`

The liveness route is from Project 1. The new route is readiness, because an ML API process can be alive while required model artifacts are missing.

```python
from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.schemas.common import SuccessResponse
from app.schemas.health import HealthData, ReadinessData

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=SuccessResponse[HealthData])
def health_check() -> SuccessResponse[HealthData]:
    return SuccessResponse(
        data=HealthData(
            app=settings.app_name,
            environment=settings.app_env,
        )
    )


@router.get("/ready", response_model=SuccessResponse[ReadinessData])
def readiness_check(response: Response) -> SuccessResponse[ReadinessData]:
    model_exists = settings.model_path.exists()
    metrics_exists = settings.metrics_path.exists()
    manifest_exists = settings.manifest_path.exists()
    ready = model_exists and metrics_exists and manifest_exists

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return SuccessResponse(
        data=ReadinessData(
            ready=ready,
            model_artifact_exists=model_exists,
            metrics_artifact_exists=metrics_exists,
            manifest_artifact_exists=manifest_exists,
        )
    )
```

### How to read this file


### Liveness vs readiness

Liveness asks:

```text
Is the web process alive?
```

Readiness asks:

```text
Are required serving artifacts present?
```

Current readiness checks existence of:

```text
model
metrics
manifest
```

Full integrity validation happens when `ModelService` loads.

If any required file is missing:

```python
response.status_code = HTTP_503_SERVICE_UNAVAILABLE
```

The API process still exists, but it is not ready to serve ML inference.

## `app/api/v1/endpoints/prediction.py`

This file shows how the Project 1 service-layer pattern scales from one request to batch ML inference.

```python
from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.common import SuccessResponse
from app.schemas.lead import (
    BatchLeadScoreData,
    BatchLeadScoreRequest,
    LeadFeatures,
    LeadScoreData,
)
from app.services.model_service import ModelService, get_model_service

router = APIRouter()
ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/predict-lead-score",
    response_model=SuccessResponse[LeadScoreData],
)
def predict_lead_score(
    lead: LeadFeatures,
    service: ModelServiceDependency,
) -> SuccessResponse[LeadScoreData]:
    prediction = service.predict(lead)
    return SuccessResponse(data=prediction)


@router.post(
    "/predict-lead-scores",
    response_model=SuccessResponse[BatchLeadScoreData],
)
def predict_lead_scores(
    request: BatchLeadScoreRequest,
    service: ModelServiceDependency,
) -> SuccessResponse[BatchLeadScoreData]:
    predictions = service.predict_many(request.leads)
    return SuccessResponse(
        data=BatchLeadScoreData(
            count=len(predictions),
            predictions=predictions,
        )
    )
```

### How to read this file


### Single route

```python
service.predict(lead)
```

One validated lead in, one score out.

### Batch route

```python
service.predict_many(request.leads)
```

The route still has no model logic.

It only:

```text
accepts HTTP input
calls service
wraps typed response
```

### Why batch is a separate route

A batch request has different operational meaning:

```text
one lead  → interactive scoring
many leads→ CRM/import workflow
```

The response includes:

```python
count=len(predictions)
```

This is useful for clients verifying batch result size.

## `app/api/v1/endpoints/model.py`

Project 1 did not have model observability routes. Project 2 exposes read-only metadata so a consumer can inspect what model is serving.

```python
from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.common import SuccessResponse
from app.schemas.model import MetricsData, ModelInfoData, ModelManifestData
from app.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/model")
ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]


@router.get("/info", response_model=SuccessResponse[ModelInfoData])
def model_info(service: ModelServiceDependency) -> SuccessResponse[ModelInfoData]:
    return SuccessResponse(data=service.get_model_info())


@router.get("/metrics", response_model=SuccessResponse[MetricsData])
def model_metrics(service: ModelServiceDependency) -> SuccessResponse[MetricsData]:
    return SuccessResponse(data=service.get_metrics())


@router.get("/manifest", response_model=SuccessResponse[ModelManifestData])
def model_manifest(service: ModelServiceDependency) -> SuccessResponse[ModelManifestData]:
    return SuccessResponse(data=service.get_manifest())
```

### How to read this file


Routes:

```text
GET /model/info
GET /model/metrics
GET /model/manifest
```

### `/model/info`

Answers:

```text
What model is loaded?
When was it trained?
What threshold is active?
What feature groups exist?
```

### `/model/metrics`

Answers:

```text
How did the selected pipeline behave on held-out test data?
What did candidate validation look like?
How was the threshold selected?
What do calibration bins look like?
```

### `/model/manifest`

Answers:

```text
What artifact hash is expected?
What exact feature order is expected?
What schema/model version is active?
```

These are observability routes, not training routes.

The API never trains a model in an HTTP request.

## `app/services/model_service.py`

This is the core online inference file. In Project 1, `GeminiService` managed an external AI SDK. Here, `ModelService` manages model lifecycle, integrity, vectorized inference, threshold policy, and local explanations.

```python
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
```

### How to read this file


### Constructor lifecycle

Read this order carefully:

```text
get settings
↓
require artifacts
↓
read manifest
↓
verify model hash
↓
joblib.load
↓
read metrics
↓
verify bundle contract
↓
cache feature names and classifier
```

The order is security/reliability conscious.

We verify file hash **before** loading the pickle/joblib-style artifact.

### `AppError` pass-through

```python
except AppError:
    raise
```

Why?

Our controlled errors already have meaningful codes such as:

```text
MODEL_ARTIFACT_HASH_MISMATCH
```

If we caught them as generic `Exception`, we would hide the real code behind:

```text
MODEL_ARTIFACT_LOAD_FAILED
```

### `predict` calls `predict_many`

```python
return self.predict_many([lead])[0]
```

We keep one inference implementation.

Bad design:

```text
single prediction logic
batch prediction logic
```

Two copies can drift.

Our design:

```text
predict_many = core implementation
predict = one-item adapter
```

### Manifest-controlled feature order

```python
columns=self.manifest["feature_contract"]
```

A Python dictionary preserves insertion order today, but we do not rely on incidental request serialization order.

We explicitly build the DataFrame in the order persisted with the model contract.

### Vectorized `predict_proba`

```python
probabilities = self.pipeline.predict_proba(frame)[:, 1]
```

Suppose the result is:

```text
[[0.90, 0.10],
 [0.25, 0.75]]
```

Columns represent class probabilities.

For binary classes `[0, 1]`:

```text
column 0 = P(class 0)
column 1 = P(class 1)
```

`[:, 1]` means:

```text
all rows
column 1
```

Result:

```text
[0.10, 0.75]
```

### Local contribution matrix

```python
contributions = transformed_array * self.classifier.coef_[0]
```

Shapes:

```text
transformed_array        = (number_of_leads, transformed_feature_count)
classifier.coef_[0]      = (transformed_feature_count,)
```

NumPy broadcasting applies the coefficient vector to every row.

Result:

```text
(number_of_leads, transformed_feature_count)
```

Each cell is one local log-odds term.

### `zip(..., strict=True)`

```python
zip(leads, probabilities, contributions, strict=True)
```

Normal `zip` silently stops at the shortest iterable.

`strict=True` raises if lengths differ.

That protects the assumption:

```text
one lead ↔ one probability ↔ one contribution row
```

### `_top_contributions`

We remove nearly-zero values:

```python
if abs(float(value)) > 1e-9
```

Then split:

```text
positive > 0
negative < 0
```

Positive sorted descending.

Negative sorted ascending so the most negative values come first.

We return the top three from each direction.

### Humanizing transformed names

Scikit-learn names can look like:

```text
categorical__lead_source_referral
```

The API converts that into:

```text
lead_source=referral
```

This is presentation logic.

The numeric model calculation still uses scikit-learn's transformed feature array.

### Threshold-aware confidence

Old Project 2 used distance from `0.50`.

That becomes wrong after threshold tuning.

Our selected threshold is `0.24`.

A probability of `0.24` is exactly on the decision boundary and should have confidence distance `0`.

Formula:

```python
abs(probability - threshold) / max(threshold, 1 - threshold)
```

It normalizes distance by the farther possible probability boundary.

This is **operational boundary distance**.

It is not calibrated certainty.

### Explanation text

The explanation uses actual contribution objects.

We no longer use manually authored rules such as:

```text
if demo requests > 0: say demo interest
```

Instead:

```text
model transformed row
×
learned coefficient
=
reported log-odds contribution
```

The final sentence explicitly says:

```text
not causal effects
```

That honesty matters.

### `@lru_cache`

Same idea as cached settings in Project 1.

```python
@lru_cache
def get_model_service():
```

We do not want to:

```text
hash model
load joblib
extract pipeline
```

for every HTTP request.

The successful service object is reused in the process.

## `scripts/score_csv.py`

This is the first project CLI that reuses the same service outside FastAPI. It proves the service layer is not coupled to HTTP.

```python
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
```

### How to read this file


### `Protocol`

```python
class BatchPredictor(Protocol):
    def predict_many(...): ...
```

A protocol describes required behavior.

`score_csv` does not care whether the object is literally `ModelService`.

It only needs an object with compatible:

```python
predict_many(...)
```

That is structural typing.

It also makes testing easy because we can pass `FakeBatchPredictor`.

### CSV feature validation

```python
missing_columns = set(FEATURE_COLUMNS) - set(dataframe.columns)
```

If a required column is absent, we fail before model inference.

### `to_dict(orient="records")`

A DataFrame like:

```text
row 1
row 2
```

becomes:

```python
[
    {"feature_a": ..., "feature_b": ...},
    {"feature_a": ..., "feature_b": ...},
]
```

Then every record becomes a Pydantic `LeadFeatures` model.

So the CLI reuses the same feature validation contract as the API.

### One batch call

```python
predictions = predictor.predict_many(leads)
```

We do not call the model once per CSV row.

### JSON inside CSV cells

Top factor lists are nested structures.

CSV has flat cells.

We serialize the list using:

```python
json.dumps(...)
```

The CSV cell contains JSON text representing the factor list.

# 6. API route map


The full API routes are:

| Method | Full route | Endpoint file | Service call |
|---|---|---|---|
| GET | `/api/v1/health` | `health.py` | none |
| GET | `/api/v1/ready` | `health.py` | none |
| GET | `/api/v1/model/info` | `model.py` | `get_model_info()` |
| GET | `/api/v1/model/metrics` | `model.py` | `get_metrics()` |
| GET | `/api/v1/model/manifest` | `model.py` | `get_manifest()` |
| POST | `/api/v1/predict-lead-score` | `prediction.py` | `predict()` |
| POST | `/api/v1/predict-lead-scores` | `prediction.py` | `predict_many()` |

## How route prefixes stack

`app/main.py`:

```python
prefix=settings.api_v1_prefix
```

Default:

```text
/api/v1
```

`model.py` router:

```python
APIRouter(prefix="/model")
```

Endpoint:

```python
@router.get("/metrics")
```

Final route:

```text
/api/v1 + /model + /metrics
=
/api/v1/model/metrics
```

This is the same router composition concept from Project 1, now with a nested model route group.

# 7. Real prediction walkthrough


Input:

```json
{
  "lead_source": "referral",
  "industry": "saas",
  "company_size": 120,
  "annual_revenue_k": 2500,
  "website_visits_30d": 28,
  "email_opens_30d": 8,
  "demo_requests_30d": 2,
  "form_submissions_30d": 3,
  "days_since_last_contact": 4,
  "decision_maker_engaged": true
}
```

## Step 1 — Pydantic

`LeadFeatures` validates:

```text
category strings
integer ranges
float range
boolean
```

## Step 2 — DataFrame

The service creates one row using manifest feature order.

## Step 3 — Preprocessor

Numeric values are standardized.

Categories are one-hot encoded.

## Step 4 — Logistic regression

The model calculates log-odds and sigmoid probability.

Actual deterministic result:

```text
probability = 0.9744
```

## Step 5 — Lead score

```text
lead_score = probability × 100
           = 97.44
```

## Step 6 — Threshold

```text
0.9744 >= 0.24
```

Therefore:

```text
qualified
```

## Step 7 — Confidence distance

The prediction is far above the selected boundary.

Result:

```text
0.9663
high
```

## Step 8 — Contribution calculation

Top positive local terms:

```text
demo_requests_30d   +1.1483 log-odds
lead_source=referral +0.5977 log-odds
website_visits_30d  +0.5647 log-odds
```

These are not manually invented labels.

They come from:

```text
transformed value × fitted coefficient
```

## `Dockerfile`

Project 1 had one runtime build. Project 2 introduces a true training stage because the container needs a generated model artifact before the API starts.

```dockerfile
FROM python:3.12-slim AS trainer

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY common ./common
COPY ml ./ml
COPY data ./data

RUN python -m ml.train


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY common ./common
COPY --from=trainer /build/artifacts ./artifacts

EXPOSE 8000

HEALTHCHECK     --interval=30s     --timeout=3s     --start-period=10s     --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### How to read this file


### Stage 1: `trainer`

```dockerfile
FROM python:3.12-slim AS trainer
```

This stage installs ML plotting dependencies and runs:

```dockerfile
RUN python -m ml.train
```

It creates `/build/artifacts`.

### Stage 2: `runtime`

Starts from a fresh Python image.

It installs only `requirements.txt`.

It copies:

```text
app
common
trained artifacts
```

It does not copy the ML plotting dependency set.

### Cross-stage copy

```dockerfile
COPY --from=trainer /build/artifacts ./artifacts
```

This is the key multi-stage line.

It says:

```text
from trainer stage
copy generated artifacts
into runtime stage
```

### Why `common/` is copied to both stages

Training needs hashing to build the manifest.

Runtime needs hashing to verify the model.

That is exactly why the helper lives in a shared package.

## `.github/workflows/ci.yml`

Project 1 CI ran quality, tests, and Docker. Project 2 adds an ML training reproducibility gate and a hard coverage floor.

```yaml
name: CI

on:
  push:
    branches:
      - main

  pull_request:
    branches:
      - main

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    name: Lint and Format
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-dev.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Lint
        run: ruff check .

      - name: Check formatting
        run: ruff format --check .

  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-dev.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest \
            --cov=app \
            --cov=common \
            --cov=ml \
            --cov=scripts \
            --cov-report=term-missing \
            --cov-report=xml \
            --cov-fail-under=80

  train-smoke:
    name: Train and Verify Artifacts
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-ml.txt

      - name: Install ML dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ml.txt

      - name: Train model
        run: python -m ml.train --artifacts .ci-artifacts

      - name: Verify generated artifacts
        run: |
          test -f .ci-artifacts/model.joblib
          test -f .ci-artifacts/metrics.json
          test -f .ci-artifacts/model_manifest.json
          test -f .ci-artifacts/confusion_matrix.png
          test -f .ci-artifacts/calibration_curve.png
          test -f .ci-artifacts/threshold_analysis.png

  docker:
    name: Build Docker Image
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Build image
        run: docker build --tag lead-scoring-ml-api:ci .
```

### How to read this file


### Quality job

Same concept as Project 1:

```text
Ruff lint
Ruff format check
```

### Test job

Coverage now includes:

```text
app
common
ml
scripts
```

The new gate:

```text
--cov-fail-under=80
```

CI fails if measured coverage drops below 80%.

### Training smoke job

This is ML-specific CI.

```python
python -m ml.train --artifacts .ci-artifacts
```

Then shell commands verify all six expected files.

Why use `.ci-artifacts`?

It avoids mixing the smoke-test outputs with repository evidence artifacts.

### Docker job

The Docker build also runs the trainer stage.

Therefore Docker validates that the model can be generated in a clean Linux build environment.

## `.github/workflows/codeql.yml`

CodeQL is new security automation. Ruff finds Python quality issues; CodeQL performs semantic security analysis.

```yaml
name: CodeQL

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  schedule:
    - cron: "17 3 * * 1"

permissions:
  contents: read
  security-events: write

jobs:
  analyze:
    name: Analyze Python
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v4
        with:
          languages: python
          build-mode: none

      - name: Analyze
        uses: github/codeql-action/analyze@v4
```

### How to read this file


### Triggers

```text
push to main
pull request to main
weekly schedule
```

Cron:

```text
17 3 * * 1
```

Means weekly on Monday at 03:17 UTC.

### Permissions

```yaml
security-events: write
```

CodeQL needs permission to upload analysis results.

### `build-mode: none`

Python is interpreted.

The CodeQL Action documentation says interpreted languages use `none` build mode.

### Two CodeQL actions

```text
init    → initialize analysis
analyze → finalize database, run queries, upload results
```

## `.github/workflows/release.yml`

This workflow turns a version tag into a reproducible GitHub Release with evaluation evidence.

```yaml
name: Create Release

on:
  push:
    tags:
      - "v*.*.*"

permissions:
  contents: write

jobs:
  release:
    name: Train and Create GitHub Release
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-ml.txt

      - name: Install ML dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ml.txt

      - name: Train release artifacts
        run: python -m ml.train

      - name: Create release with model and evaluation assets
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "${GITHUB_REF_NAME}" \
            artifacts/model.joblib \
            artifacts/metrics.json \
            artifacts/model_manifest.json \
            artifacts/confusion_matrix.png \
            artifacts/calibration_curve.png \
            artifacts/threshold_analysis.png \
            --generate-notes \
            --title "${GITHUB_REF_NAME}"
```

### How to read this file


### Tag trigger

```yaml
"v*.*.*"
```

Examples:

```text
v2.0.0
v2.1.3
```

### Why retrain in release workflow?

The release assets should be generated from the tagged repository state in a clean GitHub runner.

The workflow installs ML dependencies and runs:

```text
python -m ml.train
```

### `GH_TOKEN`

GitHub CLI reads this environment variable for authentication.

```yaml
GH_TOKEN: ${{ github.token }}
```

### `gh release create`

Creates the release and uploads evaluation assets.

The release uploads the exact `model.joblib` that was hashed into `model_manifest.json`.

That makes the release manifest useful: a consumer can verify the downloaded model SHA-256 against the manifest.

The release includes:

```text
model.joblib
metrics.json
model_manifest.json
three evaluation plots
```

The model is still ignored by normal Git commits because it is generated binary state. The official tagged release is the trusted distribution boundary.

### `--generate-notes`

GitHub generates release notes based on repository history between releases.

## `.github/workflows/publish-ghcr.yml`

This workflow publishes the Docker image to GitHub Container Registry directly from a pushed semantic-version tag.

```yaml
name: Publish GHCR Image

on:
  push:
    tags:
      - "v*.*.*"

permissions:
  contents: read
  packages: write

jobs:
  publish:
    name: Build and Push Container
    runs-on: ubuntu-latest

    env:
      REGISTRY: ghcr.io
      IMAGE_NAME: ${{ github.repository }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Log in to GHCR
        uses: docker/login-action@v4
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract image metadata
        id: meta
        uses: docker/metadata-action@v6
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=tag
            type=raw,value=latest

      - name: Build and push image
        uses: docker/build-push-action@v7
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          annotations: ${{ steps.meta.outputs.annotations }}
```

### How to read this file


### Why the GHCR workflow also triggers directly on the tag

Both release workflows begin from the same user-pushed tag:

```text
                 ┌─→ release workflow → GitHub Release
push v2.0.0 ─────┤
                 └─→ GHCR workflow   → container image
```

We do **not** depend on this chain:

```text
release workflow creates release with GITHUB_TOKEN
  ↓
release event starts another workflow
```

GitHub prevents most `GITHUB_TOKEN`-triggered events from recursively starting new workflow runs. Independent tag triggers avoid that hidden automation failure.

### Registry

```text
ghcr.io
```

### Image name

```yaml
IMAGE_NAME: ${{ github.repository }}
```

For this repository, conceptually:

```text
Saifee16/lead-scoring-ml-api
```

### Authentication

```yaml
username: ${{ github.actor }}
password: ${{ secrets.GITHUB_TOKEN }}
```

No Docker Hub password is needed.

### Metadata action

Generates Docker tags and labels.

We configure:

```text
release tag
latest
```

### Build/push action

```yaml
push: true
```

The Docker image is built from the repository Dockerfile and pushed to GHCR.

The workflow follows the current official Docker action major versions used in their upstream examples: login v4, metadata v6, and build-push v7. Dependabot monitors GitHub Actions updates.

# 8. Tests: what is new


The API test technique itself came from Project 1.

The **new test ideas** are ML-specific.

## `tests/conftest.py`

The fixture structure is familiar from Project 1, but `FakeModelService` now mirrors a richer ML service contract including batch scoring, metrics, and manifest output.

```python
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
```

### How to read this file


### Why fake the model service for endpoint tests?

Endpoint tests should answer:

```text
Does HTTP wiring work?
Does Pydantic validate?
Does the route call the service contract?
Does JSON shape match?
```

They should not repeatedly train or load a real model.

The real model service gets its own tests.

### Dependency override

Same FastAPI pattern as Project 1:

```python
app.dependency_overrides[get_model_service] = lambda: FakeModelService()
```

This replaces the production dependency only during API tests.

### Why fake `predict_many` too?

The batch endpoint depends on the same service interface.

The fake returns one deterministic result per input lead.

That lets us verify count and list structure.

## New ML test blocks


### Reproducibility test

```python
first = generate_synthetic_leads(rows=500, seed=42)
second = generate_synthetic_leads(rows=500, seed=42)
assert first.equals(second)
```

This verifies the educational reproducibility promise.

### Threshold policy test

Creates tiny known arrays:

```python
y_true = [0, 0, 0, 1, 1, 1]
probabilities = [0.1, 0.3, 0.55, 0.35, 0.6, 0.9]
```

Then verifies different threshold strategies resolve correctly.

### Training artifact test

Trains on a temporary dataset and requires:

```text
model.joblib
metrics.json
model_manifest.json
confusion_matrix.png
calibration_curve.png
threshold_analysis.png
```

`tmp_path` means pytest gives the test an isolated temporary directory.

### Hash test

```python
assert manifest_hash == sha256_file(model_path)
```

Verifies training wrote the correct model fingerprint.

### Real service test

This test actually:

```text
generates data
trains model
loads ModelService
scores a lead
```

That is an integration-style test across the offline artifact and online service boundary.

### Tampering test

```python
with model_path.open("ab") as file:
    file.write(b"tampered")
```

`ab` means append binary.

The file changes.

The manifest hash does not.

Expected:

```text
MODEL_ARTIFACT_HASH_MISMATCH
```

Importantly, the service rejects the mismatch before `joblib.load`.

### Batch test

Scores two identical leads and checks two outputs.

This verifies the vectorized path returns one result per input.

### CSV test

A fake `BatchPredictor` is passed into `score_csv`.

This tests the CLI transformation and output columns without loading a real model.

That is why the `Protocol` was useful.

# 9. Artifact files


## `artifacts/metrics.json`

This is machine-readable evaluation evidence.

Important top-level fields:

```text
selected_model     = logistic_default
selection_metric   = brier_score
selection_split    = validation
threshold          = 0.24
```

### Why `selection_split` is explicit

A reviewer should be able to see:

```text
model decisions were made on validation
```

not assume the test set was used.

## `artifacts/model_manifest.json`

Current schema version:

```text
1.0
```

Current model version:

```text
2.0.0
```

Current feature count:

```text
10
```

### Model SHA-256

```text
1282a78f656b6ceb69ef4733d53f1246592bbe6efc1babd39897e377f0ffd0a1
```

### Dataset SHA-256

```text
552fb426b2276ef16903c9f6b99198aeac38be1a2e76248c103a7d8a03930cec
```

### `feature_count`

```text
10
```

This is redundant with `len(feature_contract)` on purpose. It gives reviewers and validation code a quick explicit contract summary.

### `library_versions`

The manifest records:

```text
Python
scikit-learn
pandas
NumPy
joblib
```

`ModelService` checks the ML runtime libraries before `joblib.load`. A version mismatch becomes a controlled `MODEL_RUNTIME_VERSION_MISMATCH` error instead of silently attempting to load an artifact under a different ML stack. Python is recorded for provenance; the runtime compatibility guard focuses on the libraries that directly define or serialize the fitted pipeline.

Do not memorize hashes.

Understand why they exist.

## PNGs

### `confusion_matrix.png`

Shows class decision errors at selected threshold.

### `calibration_curve.png`

Shows predicted probability vs observed positive frequency.

### `threshold_analysis.png`

Shows how precision, recall, F1, and normalized business cost change as threshold moves.

# 10. Calibration: read the actual result honestly


A dangerous portfolio mistake is:

```text
I generated a calibration curve
therefore my model is calibrated
```

Wrong.

The curve is a diagnostic.

Look at the current calibration bins in `metrics.json`.

Bin 1:

```text
mean predicted ≈ 0.054
observed positive ≈ 0.050
```

Bin 2:

```text
mean predicted ≈ 0.092
observed positive ≈ 0.060
```

Bin 3:

```text
mean predicted ≈ 0.125
observed positive ≈ 0.220
```

Bin 4:

```text
mean predicted ≈ 0.158
observed positive ≈ 0.200
```

Bin 5:

```text
mean predicted ≈ 0.199
observed positive ≈ 0.190
```

Bin 6:

```text
mean predicted ≈ 0.239
observed positive ≈ 0.220
```

Bin 7:

```text
mean predicted ≈ 0.292
observed positive ≈ 0.300
```

Bin 8:

```text
mean predicted ≈ 0.365
observed positive ≈ 0.390
```

Bin 9:

```text
mean predicted ≈ 0.468
observed positive ≈ 0.430
```

Bin 10:

```text
mean predicted ≈ 0.676
observed positive ≈ 0.610
```


The current held-out test result:

```text
Brier score = 0.169
log loss    = 0.512
```

Some high-probability bins overpredict observed frequency.

Therefore say in an interview:

> "I evaluate probability reliability with a calibration curve, Brier score, and log loss. The synthetic model is not perfectly calibrated. I chose the better probability candidate, but I do not claim deployment-grade calibration. On real CRM data I would validate reliability and fit a calibration layer using independent or cross-validated calibration data if necessary."

That is a strong answer because it is accurate.

# 11. Common bugs and exact debugging locations


## Bug: `/ready` returns 503

Check:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_manifest.json
```

Then run:

```powershell
python -m ml.train
```

## Bug: `MODEL_ARTIFACT_HASH_MISMATCH`

Check:

```text
Was model.joblib replaced?
Was model.joblib regenerated without regenerating manifest?
Did files come from different training runs?
```

Correct fix:

```powershell
python -m ml.train
```

Do not manually edit the hash to silence the error.

## Bug: CSV missing required columns

Check:

```text
ml/features.py → FEATURE_COLUMNS
```

Then compare CSV headers.

## Bug: API returns 422

Check:

```text
app/schemas/lead.py
```

Likely causes:

```text
unknown category
negative range
wrong type
batch > 1000
empty batch
```

## Bug: model trains but API fails loading

Check:

```text
artifact dependency versions
model_manifest.json
requirements.txt
```

Scikit-learn model persistence is environment-sensitive.

## Bug: feature contribution names look wrong

Check:

```text
preprocessor.get_feature_names_out()
ModelService._humanize_feature_name()
```

## Bug: prediction label changed after threshold strategy change

Expected possibility.

Check:

```text
metrics.json → threshold_analysis
model_manifest.json → threshold
/model/info → threshold
```

Probability can remain similar while label changes because threshold changed.

## Bug: CI train-smoke fails but local training works

Compare:

```text
Python version
requirements files
case-sensitive paths
uncommitted dataset changes
```

GitHub Ubuntu is case-sensitive.

Windows often hides case mistakes.

## Bug: release workflow does not run

The trigger is a pushed semantic-like tag:

```powershell
git tag v2.0.0
git push origin v2.0.0
```

A normal commit to `main` does not trigger `release.yml`.

## Bug: GHCR image not published

`publish-ghcr.yml` triggers on a pushed semantic-version tag:

```text
v*.*.*
```

Check that the tag exists on GitHub, the workflow ran for that tag, and the workflow has `packages: write`.

# 11.1 Why direct dependency versions are pinned


The requirements files pin the direct dependencies used for this benchmark.

Example:

```text
scikit-learn==1.8.0
numpy==2.3.5
pandas==2.2.3
joblib==1.5.3
```

Why is this new compared with the first starter kit?

Project 2 persists a fitted Python ML object. The exact library environment matters much more than it did for a thin HTTP/Gemini wrapper.

The policy is:

```text
validated versions stay pinned
        ↓
Dependabot proposes an upgrade PR
        ↓
CI retrains + tests + builds Docker
        ↓
human reviews metric/artifact changes
```

This prevents a normal `pip install` from silently changing scikit-learn and invalidating the benchmark or persisted model assumptions.

# 12. Exact local workflow


From a clean clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Generate data:

```powershell
python -m ml.generate_data --rows 5000 --seed 42
```

Train:

```powershell
python -m ml.train
```

Quality:

```powershell
ruff check .
ruff format --check .
```

Tests with CI-equivalent coverage:

```powershell
pytest `
  --cov=app `
  --cov=common `
  --cov=ml `
  --cov=scripts `
  --cov-report=term-missing `
  --cov-report=xml `
  --cov-fail-under=80
```

Run API:

```powershell
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Score CSV:

```powershell
python -m scripts.score_csv `
  --input data/leads.csv `
  --output scored_leads.csv
```

Docker:

```powershell
docker build -t lead-scoring-ml-api .
docker run --env-file .env -p 8000:8000 lead-scoring-ml-api
```

Compose:

```powershell
docker compose up --build
```

# 13. Git and release workflow


## Development PR

```powershell
git checkout -b feature/project-2-complete
git add .
git commit -m "Build production-oriented lead scoring ML API"
git push -u origin feature/project-2-complete
```

Create PR:

```text
feature/project-2-complete → main
```

Wait for:

```text
CI green
CodeQL green / no blocking issue
```

Merge.

Update local main:

```powershell
git checkout main
git pull origin main
```

## Create release

After main is clean:

```powershell
git tag v2.0.0
git push origin v2.0.0
```

Automation starts two independent workflows from the same pushed tag:

```text
                 ┌─→ retrain → GitHub Release
version tag ─────┤             model + manifest + metrics + plots
                 │
                 └─→ Docker build → GHCR image
```

The two workflows do not depend on one workflow-created event triggering the other.

# 14. Interview questions you must be able to answer

## Q1. Why not use accuracy alone?

**Answer:** Because the target is imbalanced and the business cost of false negatives and false positives differs. Accuracy can hide a model that mostly predicts the majority class.

## Q2. Why three data splits?

**Answer:** Training fits coefficients, validation selects the candidate and threshold, and test provides an untouched final evaluation.

## Q3. Why select by Brier score?

**Answer:** The API exposes probabilities, so candidate probability quality matters. Threshold policy is handled separately.

## Q4. Why did balanced logistic regression have better F1 but lose selection?

**Answer:** Class weighting improved positive recovery at threshold 0.50 but produced worse validation Brier score and log loss. We selected the probability model, then lowered the operational threshold.

## Q5. What is threshold tuning?

**Answer:** Choosing the probability cutoff that converts a continuous probability into a business action.

## Q6. Why is 0.50 not sacred?

**Answer:** It is the default binary decision cutoff, not a universal business optimum.

## Q7. What is Brier score?

**Answer:** Mean squared error between predicted probability and binary outcome; lower is better.

## Q8. What is calibration?

**Answer:** Agreement between predicted probability levels and observed event frequency.

## Q9. Are your probabilities calibrated?

**Answer:** Not perfectly. I diagnose reliability with calibration bins/curve, Brier score, and log loss and explicitly document remaining error.

## Q10. How are local explanations computed?

**Answer:** After preprocessing, each transformed feature value is multiplied by its fitted logistic coefficient. The largest positive and negative log-odds terms are returned.

## Q11. Are those causal effects?

**Answer:** No. They are local linear-model contributions and can reflect correlation or data-generation assumptions.

## Q12. Why use Pipeline?

**Answer:** It packages fitted preprocessing and the classifier together so training and inference apply the same transformations.

## Q13. Why use ColumnTransformer?

**Answer:** Numeric and categorical columns need different preprocessing.

## Q14. What does class_weight="balanced" do conceptually?

**Answer:** It changes class weighting during fitting so minority-class errors receive greater influence.

## Q15. Why vectorized batch inference?

**Answer:** One DataFrame and one predict_proba call reduce Python-level per-row model calls and keep outputs aligned.

## Q16. Why a model manifest?

**Answer:** To persist model/schema identity, feature order, dependency versions, threshold policy, and artifact hashes.

## Q17. Does SHA-256 make joblib safe?

**Answer:** No. It detects mismatch relative to the manifest. It does not authenticate an attacker-controlled manifest. Only trusted artifacts should be loaded.

## Q18. Why not train inside an API endpoint?

**Answer:** Training is expensive, state-changing, and a separate offline lifecycle. Inference should load a versioned artifact.

## Q19. Why keep model.joblib out of Git?

**Answer:** It is generated, environment-sensitive binary state. Normal development reproduces it through training; tagged releases may distribute the exact trusted binary together with its SHA-256 manifest.

## Q20. What would change for real CRM data?

**Answer:** Time-aware validation, missing-value policy, leakage checks, data/privacy review, calibration validation, business-defined costs, drift monitoring, retraining/version governance.

# 15. Scope boundaries


This project intentionally does **not** add:

```text
Kubernetes
Kafka
Celery
Redis
feature store
MLflow
Kubeflow
SHAP
LLM agents
vector database
```

Not because those technologies are bad.

Because Project 2 has one learning objective:

> Understand and ship a supervised ML model lifecycle with honest evaluation and serving.

Adding every infrastructure tool would hide the core learning signal.

## Why no SHAP?

The selected estimator is linear.

We can directly inspect:

```text
transformed value × coefficient
```

This gives a precise model-specific local log-odds decomposition.

SHAP is valuable for many models and explanation requirements, but adding it here just for a keyword would be unnecessary complexity.

## Why no calibrated classifier wrapper yet?

The calibration curve is intentionally diagnostic.

The current synthetic benchmark shows probability reliability is not perfect.

A real deployment should decide whether to calibrate using representative data and an independent/cross-validated calibration procedure.

Adding a calibrator just to make the curve look better on synthetic data would be less honest than documenting the limitation.

# 16. Final mental model


Memorize this architecture, not the syntax:

```text
DATA
  ↓
feature contract
  ↓
TRAIN SPLIT
  ↓
fit candidate models
  ↓
VALIDATION SPLIT
  ↓
select probability model
  ↓
select business threshold
  ↓
refit chosen model specification
  ↓
TEST SPLIT
  ↓
honest final evaluation
  ↓
model + metrics + manifest + plots
  ↓
verify artifact
  ↓
load once
  ↓
vectorized predict_proba
  ↓
probability
  ↓
threshold → business label
  ↓
linear contributions → explanation
  ↓
API / batch / CSV
  ↓
CI / release / GHCR
```

## One-sentence project explanation

> I built a reproducible lead-scoring ML system that separates probability model selection from business threshold policy, evaluates calibration and classification trade-offs on proper train/validation/test splits, serves vectorized FastAPI inference with transparent logistic log-odds contributions, validates artifact hashes before loading, supports CSV batch scoring, and ships through CI, CodeQL, Docker, GitHub Releases, and GHCR.

## What you should understand before moving to Project 3

You are ready to move on when you can explain, without reading code:

1. why train, validation, and test are separate
2. why a probability is not a decision
3. why 0.50 is not a universal threshold
4. why default logistic was selected over balanced logistic
5. what Brier score measures
6. what a calibration curve compares
7. what precision and recall mean in this lead context
8. how the business-cost formula chooses a threshold
9. what `Pipeline` protects you from
10. how one-hot encoding works
11. how `predict_proba()[:, 1]` works
12. how local log-odds contributions are calculated
13. why contributions are not causal
14. why the manifest stores feature order and hashes
15. why SHA-256 is integrity checking, not publisher authentication
16. why `predict_many` is vectorized
17. why training is offline and inference is online
18. how a version tag becomes a release and GHCR image

Read this file alongside the source. Run the commands. Break one assumption at a time and observe the failure. That is the learning workflow for Project 2.
