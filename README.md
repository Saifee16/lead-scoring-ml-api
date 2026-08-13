# Lead Scoring ML API

[![CI](https://github.com/Saifee16/lead-scoring-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Saifee16/lead-scoring-ml-api/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Saifee16/lead-scoring-ml-api/actions/workflows/codeql.yml/badge.svg)](https://github.com/Saifee16/lead-scoring-ml-api/actions/workflows/codeql.yml)

A reusable FastAPI ML inference service that demonstrates the full supervised-learning lifecycle: reproducible data generation, leakage-aware train/validation/test separation, preprocessing pipelines, probability-quality model selection, business-aware threshold tuning, calibration diagnostics, local logistic-regression contributions, artifact integrity verification, batch scoring, CSV scoring, Docker, CI, releases, and GHCR publishing.

> **Portfolio scope:** the committed dataset is synthetic. This repository demonstrates ML engineering architecture and evaluation discipline. Its metrics are not claims about real CRM performance.

## Why this repository is different

Most lead-scoring demos stop at a notebook and one accuracy number.

This project deliberately separates three concerns:

```text
probability model
      ↓
probability diagnostics
      ↓
decision policy / threshold
```

The model candidate is selected on the **validation split by lower Brier score** because the API exposes `conversion_probability`.

The qualification threshold is selected separately using a configurable business-cost policy where a false negative costs more than a false positive.

The held-out test set is used only after model and threshold decisions are made.

## Standout features

- Deterministic 5,000-row synthetic B2B lead dataset
- Explicit feature contract shared by training and CSV scoring
- Train / validation / test lifecycle
- `Pipeline` + `ColumnTransformer`
- Numeric standardization and categorical one-hot encoding
- Default and class-balanced logistic-regression candidates
- Candidate selection by validation Brier score
- Calibration curve, Brier score, and log loss
- F1-optimal and business-cost threshold analysis
- Threshold policy separated from probability estimation
- Held-out test metrics and confusion matrix
- True local logistic-regression log-odds contributions
- Clear warning that contributions are not causal effects
- Single-lead and vectorized batch API inference
- CSV scoring CLI for CRM-style exports
- Model manifest with feature contract and SHA-256 hashes
- SHA-256 verification before `joblib.load`
- Typed FastAPI schemas and Swagger examples
- Health, readiness, model info, metrics, and manifest routes
- 26 automated tests
- 80% minimum CI coverage gate
- Multi-stage Docker training/runtime image
- CodeQL workflow
- GitHub release workflow with the trusted model and evaluation assets
- GHCR container publishing directly from semantic version tags
- `MODEL_CARD.md`, `DATA_CARD.md`, and deep `NOTES.md`
- Validated direct dependency versions pinned; Dependabot proposes upgrades
- Runtime ML-library compatibility check against the model manifest

## Reproducibility policy

The direct dependency versions in the three requirements files are pinned to the versions used for the validated benchmark. This matters because persisted scikit-learn artifacts and even evaluation behavior can depend on library versions. Dependabot remains configured so dependency upgrades arrive as reviewable pull requests rather than silently changing the environment.

## Actual deterministic benchmark

Dataset seed: `42`

Positive conversion rate: `26.70%`

Data split:

| Split | Rows | Purpose |
|---|---:|---|
| Training | 3200 | Fit candidate models |
| Validation | 800 | Select model and tune threshold |
| Test | 1000 | Final untouched evaluation |

### Candidate comparison on validation data

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `logistic_default` | 0.766 | 0.608 | 0.355 | 0.448 | 0.782 | 0.157 | 0.480 |
| `logistic_balanced` | 0.715 | 0.479 | 0.729 | 0.578 | 0.783 | 0.196 | 0.579 |

Selected model:

```text
logistic_default
```

Selection metric:

```text
minimize validation brier_score
```

The balanced model has higher F1 at the default `0.50` threshold, but its probability estimates have materially worse Brier score and log loss on this validation split. Because the API publishes a probability, this project selects the probability model first and handles recall/precision trade-offs in the threshold policy.

### Threshold policy on validation data

Assumed relative costs:

```text
false positive cost = 1
false negative cost = 5
```

These are **relative cost units**, not dollars.

| Strategy | Threshold | Precision | Recall | F1 | FP | FN | Relative cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| `default` | 0.50 | 0.608 | 0.355 | 0.448 | 49 | 138 | 739 |
| `f1` | 0.26 | 0.479 | 0.757 | 0.587 | 176 | 52 | 436 |
| `business_cost` | 0.24 | 0.458 | 0.780 | 0.577 | 198 | 47 | 433 |

Selected threshold:

```text
0.24 using business_cost strategy
```

The key design principle is:

> The model estimates a probability. The threshold converts that probability into a business action.

### Final held-out test performance

| Metric | Value |
|---|---:|
| Accuracy | 0.652 |
| Precision | 0.409 |
| Recall | 0.685 |
| F1 | 0.513 |
| ROC-AUC | 0.729 |
| Brier score | 0.169 |
| Log loss | 0.512 |

Confusion matrix at threshold `0.24`:

```text
True negative:  469
False positive: 264
False negative: 84
True positive:  183
```

The lower threshold intentionally increases recall because the configured policy says missed converters are five times as costly as unnecessary sales review.

## Evaluation assets

Generated by `python -m ml.train`:

```text
artifacts/confusion_matrix.png
artifacts/calibration_curve.png
artifacts/threshold_analysis.png
artifacts/metrics.json
artifacts/model_manifest.json
artifacts/model.joblib
```

### Calibration warning

The model's held-out Brier score is `0.169` and log loss is `0.512`.

The calibration curve is a **diagnostic**, not a claim that the probabilities are perfectly calibrated. The synthetic benchmark still shows reliability error in some probability ranges. A real production deployment should validate calibration on organization-specific historical data and may add a separate calibration stage if needed.

## Architecture

### Offline training

```text
data/leads.csv
      |
      v
validate feature contract
      |
      v
train / validation / test split
      |
      v
candidate pipelines
      |
      +--> logistic_default
      |
      +--> logistic_balanced
      |
      v
validation probability metrics
      |
      v
select lower Brier score
      |
      v
validation threshold analysis
      |
      +--> default threshold
      +--> F1-optimal threshold
      +--> business-cost threshold
      |
      v
refit selected pipeline on train + validation
      |
      v
held-out test evaluation
      |
      +--> confusion matrix
      +--> calibration curve
      +--> threshold analysis plot
      |
      v
model.joblib + metrics.json + model_manifest.json
```

### Online inference

```text
HTTP request / CSV row
        |
        v
Pydantic LeadFeatures
        |
        v
ModelService
        |
        +--> require model + metrics + manifest
        |
        +--> verify model SHA-256
        |
        +--> joblib.load trusted artifact
        |
        +--> DataFrame with manifest feature order
        |
        +--> pipeline.predict_proba(...)
        |
        +--> transformed feature values × coefficients
        |
        +--> top local log-odds contributions
        |
        +--> apply selected business threshold
        |
        v
LeadScoreData
        |
        v
typed JSON response
```

## Project structure

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
│   └── model.joblib          # generated; ignored by Git
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

Empty `__init__.py` files explicitly mark normal Python packages. They are intentionally empty.

## Quick start

### 1. Clone

```bash
git clone https://github.com/Saifee16/lead-scoring-ml-api.git
cd lead-scoring-ml-api
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install development dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Create local configuration

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 5. Reproduce the dataset

```bash
python -m ml.generate_data --rows 5000 --seed 42
```

### 6. Train and generate artifacts

```bash
python -m ml.train
```

Default threshold policy:

```text
business_cost
```

Alternative training policies:

```bash
python -m ml.train --threshold-strategy default
python -m ml.train --threshold-strategy f1
python -m ml.train --threshold-strategy business_cost
```

Change relative business costs:

```bash
python -m ml.train   --threshold-strategy business_cost   --false-positive-cost 1   --false-negative-cost 10
```

### 7. Run the API

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Process liveness |
| `GET` | `/api/v1/ready` | Required artifact presence |
| `GET` | `/api/v1/model/info` | Model and training metadata |
| `GET` | `/api/v1/model/metrics` | Test metrics, calibration, threshold policy |
| `GET` | `/api/v1/model/manifest` | Model hash and feature contract |
| `POST` | `/api/v1/predict-lead-score` | Score one lead |
| `POST` | `/api/v1/predict-lead-scores` | Vectorized batch scoring |

## Single prediction

```http
POST /api/v1/predict-lead-score
```

Request:

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

Example result from the deterministic trained artifact:

```json
{
  "success": true,
  "data": {
    "prediction": "qualified",
    "lead_score": 97.44,
    "conversion_probability": 0.9744,
    "confidence": 0.9663,
    "confidence_level": "high",
    "threshold": 0.24,
    "top_positive_factors": [
      {
        "feature": "demo_requests_30d",
        "log_odds_contribution": 1.1483,
        "direction": "positive"
      },
      {
        "feature": "lead_source=referral",
        "log_odds_contribution": 0.5977,
        "direction": "positive"
      },
      {
        "feature": "website_visits_30d",
        "log_odds_contribution": 0.5647,
        "direction": "positive"
      }
    ],
    "top_negative_factors": [],
    "explanation": "The model estimates a high conversion probability...",
    "model_version": "2.0.0"
  }
}
```

### What the factor contributions mean

For logistic regression:

```text
log_odds = intercept + Σ(transformed_feature × coefficient)
```

The API returns the largest per-row terms from:

```text
transformed_feature × coefficient
```

They are local contributions to model **log-odds**.

They are not causal effects and they are not SHAP values.

## Batch prediction

```http
POST /api/v1/predict-lead-scores
```

Request:

```json
{
  "leads": [
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
    },
    {
      "lead_source": "outbound",
      "industry": "services",
      "company_size": 18,
      "annual_revenue_k": 120,
      "website_visits_30d": 1,
      "email_opens_30d": 0,
      "demo_requests_30d": 0,
      "form_submissions_30d": 0,
      "days_since_last_contact": 80,
      "decision_maker_engaged": false
    }
  ]
}
```

The service creates one DataFrame and calls `predict_proba` once for the batch instead of looping through one model call per lead.

Maximum API batch size defaults to `1000` and can be configured with `MAX_BATCH_SIZE`.

## Score a CSV

Input CSV must contain the ten feature columns defined in `ml/features.py`.

```bash
python -m scripts.score_csv   --input data/leads.csv   --output scored_leads.csv
```

The output keeps the original columns and adds:

```text
prediction
lead_score
conversion_probability
confidence
confidence_level
threshold
top_positive_factors
top_negative_factors
model_version
```

## Model manifest and integrity

`artifacts/model_manifest.json` stores:

```text
schema version
model version
algorithm
selected candidate
training timestamp
model SHA-256
dataset SHA-256
feature contract
target
selected threshold
threshold strategy
library versions
```

`ModelService` calculates the SHA-256 of `model.joblib` and compares it with the manifest **before** `joblib.load`.

Important limitation:

> A hash detects mismatch relative to the manifest. It is not a digital signature and does not prove the manifest itself came from a trusted publisher.

Only load model artifacts from a trusted source.

## Tests and quality

```bash
ruff check .
ruff format --check .
pytest
```

CI coverage command:

```bash
pytest   --cov=app   --cov=common   --cov=ml   --cov=scripts   --cov-report=term-missing   --cov-report=xml   --cov-fail-under=80
```

Current validation suite:

```text
26 tests
```

Tests cover:

- deterministic data generation
- input row minimum
- threshold-policy analysis
- artifact generation
- manifest model hash
- dataset hash
- missing dataset
- invalid business costs
- model info route
- metrics route
- manifest route
- real trained-model inference
- local contribution output
- vectorized batch inference
- model hash mismatch rejection
- missing model handling
- threshold-aware confidence
- single prediction API
- batch prediction API
- batch size validation
- schema validation
- CSV scoring
- CSV feature-contract validation

## Docker

Build:

```bash
docker build -t lead-scoring-ml-api .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 lead-scoring-ml-api
```

The Dockerfile uses two stages:

```text
trainer
   ↓ generates artifacts
runtime
   ↓ copies only app/common/runtime dependencies + generated artifacts
```

The runtime image does not need Matplotlib.

The runtime stage sets `APP_ENV=production`, disables FastAPI debug mode by default,
and runs Uvicorn as an unprivileged `app` user.

Docker Compose:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

## GitHub automation

### CI

`.github/workflows/ci.yml`

Runs:

```text
Ruff lint
Ruff format check
26-test suite
80% coverage floor
training smoke test
six-artifact verification
Docker build
```

### CodeQL

`.github/workflows/codeql.yml`

Runs Python CodeQL analysis on pushes, pull requests, and a weekly schedule.

### Release

Push a semantic version tag:

```bash
git tag v2.0.0
git push origin v2.0.0
```

The release workflow retrains from the tagged repository state and creates a GitHub Release with the exact trusted model referenced by its manifest plus evaluation evidence:

```text
model.joblib
metrics.json
model_manifest.json
confusion_matrix.png
calibration_curve.png
threshold_analysis.png
```

The release uses generated release notes.

### GHCR

The same pushed semantic-version tag independently triggers `.github/workflows/publish-ghcr.yml`, which builds the Dockerfile and publishes:

```text
ghcr.io/saifee16/lead-scoring-ml-api:<tag>
ghcr.io/saifee16/lead-scoring-ml-api:latest
```

Example after tag `v2.0.0` is pushed and the package workflow succeeds:

```bash
docker pull ghcr.io/saifee16/lead-scoring-ml-api:v2.0.0
```

## Documentation

- [`NOTES.md`](NOTES.md) — deep code and ML learning manual
- [`MODEL_CARD.md`](MODEL_CARD.md) — model behavior, evaluation, intended use, limitations
- [`DATA_CARD.md`](DATA_CARD.md) — dataset contract, generation logic, assumptions, bias
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow
- [`SECURITY.md`](SECURITY.md) — artifact and secret security policy

## Reusing this repository

For a real company:

1. Replace the synthetic dataset with validated historical lead outcomes.
2. Update the feature contract.
3. Add data quality and missing-value policy.
4. Retrain candidates.
5. Select probability model using an appropriate validation metric.
6. Validate calibration.
7. Define false-positive and false-negative business costs with stakeholders.
8. Tune the decision threshold on validation data.
9. Evaluate once on untouched test data.
10. Version the model and schema.
11. Publish trusted artifacts or a container.
12. Monitor production input drift, outcome drift, and calibration.

Do not reuse the synthetic model as a real sales decision system.

## License

MIT. See `LICENSE`.
