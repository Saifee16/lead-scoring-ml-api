# Contributing

Contributions are welcome when they preserve the project's core goal: a small, inspectable supervised-ML lifecycle project.

## Setup

```bash
git clone https://github.com/Saifee16/lead-scoring-ml-api.git
cd lead-scoring-ml-api
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

macOS/Linux:

```bash
source .venv/bin/activate
cp .env.example .env
```

Install:

```bash
pip install -r requirements-dev.txt
```

## Reproduce training

```bash
python -m ml.generate_data --rows 5000 --seed 42
python -m ml.train
```

## Required local checks

```bash
ruff check .
ruff format --check .
pytest   --cov=app   --cov=common   --cov=ml   --cov=scripts   --cov-report=term-missing   --cov-fail-under=80
```

For source changes, also build Docker:

```bash
docker build -t lead-scoring-ml-api .
```

## ML contribution rules

A model or feature change should explain:

- what problem it solves
- which split is used for the decision
- why the evaluation metric is appropriate
- whether probabilities or only ranking/class labels matter
- whether the feature exists at prediction time
- whether the change introduces leakage
- how threshold policy is affected
- whether `MODEL_CARD.md` or `DATA_CARD.md` must change

Do not select a model on the held-out test set.

Do not present synthetic metrics as real business performance.

## Artifact rules

`artifacts/model.joblib` is generated and ignored by Git.

Evaluation JSON and PNG artifacts are committed as reproducibility evidence.

When artifact structure changes:

1. update `SCHEMA_VERSION`
2. update manifest schemas
3. update `ModelService` validation
4. update tests
5. update `MODEL_CARD.md`
6. update `NOTES.md`

Never load an untrusted `joblib` artifact.

## Pull requests

Use a focused branch:

```bash
git checkout -b feature/my-change
```

A pull request should describe:

- summary
- reason
- ML/evaluation impact
- tests run
- generated artifact changes

CI and CodeQL must pass before merge.
