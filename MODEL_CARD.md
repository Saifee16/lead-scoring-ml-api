# Model Card: Lead Conversion Scoring Model

## Model details

| Field | Value |
|---|---|
| Model version | `2.0.0` |
| Schema version | `1.0` |
| Algorithm | `LogisticRegression` |
| Selected candidate | `logistic_default` |
| Target | `converted` |
| Threshold strategy | `business_cost` |
| Selected threshold | `0.24` |
| Training data | Deterministic synthetic B2B leads |
| Dataset rows | 5000 |

## Intended use

This model is intended to demonstrate an end-to-end supervised ML engineering workflow for binary lead-conversion scoring.

Appropriate uses:

- portfolio demonstration
- ML lifecycle learning
- FastAPI model-serving examples
- threshold-policy demonstrations
- artifact integrity demonstrations
- batch inference examples

## Out-of-scope use

Do not use the included trained model to:

- make real sales prioritization decisions
- forecast company revenue
- deny access to products or services
- make employment, lending, insurance, healthcare, or other high-impact decisions
- claim real-world conversion uplift

The dataset is synthetic and encodes relationships designed by the repository author.

## Target

```text
converted
```

Binary values:

```text
0 = did not convert
1 = converted
```

## Input feature contract

The persisted artifact expects exactly the ordered feature contract in `model_manifest.json`:

- `company_size`
- `annual_revenue_k`
- `website_visits_30d`
- `email_opens_30d`
- `demo_requests_30d`
- `form_submissions_30d`
- `days_since_last_contact`
- `lead_source`
- `industry`
- `decision_maker_engaged`


## Preprocessing

Numeric columns use `StandardScaler`.

Categorical columns use `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

The preprocessor and classifier are stored inside one scikit-learn `Pipeline` so inference applies the same transformations learned during training.

## Candidate models

Two logistic-regression candidates are compared:

```text
logistic_default   → class_weight=None
logistic_balanced  → class_weight="balanced"
```

Validation candidate results:

| Candidate | F1 | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| `logistic_default` | 0.448 | 0.782 | 0.157 | 0.480 |
| `logistic_balanced` | 0.578 | 0.783 | 0.196 | 0.579 |

## Model selection policy

The selected candidate minimizes validation Brier score.

Why:

The API exposes `conversion_probability`. This project therefore prioritizes probability quality at candidate selection time instead of choosing the candidate with the best F1 at an arbitrary `0.50` threshold.

The balanced model has better F1 at `0.50` on validation data, but worse Brier score and log loss.

The threshold is optimized separately.

## Decision threshold policy

Relative error costs:

```text
false positive = 1 cost unit
false negative = 5 cost units
```

These are educational relative units, not dollars.

The validation threshold with minimum configured cost is `0.24`.

This policy intentionally favors recall because the educational scenario assumes missing a real converter is more expensive than asking sales to review a weak lead.

## Final held-out test evaluation

| Metric | Value |
|---|---:|
| Accuracy | 0.652 |
| Precision | 0.409 |
| Recall | 0.685 |
| F1 | 0.513 |
| ROC-AUC | 0.729 |
| Brier score | 0.169 |
| Log loss | 0.512 |

Confusion matrix:

| | Actual 0 | Actual 1 |
|---|---:|---:|
| Predicted 0 | 469 | 84 |
| Predicted 1 | 264 | 183 |

## Probability calibration

Artifacts:

```text
artifacts/calibration_curve.png
artifacts/metrics.json → calibration
```

The held-out test Brier score is `0.169`.

The reliability bins show that predicted probability and observed positive frequency are not perfectly aligned in every range.

Therefore:

> `conversion_probability` is a model estimate, not guaranteed real-world frequency.

A real deployment should validate probability calibration on representative historical data. If calibration is inadequate, a calibrator must be fit using data independent of the base model's fitting data or through an appropriate cross-validation calibration procedure.

## Local explanation method

The selected estimator is logistic regression.

For a transformed row:

```text
log_odds = intercept + Σ(x'_j × w_j)
```

The API computes:

```text
local contribution_j = transformed feature value_j × learned coefficient_j
```

It returns the top positive and negative contributions.

Interpretation:

- positive contribution increases model log-odds relative to the intercept and other terms
- negative contribution decreases model log-odds

Limitations:

- contributions are in log-odds space
- preprocessing changes numeric feature scale
- one-hot features represent category indicators
- correlated features can split or shift coefficients
- these are not causal effects
- these are not SHAP values
- the intercept is not returned as a factor

## Confidence field

The API confidence field measures normalized distance from the **selected decision threshold**:

```text
confidence = abs(probability - threshold) / max(threshold, 1 - threshold)
```

It is clipped to `[0, 1]`.

This means:

- `0` = exactly on the decision boundary
- larger value = farther from the operational threshold

It is not calibrated certainty.

## Artifact persistence

The pipeline is stored using `joblib`.

`model_manifest.json` stores:

- model SHA-256
- dataset SHA-256
- feature contract
- schema version
- model version
- threshold and threshold strategy
- dependency versions

The service verifies the model file hash before loading it.

Hash limitation:

A SHA-256 comparison detects mismatch against the manifest. It does not authenticate the manifest. Signed provenance or trusted distribution controls are required for stronger publisher authenticity.

## Known limitations

1. Synthetic data does not represent a real CRM population.
2. Feature-target relationships are intentionally designed.
3. No missing-value policy is implemented because the synthetic generator emits complete rows.
4. No temporal validation is possible because the synthetic dataset has no event timestamps.
5. No production drift monitoring exists.
6. No fairness claim is made.
7. The threshold costs are illustrative.
8. Probability calibration is diagnostic rather than guaranteed.
9. The feature contribution method is linear-model-specific.
10. The model should be retrained for organization-specific data.

## Fairness considerations

The current feature contract does not intentionally include protected demographic attributes.

That does not prove fairness.

Proxy variables, sampling bias, historical outcome bias, and business process bias can still create unequal outcomes in a real dataset.

A real deployment requires a separate fairness and impact review appropriate to the data and decision context.

## Retraining guidance

For a real dataset:

1. Define the target with business stakeholders.
2. Validate feature availability at prediction time.
3. Remove leakage features.
4. Use time-aware splits when lead outcomes are temporal.
5. Compare simple baselines.
6. Check probability quality.
7. Tune threshold on validation data.
8. Evaluate once on untouched test data.
9. Version artifacts and schemas.
10. Monitor production drift and observed outcomes.
