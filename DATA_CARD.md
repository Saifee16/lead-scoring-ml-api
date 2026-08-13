# Data Card: Synthetic B2B Lead Dataset

## Dataset summary

| Field | Value |
|---|---|
| File | `data/leads.csv` |
| Rows | 5000 |
| Generator | `ml/generate_data.py` |
| Random seed | `42` |
| Positive target rate | `26.70%` |
| Target | `converted` |
| License | Repository MIT license |
| Data type | Fully synthetic |

## Purpose

The dataset exists to teach and demonstrate:

- binary classification
- mixed numeric/categorical preprocessing
- class imbalance
- probability evaluation
- threshold tuning
- API inference
- batch scoring
- artifact lifecycle

It is not a copy, scrape, or anonymization of real customer or CRM data.

## Reproduction

```bash
python -m ml.generate_data --rows 5000 --seed 42
```

The generator uses NumPy's `default_rng(seed)` so the same row count and seed produce the same deterministic dataset under the tested dependency environment.

The generated dataset SHA-256 is recorded in `artifacts/model_manifest.json`.

## Feature dictionary

| Feature | Type | API range/domain | Meaning |
|---|---|---|---|
| `company_size` | integer | 1–1000 | Synthetic employee-count proxy |
| `annual_revenue_k` | float | 0–50,000 | Synthetic annual revenue in thousands |
| `website_visits_30d` | integer | 0–500 | Website visits in prior 30 days |
| `email_opens_30d` | integer | 0–200 | Email opens in prior 30 days |
| `demo_requests_30d` | integer | 0–20 | Demo requests in prior 30 days |
| `form_submissions_30d` | integer | 0–50 | Form submissions in prior 30 days |
| `days_since_last_contact` | integer | 0–365 | Recency of last contact |
| `lead_source` | categorical | 5 values | Synthetic acquisition source |
| `industry` | categorical | 5 values | Synthetic business vertical |
| `decision_maker_engaged` | boolean | true/false | Whether a decision-maker is synthetically engaged |
| `converted` | binary target | 0/1 | Synthetic conversion outcome |

## Categorical domains

### Lead source

```text
organic
paid_search
referral
social
outbound
```

### Industry

```text
saas
ecommerce
healthcare
finance
services
```

## Generation distributions

The generator intentionally mixes distribution families so the dataset is not just uniform random noise.

### Company size

Generated from a log-normal distribution, rounded, and clipped.

Why:

A long-tailed distribution is a simple synthetic approximation where many companies are smaller and a smaller number are much larger.

### Annual revenue

Generated from a log-normal distribution and clipped.

### Engagement counts

Website visits, email opens, demo requests, and form submissions use Poisson draws.

Why:

Poisson distributions are commonly used as simple count-data generators. This is a synthetic design choice, not a claim that real lead engagement exactly follows a Poisson process.

### Contact recency

Generated from an exponential distribution, rounded, and clipped.

### Decision-maker engagement

Generated from a Bernoulli/binomial draw with a configured probability.

## Synthetic target construction

The generator creates a latent score from designed feature effects:

```text
base intercept
+ website activity effect
+ email engagement effect
+ demo interest effect
+ form submission effect
- stale contact effect
+ decision-maker effect
+ small revenue effect
+ small company-size effect
+ source effect
+ industry effect
+ random noise
```

Then:

```text
latent score
    ↓ sigmoid
synthetic probability
    ↓ Bernoulli draw
converted = 0 or 1
```

This creates a learnable but noisy binary-classification problem.

## Important bias warning

The feature effects are designed by the project author.

For example, the generator gives a positive synthetic effect to referral leads and a negative synthetic effect to outbound leads.

That means the trained model can learn those authored assumptions.

This is intentional for educational reproducibility, but it also demonstrates an important ML lesson:

> A model can learn the assumptions and biases embedded in its training data.

Do not interpret learned coefficients as universal truths about industries or acquisition channels.

## Missing data

The current generator emits complete rows.

Therefore the current pipeline does not include imputation.

A real CRM dataset will likely contain:

- missing revenue
- unknown company size
- incomplete engagement tracking
- inconsistent source names
- stale duplicates
- delayed conversion labels

A reusable real deployment must define explicit data-quality and missing-value policies before retraining.

## Leakage considerations

The current features are designed to be conceptually available before the `converted` target.

For real data, verify that each feature was available at prediction time.

Example leakage:

```text
feature = payment_completed_at
prediction target = converted
```

A model using a post-conversion field would appear excellent but be operationally useless.

## Temporal limitations

This synthetic dataset has no timestamps for lead creation or conversion.

Therefore the project uses random stratified train/validation/test splits.

For real longitudinal CRM data, prefer a temporal validation design when deployment predicts future leads from past leads.

## Privacy

The dataset contains no real names, emails, phone numbers, company identifiers, or personal customer records.

## Recommended real-data replacement process

1. Create a data dictionary.
2. Define conversion outcome and observation window.
3. Remove direct identifiers not needed for modeling.
4. Verify feature availability time.
5. Audit missingness and duplicates.
6. Use a time-aware split where appropriate.
7. Compare class balance across splits.
8. Retrain and regenerate all artifacts.
9. Update this data card.
10. Review privacy and fairness implications.
