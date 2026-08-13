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
