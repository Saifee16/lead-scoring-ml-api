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
