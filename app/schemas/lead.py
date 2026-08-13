from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LeadSource = Literal["organic", "paid_search", "referral", "social", "outbound"]
Industry = Literal["saas", "ecommerce", "healthcare", "finance", "services"]
PredictionLabel = Literal["qualified", "not_qualified"]
ConfidenceLevel = Literal["low", "medium", "high"]
ContributionDirection = Literal["positive", "negative"]


class LeadFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
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
        },
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
    model_config = ConfigDict(extra="forbid")

    leads: list[LeadFeatures] = Field(min_length=1)


class BatchLeadScoreData(BaseModel):
    count: int = Field(ge=1)
    predictions: list[LeadScoreData]
