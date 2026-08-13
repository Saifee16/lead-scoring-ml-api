import pytest
from httpx import AsyncClient

VALID_LEAD = {
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


@pytest.mark.anyio
async def test_predict_lead_score(client: AsyncClient) -> None:
    response = await client.post("/api/v1/predict-lead-score", json=VALID_LEAD)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["prediction"] == "qualified"
    assert body["data"]["lead_score"] == 82.5
    assert body["data"]["conversion_probability"] == 0.825
    assert body["data"]["threshold"] == 0.38
    assert body["data"]["top_positive_factors"][0]["direction"] == "positive"
    assert body["data"]["model_version"] == "test-2.0.0"


@pytest.mark.anyio
async def test_predict_lead_scores_batch(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/predict-lead-scores",
        json={"leads": [VALID_LEAD, VALID_LEAD]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["count"] == 2
    assert len(body["data"]["predictions"]) == 2


@pytest.mark.anyio
async def test_batch_rejects_more_than_configured_limit(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/predict-lead-scores",
        json={"leads": [VALID_LEAD] * 1001},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "LEAD_BATCH_TOO_LARGE"


@pytest.mark.anyio
async def test_prediction_rejects_unknown_lead_source(client: AsyncClient) -> None:
    payload = {**VALID_LEAD, "lead_source": "telegram"}
    response = await client.post("/api/v1/predict-lead-score", json=payload)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_prediction_rejects_negative_company_size(client: AsyncClient) -> None:
    payload = {**VALID_LEAD, "company_size": -5}
    response = await client.post("/api/v1/predict-lead-score", json=payload)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_prediction_rejects_extra_feature(client: AsyncClient) -> None:
    payload = {**VALID_LEAD, "crm_notes": "not part of the model feature contract"}
    response = await client.post("/api/v1/predict-lead-score", json=payload)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_batch_rejects_extra_top_level_field(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/predict-lead-scores",
        json={"leads": [VALID_LEAD], "source": "crm-export"},
    )

    assert response.status_code == 422
