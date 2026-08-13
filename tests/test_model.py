import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_model_info(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["algorithm"] == "LogisticRegression"
    assert body["data"]["selected_model"] == "logistic_balanced"
    assert body["data"]["threshold_strategy"] == "business_cost"


@pytest.mark.anyio
async def test_model_metrics(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["selection_metric"] == "brier_score"
    assert body["data"]["selection_split"] == "validation"
    assert body["data"]["metrics"]["brier_score"] == 0.22
    assert body["data"]["threshold_analysis"]["selected_strategy"] == "business_cost"


@pytest.mark.anyio
async def test_model_manifest(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["schema_version"] == "1.0"
    assert len(body["data"]["model_sha256"]) == 64
