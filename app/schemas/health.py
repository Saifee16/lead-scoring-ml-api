from typing import Literal

from pydantic import BaseModel


class HealthData(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    environment: str


class ReadinessData(BaseModel):
    ready: bool
    model_artifact_exists: bool
    metrics_artifact_exists: bool
    manifest_artifact_exists: bool
