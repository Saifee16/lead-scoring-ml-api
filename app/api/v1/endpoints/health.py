from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.schemas.common import SuccessResponse
from app.schemas.health import HealthData, ReadinessData

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=SuccessResponse[HealthData])
def health_check() -> SuccessResponse[HealthData]:
    return SuccessResponse(
        data=HealthData(
            app=settings.app_name,
            environment=settings.app_env,
        )
    )


@router.get("/ready", response_model=SuccessResponse[ReadinessData])
def readiness_check(response: Response) -> SuccessResponse[ReadinessData]:
    model_exists = settings.model_path.exists()
    metrics_exists = settings.metrics_path.exists()
    manifest_exists = settings.manifest_path.exists()
    ready = model_exists and metrics_exists and manifest_exists

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return SuccessResponse(
        data=ReadinessData(
            ready=ready,
            model_artifact_exists=model_exists,
            metrics_artifact_exists=metrics_exists,
            manifest_artifact_exists=manifest_exists,
        )
    )
