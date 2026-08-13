from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.common import SuccessResponse
from app.schemas.model import MetricsData, ModelInfoData, ModelManifestData
from app.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/model")
ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]


@router.get("/info", response_model=SuccessResponse[ModelInfoData])
def model_info(service: ModelServiceDependency) -> SuccessResponse[ModelInfoData]:
    return SuccessResponse(data=service.get_model_info())


@router.get("/metrics", response_model=SuccessResponse[MetricsData])
def model_metrics(service: ModelServiceDependency) -> SuccessResponse[MetricsData]:
    return SuccessResponse(data=service.get_metrics())


@router.get("/manifest", response_model=SuccessResponse[ModelManifestData])
def model_manifest(service: ModelServiceDependency) -> SuccessResponse[ModelManifestData]:
    return SuccessResponse(data=service.get_manifest())
