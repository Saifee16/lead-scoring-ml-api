from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.common import SuccessResponse
from app.schemas.lead import (
    BatchLeadScoreData,
    BatchLeadScoreRequest,
    LeadFeatures,
    LeadScoreData,
)
from app.services.model_service import ModelService, get_model_service

router = APIRouter()
ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/predict-lead-score",
    response_model=SuccessResponse[LeadScoreData],
)
def predict_lead_score(
    lead: LeadFeatures,
    service: ModelServiceDependency,
) -> SuccessResponse[LeadScoreData]:
    prediction = service.predict(lead)
    return SuccessResponse(data=prediction)


@router.post(
    "/predict-lead-scores",
    response_model=SuccessResponse[BatchLeadScoreData],
)
def predict_lead_scores(
    request: BatchLeadScoreRequest,
    service: ModelServiceDependency,
) -> SuccessResponse[BatchLeadScoreData]:
    predictions = service.predict_many(request.leads)
    return SuccessResponse(
        data=BatchLeadScoreData(
            count=len(predictions),
            predictions=predictions,
        )
    )
