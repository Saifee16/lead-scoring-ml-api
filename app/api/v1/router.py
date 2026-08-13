from fastapi import APIRouter

from app.api.v1.endpoints import health, model, prediction

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(prediction.router, tags=["Prediction"])
api_router.include_router(model.router, tags=["Model"])
