"""
api/routers/predict.py
All prediction endpoints.
"""

from fastapi import APIRouter, HTTPException

from api.schemas.requests import (
    ChurnRequest,
    CustomerRiskRequest,
    RevenueRequest,
    SummaryRequest,
)
from api.schemas.responses import (
    ChurnResponse,
    CustomerRiskResponse,
    RevenueResponse,
    SummaryResponse,
)
from api.services.model_service import model_service

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


@router.post("/predict/churn", response_model=ChurnResponse)
async def predict_churn(request: ChurnRequest):
    """
    Predict churn probability for a single customer.
    Returns probability, risk tier, top factors, and recommendation.
    """
    try:
        result = model_service.predict_churn(request.model_dump())
        return ChurnResponse(customer_id=request.customer_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/revenue", response_model=RevenueResponse)
async def predict_revenue(request: RevenueRequest):
    """
    Forecast monthly revenue for the next N periods.
    """
    try:
        result = model_service.predict_revenue(request.periods)
        return RevenueResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/customer-risk", response_model=CustomerRiskResponse)
async def customer_risk(request: CustomerRiskRequest):
    """
    Full risk assessment combining churn probability,
    anomaly score, and customer segment.
    """
    try:
        result = model_service.assess_risk(request.model_dump())
        return CustomerRiskResponse(
            customer_id=request.customer_id,
            **result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate an AI-style executive summary for the chosen report type.
    """
    try:
        result = model_service.generate_summary(
            request.report_type,
            request.date_range,
        )
        return SummaryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
