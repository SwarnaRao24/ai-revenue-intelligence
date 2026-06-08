"""
api/schemas/responses.py
Pydantic response models for all endpoints.
"""

from typing import Any
from pydantic import BaseModel


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    direction: str   # "increases" or "decreases"


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_tier: str           # HIGH / MEDIUM / LOW
    top_factors: list[FeatureImpact]
    recommendation: str
    model_version: str


class RevenueForecastPoint(BaseModel):
    period: str
    forecast: float
    lower_bound: float
    upper_bound: float


class RevenueResponse(BaseModel):
    periods: int
    granularity: str
    forecast: list[RevenueForecastPoint]
    total_forecast: float
    trend: str               # "growing" / "stable" / "declining"
    message: str


class RiskLevel(BaseModel):
    score: float
    tier: str
    churn_probability: float
    anomaly_score: float
    segment: str
    recommendations: list[str]


class CustomerRiskResponse(BaseModel):
    customer_id: str
    risk: RiskLevel
    key_risk_drivers: list[str]


class SummaryResponse(BaseModel):
    report_type: str
    date_range: str
    summary: str
    key_metrics: dict[str, Any]
    generated_at: str