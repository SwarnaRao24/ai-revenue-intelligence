"""
api/schemas/requests.py
Pydantic request models for all endpoints.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    customer_id: str = Field(..., example="CUST_001")
    tenure_months: float = Field(..., ge=0, example=12)
    monthly_charges: float = Field(..., ge=0, example=65.50)
    total_charges: float = Field(..., ge=0, example=786.0)
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., example="Month-to-month"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., example="Electronic check")
    internet_service: Literal["Fiber optic", "DSL", "No"] = Field(
        ..., example="Fiber optic"
    )
    online_security: Literal["Yes", "No", "No internet service"] = "No"
    tech_support: Literal["Yes", "No", "No internet service"] = "No"
    num_support_tickets: int = Field(0, ge=0, example=3)
    is_senior_citizen: int = Field(0, ge=0, le=1, example=0)
    has_partner: Literal["Yes", "No"] = "No"
    has_dependents: Literal["Yes", "No"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"
    gender: Literal["Male", "Female"] = "Male"


class RevenueRequest(BaseModel):
    periods: int = Field(6, ge=1, le=24, example=6)
    granularity: Literal["monthly"] = "monthly"


class CustomerRiskRequest(BaseModel):
    customer_id: str = Field(..., example="CUST_001")
    tenure_months: float = Field(..., example=8)
    monthly_charges: float = Field(..., example=85.0)
    total_charges: float = Field(..., example=680.0)
    contract_type: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    num_support_tickets: int = Field(0, ge=0)
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    internet_service: Literal["Fiber optic", "DSL", "No"] = "Fiber optic"
    online_security: Literal["Yes", "No", "No internet service"] = "No"
    tech_support: Literal["Yes", "No", "No internet service"] = "No"
    is_senior_citizen: int = Field(0, ge=0, le=1)
    has_partner: Literal["Yes", "No"] = "No"
    has_dependents: Literal["Yes", "No"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"
    gender: Literal["Male", "Female"] = "Male"


class SummaryRequest(BaseModel):
    report_type: Literal["executive", "churn", "revenue", "risk"] = "executive"
    date_range: Literal["last_7_days", "last_30_days", "last_90_days"] = "last_30_days"
