"""
tests/unit/test_models.py
Unit tests for ML model loading and prediction logic.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestDataLoader:
    """Tests for feature engineering pipeline."""

    def test_load_churn_features_returns_dataframe(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        assert isinstance(df, pd.DataFrame)

    def test_churn_features_has_label_column(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        assert "label" in df.columns

    def test_churn_features_label_is_binary(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        assert set(df["label"].unique()).issubset({0, 1})

    def test_churn_features_no_nulls_in_label(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        assert df["label"].isnull().sum() == 0

    def test_churn_rate_is_realistic(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        churn_rate = df["label"].mean()
        assert 0.10 <= churn_rate <= 0.50

    def test_feature_count_is_sufficient(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        # Should have at least 40 feature columns + customer_id + label
        assert len(df.columns) >= 40

    def test_no_infinite_values(self):
        from ml_training.utils.data_loader import load_churn_features
        df = load_churn_features()
        numeric = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


class TestModelService:
    """Tests for the FastAPI model service."""

    @pytest.fixture(autouse=True)
    def load_service(self):
        from api.services.model_service import model_service
        model_service.load_all()
        self.service = model_service

    def test_models_loaded(self):
        assert self.service._churn_model is not None
        assert self.service._revenue_model is not None
        assert self.service._segment_model is not None
        assert self.service._anomaly_model is not None

    def test_predict_churn_returns_probability(self):
        features = {
            "customer_id":        "TEST_001",
            "tenure_months":      12,
            "monthly_charges":    65.0,
            "total_charges":      780.0,
            "contract_type":      "Month-to-month",
            "payment_method":     "Electronic check",
            "internet_service":   "Fiber optic",
            "online_security":    "No",
            "tech_support":       "No",
            "num_support_tickets": 2,
            "is_senior_citizen":  0,
            "has_partner":        "No",
            "has_dependents":     "No",
            "paperless_billing":  "Yes",
            "gender":             "Male",
        }
        result = self.service.predict_churn(features)
        assert "churn_probability" in result
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_predict_churn_returns_risk_tier(self):
        features = {
            "tenure_months":      6,
            "monthly_charges":    90.0,
            "total_charges":      540.0,
            "contract_type":      "Month-to-month",
            "payment_method":     "Electronic check",
            "internet_service":   "Fiber optic",
            "online_security":    "No",
            "tech_support":       "No",
            "num_support_tickets": 5,
            "is_senior_citizen":  0,
            "has_partner":        "No",
            "has_dependents":     "No",
            "paperless_billing":  "Yes",
            "gender":             "Male",
        }
        result = self.service.predict_churn(features)
        assert result["risk_tier"] in {"HIGH", "MEDIUM", "LOW"}

    def test_predict_churn_returns_top_factors(self):
        features = {
            "tenure_months":      6,
            "monthly_charges":    90.0,
            "total_charges":      540.0,
            "contract_type":      "Month-to-month",
            "payment_method":     "Electronic check",
            "internet_service":   "Fiber optic",
            "online_security":    "No",
            "tech_support":       "No",
            "num_support_tickets": 5,
            "is_senior_citizen":  0,
            "has_partner":        "No",
            "has_dependents":     "No",
            "paperless_billing":  "Yes",
            "gender":             "Male",
        }
        result = self.service.predict_churn(features)
        assert isinstance(result["top_factors"], list)
        assert len(result["top_factors"]) >= 1

    def test_high_risk_customer_scores_above_threshold(self):
        """A customer with all high-risk signals should score HIGH."""
        features = {
            "tenure_months":      2,
            "monthly_charges":    95.0,
            "total_charges":      190.0,
            "contract_type":      "Month-to-month",
            "payment_method":     "Electronic check",
            "internet_service":   "Fiber optic",
            "online_security":    "No",
            "tech_support":       "No",
            "num_support_tickets": 6,
            "is_senior_citizen":  0,
            "has_partner":        "No",
            "has_dependents":     "No",
            "paperless_billing":  "Yes",
            "gender":             "Male",
        }
        result = self.service.predict_churn(features)
        assert result["churn_probability"] >= 0.5

    def test_low_risk_customer_scores_low(self):
        """A long-tenure, contracted customer should score LOW."""
        features = {
            "tenure_months":      60,
            "monthly_charges":    45.0,
            "total_charges":      2700.0,
            "contract_type":      "Two year",
            "payment_method":     "Bank transfer (automatic)",
            "internet_service":   "DSL",
            "online_security":    "Yes",
            "tech_support":       "Yes",
            "num_support_tickets": 0,
            "is_senior_citizen":  0,
            "has_partner":        "Yes",
            "has_dependents":     "Yes",
            "paperless_billing":  "No",
            "gender":             "Female",
        }
        result = self.service.predict_churn(features)
        assert result["risk_tier"] in {"LOW", "MEDIUM"}

    def test_revenue_forecast_returns_correct_periods(self):
        result = self.service.predict_revenue(periods=6)
        assert len(result["forecast"]) == 6

    def test_revenue_forecast_values_are_positive(self):
        result = self.service.predict_revenue(periods=3)
        for point in result["forecast"]:
            assert point["forecast"] > 0

    def test_revenue_trend_is_valid(self):
        result = self.service.predict_revenue(periods=6)
        assert result["trend"] in {"growing", "stable", "declining"}

    def test_summary_returns_all_fields(self):
        result = self.service.generate_summary("executive", "last_30_days")
        assert "summary" in result
        assert "key_metrics" in result
        assert "generated_at" in result
        assert len(result["summary"]) > 50