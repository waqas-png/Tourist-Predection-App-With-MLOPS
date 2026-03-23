"""
API and Unit Tests for Tourist Prediction MLOps
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for the TourismFeatureEngineer pipeline"""

    def get_sample_df(self):
        return pd.DataFrame([{
            'year': 2023, 'month': 7,
            'destination_country': 'France', 'origin_country': 'USA',
            'season': 'Summer', 'travel_type': 'Leisure',
            'avg_stay_days': 7.0, 'avg_expenditure_usd': 1500.0,
            'hotel_rating': 4, 'advance_booking_days': 30, 'group_size': 2,
            'gdp_per_capita_origin': 55000.0, 'distance_km': 7000.0,
            'visa_required': 0, 'flight_cost_usd': 800.0, 'tourism_index': 85.0,
            'prev_year_visitors_million': 50.0, 'repeat_visitor': 0,
            'digital_booking': 1, 'satisfaction_score': 4,
            'visitor_count_thousands': 70.0, 'high_tourist_destination': 1
        }] * 100)

    def test_time_features_created(self):
        from src.feature_engineering import TourismFeatureEngineer
        fe = TourismFeatureEngineer()
        df = self.get_sample_df()
        result = fe.create_time_features(df)
        assert 'quarter' in result.columns
        assert 'is_peak_season' in result.columns
        assert 'year_normalized' in result.columns

    def test_interaction_features_created(self):
        from src.feature_engineering import TourismFeatureEngineer
        fe = TourismFeatureEngineer()
        df = self.get_sample_df()
        df = fe.create_time_features(df)
        result = fe.create_interaction_features(df)
        assert 'cost_per_day' in result.columns
        assert 'value_score' in result.columns
        assert 'booking_urgency' in result.columns

    def test_fit_transform_returns_correct_shapes(self):
        from src.feature_engineering import TourismFeatureEngineer
        fe = TourismFeatureEngineer()
        df = self.get_sample_df()
        X, y, df_eng = fe.fit_transform(df)
        assert len(X) == len(df)
        assert len(y) == len(df)
        assert X.shape[1] > 15  # At least 15 features

    def test_no_nulls_after_transform(self):
        from src.feature_engineering import TourismFeatureEngineer
        fe = TourismFeatureEngineer()
        df = self.get_sample_df()
        X, y, _ = fe.fit_transform(df)
        assert X.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────
# Data Ingestion Tests
# ─────────────────────────────────────────────────────────────────

class TestDataIngestion:

    def test_synthetic_data_generated(self):
        from src.data_ingestion import create_synthetic_tourism_data
        df = create_synthetic_tourism_data()
        assert len(df) == 5000
        assert 'high_tourist_destination' in df.columns
        assert df['high_tourist_destination'].isin([0, 1]).all()

    def test_required_columns_present(self):
        from src.data_ingestion import create_synthetic_tourism_data
        df = create_synthetic_tourism_data()
        required = ['year', 'month', 'destination_country', 'tourism_index',
                    'avg_stay_days', 'flight_cost_usd', 'high_tourist_destination']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nulls_in_synthetic_data(self):
        from src.data_ingestion import create_synthetic_tourism_data
        df = create_synthetic_tourism_data()
        assert df.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────
# Model Training Tests
# ─────────────────────────────────────────────────────────────────

class TestModelTraining:

    def get_mock_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 10),
                         columns=[f'feat_{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, n))
        return X, y

    def test_get_metrics_returns_all_keys(self):
        from src.model_training import get_metrics
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        metrics = get_metrics(y_true, y_pred, y_proba)
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


# ─────────────────────────────────────────────────────────────────
# API Tests (mocked models)
# ─────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Tests using FastAPI TestClient with mocked model"""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        mock_fe = MagicMock()
        mock_fe.is_fitted = True
        mock_fe.feature_columns = [f'feat_{i}' for i in range(10)]
        mock_fe.transform.return_value = pd.DataFrame(
            np.zeros((1, 10)), columns=[f'feat_{i}' for i in range(10)]
        )

        from api.main import app, model_store, feature_engineer
        import api.main as api_module

        api_module.model_store['best_model'] = mock_model
        api_module.feature_engineer = mock_fe

        return TestClient(app)

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint(self, client):
        payload = {
            "year": 2024, "month": 7,
            "destination_country": "France", "origin_country": "USA",
            "season": "Summer", "travel_type": "Leisure",
            "avg_stay_days": 7.0, "avg_expenditure_usd": 1500.0,
            "hotel_rating": 4, "advance_booking_days": 30, "group_size": 2,
            "gdp_per_capita_origin": 55000.0, "distance_km": 7000.0,
            "visa_required": 0, "flight_cost_usd": 800.0,
            "tourism_index": 85.0, "prev_year_visitors_million": 50.0,
            "repeat_visitor": 0, "digital_booking": 1, "satisfaction_score": 4
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_invalid_season_rejected(self, client):
        payload = {
            "year": 2024, "month": 7,
            "destination_country": "France", "origin_country": "USA",
            "season": "INVALID_SEASON",  # invalid
            "travel_type": "Leisure",
            "avg_stay_days": 7.0, "avg_expenditure_usd": 1500.0,
            "hotel_rating": 4, "advance_booking_days": 30, "group_size": 2,
            "gdp_per_capita_origin": 55000.0, "distance_km": 7000.0,
            "visa_required": 0, "flight_cost_usd": 800.0,
            "tourism_index": 85.0, "prev_year_visitors_million": 50.0,
            "repeat_visitor": 0, "digital_booking": 1, "satisfaction_score": 4
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
