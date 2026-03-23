"""
FastAPI Prediction Service
Serves the best trained tourist prediction model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import time
import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🌍 Tourist Prediction MLOps API",
    description="""
    Predicts whether a location will be a **high tourist destination**
    based on travel features. Trained with RF, XGBoost, and LightGBM — 
    best model auto-selected via MLflow experiment tracking.
    
    **Dataset**: Kaggle Tourism Statistics  
    **Models**: Random Forest | XGBoost | LightGBM  
    **MLOps**: MLflow + AWS SageMaker + Docker  
    """,
    version="1.0.0",
    contact={"name": "MLOps Team", "email": "mlops@tourism-ai.com"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("models/artifacts")

# Global model store
model_store = {}
feature_engineer = None
model_metadata = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TourismPredictionRequest(BaseModel):
    """Input schema for a single prediction"""
    year: int = Field(2024, ge=2000, le=2030, example=2024)
    month: int = Field(7, ge=1, le=12, example=7)
    destination_country: str = Field("France", example="France")
    origin_country: str = Field("USA", example="USA")
    season: str = Field("Summer", example="Summer")
    travel_type: str = Field("Leisure", example="Leisure")
    avg_stay_days: float = Field(7.0, gt=0, le=365, example=7.0)
    avg_expenditure_usd: float = Field(1500.0, gt=0, example=1500.0)
    hotel_rating: int = Field(4, ge=1, le=5, example=4)
    advance_booking_days: int = Field(30, ge=0, le=365, example=30)
    group_size: int = Field(2, ge=1, le=50, example=2)
    gdp_per_capita_origin: float = Field(55000.0, gt=0, example=55000.0)
    distance_km: float = Field(7000.0, gt=0, example=7000.0)
    visa_required: int = Field(0, ge=0, le=1, example=0)
    flight_cost_usd: float = Field(800.0, gt=0, example=800.0)
    tourism_index: float = Field(85.0, ge=0, le=100, example=85.0)
    prev_year_visitors_million: float = Field(50.0, gt=0, example=50.0)
    repeat_visitor: int = Field(0, ge=0, le=1, example=0)
    digital_booking: int = Field(1, ge=0, le=1, example=1)
    satisfaction_score: int = Field(4, ge=1, le=5, example=4)

    @validator('season')
    def validate_season(cls, v):
        valid = ['Spring', 'Summer', 'Autumn', 'Winter']
        if v not in valid:
            raise ValueError(f"season must be one of {valid}")
        return v

    @validator('travel_type')
    def validate_travel_type(cls, v):
        valid = ['Leisure', 'Business', 'Education', 'Medical', 'Transit']
        if v not in valid:
            raise ValueError(f"travel_type must be one of {valid}")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction input"""
    requests: List[TourismPredictionRequest]


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: int
    prediction_label: str
    confidence: float
    probability_high: float
    probability_low: float
    model_used: str
    latency_ms: float
    feature_count: int


class ModelComparisonResponse(BaseModel):
    """Multi-model comparison response"""
    models: Dict[str, PredictionResponse]
    consensus: str
    agreement_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global model_store, feature_engineer, model_metadata
    logger.info("Loading models on startup...")

    try:
        fe_path = ARTIFACTS_DIR / "feature_engineer.pkl"
        if fe_path.exists():
            feature_engineer = joblib.load(fe_path)
            logger.success("Feature engineer loaded.")
        else:
            logger.warning("Feature engineer not found. Run training pipeline first.")

        model_names = ["best_model", "randomforest_model", "xgboost_model", "lightgbm_model"]
        for name in model_names:
            path = MODELS_DIR / f"{name}.pkl"
            if path.exists():
                model_store[name] = joblib.load(path)
                logger.success(f"Model '{name}' loaded.")

        results_path = Path("models/results/results.json")
        if results_path.exists():
            with open(results_path) as f:
                model_metadata = json.load(f)

    except Exception as e:
        logger.error(f"Error loading models: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_input(req: TourismPredictionRequest) -> pd.DataFrame:
    """Convert request to DataFrame for prediction"""
    data = req.dict()
    df = pd.DataFrame([data])
    return df


def predict_with_model(model, df_input: pd.DataFrame):
    """Run prediction with feature engineering"""
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature engineer not loaded. Train models first.")
    X = feature_engineer.transform(df_input)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred, float(proba[1]), float(proba[0])


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Tourist Prediction MLOps API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": list(model_store.keys()),
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(model_store),
        "feature_engineer": feature_engineer is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: TourismPredictionRequest):
    """
    Predict tourist destination category using the best model.
    Returns: 1 = High Tourist Destination, 0 = Low Tourist Destination
    """
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded. Run training pipeline.")

    model_key = "best_model" if "best_model" in model_store else list(model_store.keys())[0]
    model = model_store[model_key]

    t0 = time.perf_counter()
    df_input = preprocess_input(request)
    pred, prob_high, prob_low = predict_with_model(model, df_input)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        prediction=pred,
        prediction_label="High Tourist Destination" if pred == 1 else "Low Tourist Destination",
        confidence=max(prob_high, prob_low),
        probability_high=prob_high,
        probability_low=prob_low,
        model_used=model_key,
        latency_ms=round(latency, 2),
        feature_count=len(feature_engineer.feature_columns) if feature_engineer else 0
    )


@app.post("/predict/compare", response_model=ModelComparisonResponse, tags=["Prediction"])
async def predict_compare_all_models(request: TourismPredictionRequest):
    """
    Run prediction across all 3 models and compare outputs.
    Useful for understanding model disagreements.
    """
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded.")

    df_input = preprocess_input(request)
    model_map = {
        "RandomForest": "randomforest_model",
        "XGBoost": "xgboost_model",
        "LightGBM": "lightgbm_model"
    }

    responses = {}
    predictions = []

    for display_name, key in model_map.items():
        if key not in model_store:
            continue
        model = model_store[key]
        t0 = time.perf_counter()
        pred, prob_high, prob_low = predict_with_model(model, df_input)
        latency = (time.perf_counter() - t0) * 1000
        predictions.append(pred)

        responses[display_name] = PredictionResponse(
            prediction=pred,
            prediction_label="High Tourist Destination" if pred == 1 else "Low Tourist Destination",
            confidence=max(prob_high, prob_low),
            probability_high=prob_high,
            probability_low=prob_low,
            model_used=display_name,
            latency_ms=round(latency, 2),
            feature_count=len(feature_engineer.feature_columns) if feature_engineer else 0
        )

    if not predictions:
        raise HTTPException(status_code=503, detail="No individual models found.")

    majority = int(round(sum(predictions) / len(predictions)))
    agreement = predictions.count(majority) / len(predictions)

    return ModelComparisonResponse(
        models=responses,
        consensus="High Tourist Destination" if majority == 1 else "Low Tourist Destination",
        agreement_score=round(agreement, 2)
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch predict for multiple inputs"""
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded.")

    model_key = "best_model" if "best_model" in model_store else list(model_store.keys())[0]
    model = model_store[model_key]

    results = []
    for req in request.requests:
        df_input = preprocess_input(req)
        pred, prob_high, prob_low = predict_with_model(model, df_input)
        results.append({
            "prediction": pred,
            "label": "High" if pred == 1 else "Low",
            "confidence": round(max(prob_high, prob_low), 4)
        })

    return {"count": len(results), "predictions": results, "model_used": model_key}


@app.get("/models", tags=["Models"])
async def list_models():
    """List all loaded models and their metadata"""
    return {
        "loaded_models": list(model_store.keys()),
        "performance_metrics": model_metadata,
        "feature_count": len(feature_engineer.feature_columns) if feature_engineer else None
    }


@app.get("/models/comparison", tags=["Models"])
async def get_model_comparison():
    """Return saved model comparison results"""
    results_path = Path("models/results/model_comparison.csv")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Comparison results not found. Run training first.")

    import pandas as pd
    df = pd.read_csv(results_path)
    return {"comparison": df.to_dict(orient="records")}
