# 🌍 Tourist Prediction MLOps Platform

> **End-to-end MLOps pipeline** for predicting high-tourist destinations.  
> Trains and compares **3 ML models**, tracks experiments with **MLflow**, serves via **FastAPI**, and deploys to **AWS ECS Fargate**.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                              │
├──────────┬──────────┬───────────┬────────────┬─────────────────┤
│  Kaggle  │ Feature  │  Training │  MLflow    │   AWS Fargate   │
│  Data    │ Engineer │  3 Models │  Tracking  │   + ALB         │
│  ↓       │  ↓       │  ↓        │  ↓         │  ↓              │
│ 5000     │ 27 feat  │ RF/XGB/   │ Experiment │ FastAPI         │
│ records  │ created  │ LightGBM  │ comparison │ /predict        │
└──────────┴──────────┴───────────┴────────────┴─────────────────┘
```

```
Internet → Route53 → ALB (80/443)
                      ↓
               ECS Fargate Cluster
               ┌─────────────────┐
               │  API Container  │
               │  FastAPI:8080   │◄── S3 Models
               │  2 vCPU / 4GB   │◄── MLflow S3
               └─────────────────┘
                      ↓
               CloudWatch Logs + Alarms
```

---

## 🗂 Project Structure

```
tourist-prediction-mlops/
├── src/
│   ├── data_ingestion.py       # Kaggle download + synthetic data
│   ├── feature_engineering.py  # 27-feature pipeline + encoders
│   └── model_training.py       # RF + XGBoost + LightGBM + MLflow
├── api/
│   └── main.py                 # FastAPI service (predict, compare, batch)
├── infrastructure/
│   └── aws_cdk_stack.py        # AWS CDK: VPC + ECS + ALB + S3 + CW
├── tests/
│   └── test_pipeline.py        # Pytest unit + integration tests
├── .github/workflows/
│   └── mlops_pipeline.yml      # CI/CD: test → train → build → deploy
├── Dockerfile                  # Multi-stage production image
├── docker-compose.yml          # Local dev stack (API + MLflow + Grafana)
├── main.py                     # Pipeline orchestrator
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/tourist-prediction-mlops.git
cd tourist-prediction-mlops

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Configure Kaggle API

```bash
# Download your kaggle.json from kaggle.com → Account → API
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Dataset used:
# https://www.kaggle.com/datasets/tourist-arrivals
```

### 3. Run the Full Pipeline

```bash
python main.py
```

This will:
- ✅ Download/generate tourism dataset
- ✅ Engineer 27 features
- ✅ Train Random Forest, XGBoost, LightGBM
- ✅ Log all experiments to MLflow
- ✅ Compare models and select the best
- ✅ Save all artifacts to `models/`

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8080
```

- **API docs**: http://localhost:8080/docs
- **Health**: http://localhost:8080/health

### 5. View MLflow Dashboard

```bash
mlflow ui
```

Open: http://localhost:5000

---

## 🤖 Models Compared

| Model | Hyperparameters | Description |
|-------|----------------|-------------|
| **Random Forest** | n_estimators=200, max_depth=12 | Ensemble of decision trees with bootstrap sampling |
| **XGBoost** | n_estimators=300, lr=0.05, max_depth=6 | Gradient boosting with regularization |
| **LightGBM** | n_estimators=300, num_leaves=63, lr=0.05 | Leaf-wise gradient boosting, fastest training |

All models evaluated on: **Accuracy, F1 Score, ROC-AUC, Precision, Recall**

Best model auto-selected by composite score: `(F1 + ROC-AUC) / 2`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction (best model) |
| POST | `/predict/compare` | Prediction from all 3 models |
| POST | `/predict/batch` | Batch predictions |
| GET | `/models` | List loaded models |
| GET | `/models/comparison` | View training comparison |

### Example Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024,
    "month": 7,
    "destination_country": "France",
    "origin_country": "USA",
    "season": "Summer",
    "travel_type": "Leisure",
    "avg_stay_days": 7.0,
    "avg_expenditure_usd": 1500.0,
    "hotel_rating": 4,
    "advance_booking_days": 30,
    "group_size": 2,
    "gdp_per_capita_origin": 55000.0,
    "distance_km": 7000.0,
    "visa_required": 0,
    "flight_cost_usd": 800.0,
    "tourism_index": 85.0,
    "prev_year_visitors_million": 50.0,
    "repeat_visitor": 0,
    "digital_booking": 1,
    "satisfaction_score": 4
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "High Tourist Destination",
  "confidence": 0.847,
  "probability_high": 0.847,
  "probability_low": 0.153,
  "model_used": "best_model",
  "latency_ms": 4.21,
  "feature_count": 27
}
```

---

## 🐳 Docker

```bash
# Build
docker build -t tourist-prediction-api .

# Run
docker run -p 8080:8080 -v $(pwd)/models:/app/models tourist-prediction-api

# Full stack (API + MLflow + Grafana)
docker-compose up

# Training only
docker-compose --profile training run trainer
```

---

## ☁️ AWS Deployment

### Prerequisites

```bash
npm install -g aws-cdk
pip install aws-cdk-lib constructs
aws configure  # Set access key, secret, region
```

### Deploy Infrastructure

```bash
cd infrastructure
cdk bootstrap aws://YOUR_ACCOUNT_ID/eu-west-1
cdk deploy --context env=prod
```

### Push & Deploy via CI/CD

```bash
# Set GitHub Secrets:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ACCOUNT_ID
# KAGGLE_USERNAME, KAGGLE_API_KEY

git push origin main  # Triggers full CI/CD pipeline
```

The GitHub Actions pipeline will:
1. Run tests
2. Train models (if `main` branch)
3. Upload artifacts to S3
4. Build Docker image → ECR
5. Deploy to ECS Fargate
6. Run smoke tests

---

## 📊 Features Engineered (27 total)

**Original (15)**: month, avg_stay_days, avg_expenditure_usd, hotel_rating, advance_booking_days, group_size, gdp_per_capita_origin, distance_km, visa_required, flight_cost_usd, tourism_index, prev_year_visitors_million, repeat_visitor, digital_booking, satisfaction_score

**Time (3)**: quarter, is_peak_season, year_normalized

**Interaction (5)**: cost_per_day, value_score, booking_urgency, group_spending_potential, distance_penalty

**Encoded (4)**: destination_country_enc, origin_country_enc, season_enc, travel_type_enc

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## 📈 MLflow Experiment Tracking

All runs logged to `mlruns/` with:
- Model parameters & hyperparameters
- Validation metrics (accuracy, F1, ROC-AUC)
- Feature importance JSON
- Model artifacts (registered in MLflow Model Registry)

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python module path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENV` | `development` | Environment name |
| `MODEL_BUCKET` | - | S3 bucket for models (AWS) |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow server URI |
| `KAGGLE_USERNAME` | - | Kaggle account username |
| `KAGGLE_KEY` | - | Kaggle API key |
