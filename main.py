"""
Main MLOps Pipeline Orchestrator
Runs the complete end-to-end tourist prediction pipeline:
  1. Data Ingestion (Kaggle)
  2. Feature Engineering
  3. Train 3 Models (RF, XGBoost, LightGBM)
  4. Compare & Select Best
  5. Save Artifacts
"""

import sys
import json
import time
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")
logger.add("logs/pipeline.log", rotation="10 MB", level="DEBUG")

Path("logs").mkdir(exist_ok=True)


def run_pipeline():
    """Execute complete MLOps pipeline"""
    start_time = time.time()

    logger.info("=" * 65)
    logger.info("  TOURIST PREDICTION MLOps PIPELINE")
    logger.info("  Models: Random Forest | XGBoost | LightGBM")
    logger.info("  Deployment: AWS ECS Fargate")
    logger.info("=" * 65)

    # ── Step 1: Data Ingestion ─────────────────────────────────────
    logger.info("\n📦 STEP 1/4: Data Ingestion")
    from src.data_ingestion import load_and_save_data
    df = load_and_save_data()
    logger.success(f"  Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Target distribution:\n{df['high_tourist_destination'].value_counts().to_string()}")

    # ── Step 2: Feature Engineering ────────────────────────────────
    logger.info("\n⚙️  STEP 2/4: Feature Engineering & Preprocessing")
    from src.feature_engineering import run_preprocessing_pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, fe = run_preprocessing_pipeline(df)
    logger.success(f"  Features engineered: {X_train.shape[1]} features")
    logger.success(f"  Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)} samples")

    # ── Step 3: Model Training ──────────────────────────────────────
    logger.info("\n🤖 STEP 3/4: Training 3 Models + MLflow Tracking")
    from src.model_training import run_training_pipeline
    trained_models, all_results, comparison_df, best_model_name = run_training_pipeline(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # ── Step 4: Final Report ────────────────────────────────────────
    logger.info("\n📊 STEP 4/4: Model Comparison Report")
    logger.info("\n" + "="*65)
    logger.info("                  FINAL MODEL COMPARISON")
    logger.info("="*65)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    logger.info("\n" + "="*65)

    best_metrics = all_results[best_model_name]
    logger.info(f"\n🏆 WINNER: {best_model_name}")
    logger.info(f"   Accuracy  : {best_metrics['accuracy']:.4f}")
    logger.info(f"   F1 Score  : {best_metrics['f1_score']:.4f}")
    logger.info(f"   ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    logger.info(f"   Precision : {best_metrics['precision']:.4f}")
    logger.info(f"   Recall    : {best_metrics['recall']:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    logger.info("\nNext steps:")
    logger.info("  1. Run: uvicorn api.main:app --reload --port 8080")
    logger.info("  2. Open: http://localhost:8080/docs")
    logger.info("  3. MLflow UI: mlflow ui  →  http://localhost:5000")
    logger.info("  4. Deploy: docker build -t tourist-api . && docker run -p 8080:8080 tourist-api")
    logger.info("  5. AWS: cd infrastructure && cdk deploy")

    return best_model_name, comparison_df


if __name__ == "__main__":
    best_model, results = run_pipeline()
