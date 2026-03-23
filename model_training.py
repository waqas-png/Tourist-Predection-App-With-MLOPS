"""
Model Training — Daily Tourism Demand Forecasting
Trains 3 models for BOTH tasks:
  Classification: Demand Tier (Low / Medium / High / Very High)
  Regression: Raw Visitor Count

Models compared: Random Forest | Gradient Boosting | Logistic/Ridge
"""

import pandas as pd, numpy as np, json, time, joblib
from pathlib import Path
from loguru import logger
import mlflow, mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              classification_report)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("models/results")


# ── metrics helpers ──────────────────────────────────────────────

def cls_metrics(name, model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    elapsed = round(time.time() - t0, 1)
    yp = model.predict(Xte)
    acc = round(accuracy_score(yte, yp), 4)
    f1  = round(f1_score(yte, yp, average='macro'), 4)
    try:
        yprob = model.predict_proba(Xte)
        auc   = round(roc_auc_score(yte, yprob, multi_class='ovr', average='macro'), 4)
    except Exception:
        auc = 0.0
    return {'Model': name, 'Accuracy': acc, 'F1 Macro': f1,
            'ROC-AUC': auc, 'Train(s)': elapsed}, model, yp


def reg_metrics(name, model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    elapsed = round(time.time() - t0, 1)
    yp = model.predict(Xte)
    rmse = round(float(mean_squared_error(yte, yp) ** 0.5), 3)
    mae  = round(float(mean_absolute_error(yte, yp)), 3)
    r2   = round(float(r2_score(yte, yp)), 4)
    mape = round(float(np.mean(np.abs((yte - yp) / (np.abs(yte) + 1)))) * 100, 2)
    return {'Model': name, 'RMSE': rmse, 'MAE': mae,
            'R2': r2, 'MAPE%': mape, 'Train(s)': elapsed}, model, np.array(yp)


# ── training pipeline ────────────────────────────────────────────

def run_training_pipeline(X_train, X_test, X_train_s, X_test_s,
                           y_cls_train, y_cls_test,
                           y_reg_train, y_reg_test):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("mlruns")
    exp = "tourist_demand_forecasting"
    mlflow.set_experiment(exp)

    # ── Classification ───────────────────────────────────────────
    logger.info("Training classifiers (demand tier: Low/Medium/High/Very High)...")

    with mlflow.start_run(run_name="RF_classifier"):
        r_rf, m_rf_cls, p_rf = cls_metrics(
            "Random Forest",
            RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            X_train, y_cls_train, X_test, y_cls_test)
        mlflow.log_params({"model": "RandomForest", "task": "classification", "n_estimators": 200})
        mlflow.log_metrics({k: v for k, v in r_rf.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(m_rf_cls, "rf_cls_model")
    logger.info(f"  RF classifier: Acc={r_rf['Accuracy']} F1={r_rf['F1 Macro']} AUC={r_rf['ROC-AUC']}")

    with mlflow.start_run(run_name="GBM_classifier"):
        r_gb, m_gb_cls, p_gb = cls_metrics(
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            X_train, y_cls_train, X_test, y_cls_test)
        mlflow.log_params({"model": "GradientBoosting", "task": "classification"})
        mlflow.log_metrics({k: v for k, v in r_gb.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(m_gb_cls, "gb_cls_model")
    logger.info(f"  GBM classifier: Acc={r_gb['Accuracy']} F1={r_gb['F1 Macro']} AUC={r_gb['ROC-AUC']}")

    with mlflow.start_run(run_name="LR_classifier"):
        r_lr, m_lr_cls, p_lr = cls_metrics(
            "Logistic Regression",
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            X_train_s, y_cls_train, X_test_s, y_cls_test)
        mlflow.log_params({"model": "LogisticRegression", "task": "classification"})
        mlflow.log_metrics({k: v for k, v in r_lr.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(m_lr_cls, "lr_cls_model")
    logger.info(f"  LR classifier:  Acc={r_lr['Accuracy']} F1={r_lr['F1 Macro']} AUC={r_lr['ROC-AUC']}")

    cls_df = pd.DataFrame([r_rf, r_gb, r_lr]).sort_values('F1 Macro', ascending=False)
    best_cls = cls_df.iloc[0]['Model']

    # ── Regression ───────────────────────────────────────────────
    logger.info("Training regressors (visitor count)...")

    with mlflow.start_run(run_name="RF_regressor"):
        rr_rf, rm_rf, rp_rf = reg_metrics(
            "Random Forest",
            RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            X_train, y_reg_train, X_test, y_reg_test)
        mlflow.log_params({"model": "RandomForest", "task": "regression"})
        mlflow.log_metrics({k: v for k, v in rr_rf.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(rm_rf, "rf_reg_model")
    logger.info(f"  RF regressor:  RMSE={rr_rf['RMSE']} MAE={rr_rf['MAE']} R2={rr_rf['R2']}")

    with mlflow.start_run(run_name="GBM_regressor"):
        rr_gb, rm_gb, rp_gb = reg_metrics(
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            X_train, y_reg_train, X_test, y_reg_test)
        mlflow.log_params({"model": "GradientBoosting", "task": "regression"})
        mlflow.log_metrics({k: v for k, v in rr_gb.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(rm_gb, "gb_reg_model")
    logger.info(f"  GBM regressor: RMSE={rr_gb['RMSE']} MAE={rr_gb['MAE']} R2={rr_gb['R2']}")

    with mlflow.start_run(run_name="Ridge_regressor"):
        rr_rg, rm_rg, rp_rg = reg_metrics(
            "Ridge Regression",
            Ridge(alpha=1.0),
            X_train_s, y_reg_train, X_test_s, y_reg_test)
        mlflow.log_params({"model": "Ridge", "task": "regression"})
        mlflow.log_metrics({k: v for k, v in rr_rg.items() if isinstance(v, (int, float)) and k != 'Train(s)'})
        mlflow.sklearn.log_model(rm_rg, "ridge_model")
    logger.info(f"  Ridge:         RMSE={rr_rg['RMSE']} MAE={rr_rg['MAE']} R2={rr_rg['R2']}")

    reg_df = pd.DataFrame([rr_rf, rr_gb, rr_rg]).sort_values('R2', ascending=False)
    best_reg = reg_df.iloc[0]['Model']

    # ── Feature importances ──────────────────────────────────────
    tree_model = rm_rf if best_reg == "Random Forest" else rm_gb
    from src.feature_engineering import FEATURES
    fi = sorted(zip(FEATURES, tree_model.feature_importances_),
                key=lambda x: x[1], reverse=True)[:12]

    # ── Save ─────────────────────────────────────────────────────
    cls_df.to_csv(RESULTS_DIR / "cls_comparison.csv",   index=False)
    reg_df.to_csv(RESULTS_DIR / "reg_comparison.csv",   index=False)

    best_cls_model = {'Random Forest': m_rf_cls, 'Gradient Boosting': m_gb_cls, 'Logistic Regression': m_lr_cls}[best_cls]
    best_reg_model = {'Random Forest': rm_rf,    'Gradient Boosting': rm_gb,    'Ridge Regression': rm_rg}[best_reg]
    joblib.dump(best_cls_model, MODELS_DIR / "best_classifier.pkl")
    joblib.dump(best_reg_model, MODELS_DIR / "best_regressor.pkl")

    json.dump({'feature_importance': fi, 'best_classifier': best_cls,
               'best_regressor': best_reg}, open(RESULTS_DIR / "results.json", "w"), indent=2)

    logger.success(f"Best classifier: {best_cls}")
    logger.success(f"Best regressor:  {best_reg}")
    return cls_df, reg_df, best_cls, best_reg, fi
