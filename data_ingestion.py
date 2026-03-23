"""
Data Ingestion Module
Loads the Kaggle Daily Tourism Demand Forecasting dataset (2000-2015).
Dataset: 29,220 rows × 8 columns across 5 world cities.
Kaggle: https://www.kaggle.com/datasets/daily-tourism-demand-forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

RAW_DATA_DIR = Path("data/raw")

DATASET_INFO = {
    "name": "Daily Tourism Demand Forecasting 2000-2015",
    "rows": 29220,
    "columns": 8,
    "destinations": ["Paris", "Tokyo", "New York", "London", "Berlin"],
    "date_range": "2000-01-01 to 2015-12-31",
    "target": "Visitor_Count (50–299 daily per city)",
    "kaggle_slug": "iamsouravbanerjee/daily-tourism-demand-forecasting"
}


def load_dataset(path: str = None) -> pd.DataFrame:
    """Load the tourism dataset from local path or download from Kaggle."""
    if path and Path(path).exists():
        df = pd.read_csv(path)
        logger.success(f"Loaded dataset from {path}: {df.shape}")
        return df

    # Try Kaggle download
    try:
        import kaggle
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from Kaggle: {DATASET_INFO['kaggle_slug']}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_INFO["kaggle_slug"],
            path=str(RAW_DATA_DIR), unzip=True
        )
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            logger.success(f"Downloaded: {df.shape}")
            return df
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")

    raise FileNotFoundError(
        "Dataset not found. Please provide path= or configure Kaggle API credentials.\n"
        f"Dataset: {DATASET_INFO['kaggle_slug']}"
    )


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate expected schema."""
    required = ['Date', 'Destination', 'Visitor_Count', 'Hotel_Occupancy',
                'Flight_Arrivals', 'Average_Temperature', 'Economic_Index', 'Major_Event']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    logger.success(f"Schema valid. Shape: {df.shape}")
    return True


if __name__ == "__main__":
    df = load_dataset("data/raw/daily_tourism_demand_forecasting_2000_2015.csv")
    validate_dataset(df)
    print(df.head())
