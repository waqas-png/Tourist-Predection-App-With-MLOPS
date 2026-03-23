"""
Feature Engineering & Preprocessing Pipeline
Handles all transformations, encoding, and feature creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import joblib
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")


class TourismFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for tourism prediction.
    Handles encoding, scaling, and feature creation.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.is_fitted = False

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        # Quarter
        df['quarter'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12],
                               labels=[1, 2, 3, 4]).astype(int)

        # Is peak season (Jun-Aug + Dec)
        df['is_peak_season'] = df['month'].isin([6, 7, 8, 12]).astype(int)

        # Year normalized
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1e-8)

        logger.info("Time features created: quarter, is_peak_season, year_normalized")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and ratio features"""
        df = df.copy()

        # Cost per day
        df['cost_per_day'] = (df['flight_cost_usd'] + df['avg_expenditure_usd']) / (df['avg_stay_days'] + 1)

        # Value score (stay * satisfaction / cost)
        df['value_score'] = (df['avg_stay_days'] * df['satisfaction_score']) / (df['flight_cost_usd'] / 1000 + 1)

        # Booking urgency (inverse of advance days)
        df['booking_urgency'] = 1 / (df['advance_booking_days'] + 1)

        # GDP x Group size (spending potential)
        df['group_spending_potential'] = np.log1p(df['gdp_per_capita_origin']) * df['group_size']

        # Distance penalty
        df['distance_penalty'] = np.log1p(df['distance_km'])

        logger.info("Interaction features created: cost_per_day, value_score, booking_urgency, etc.")
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns"""
        df = df.copy()

        self.categorical_columns = ['destination_country', 'origin_country',
                                    'season', 'travel_type']

        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df[col + '_encoded'] = le.transform(df[col].astype(str))

        logger.info(f"Encoded {len(self.categorical_columns)} categorical columns")
        return df

    def select_features(self, df: pd.DataFrame) -> list:
        """Return final feature list for modeling"""
        feature_cols = [
            # Original numerical
            'month', 'avg_stay_days', 'avg_expenditure_usd', 'hotel_rating',
            'advance_booking_days', 'group_size', 'gdp_per_capita_origin',
            'distance_km', 'visa_required', 'flight_cost_usd', 'tourism_index',
            'prev_year_visitors_million', 'repeat_visitor', 'digital_booking',
            'satisfaction_score',
            # Time features
            'quarter', 'is_peak_season', 'year_normalized',
            # Interaction features
            'cost_per_day', 'value_score', 'booking_urgency',
            'group_spending_potential', 'distance_penalty',
            # Encoded categoricals
            'destination_country_encoded', 'origin_country_encoded',
            'season_encoded', 'travel_type_encoded',
        ]
        return [col for col in feature_cols if col in df.columns]

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'high_tourist_destination'):
        """Full pipeline: fit and transform"""
        logger.info("Starting feature engineering pipeline (fit)...")

        df = self.create_time_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categoricals(df, fit=True)

        feature_cols = self.select_features(df)
        self.feature_columns = feature_cols

        X = df[feature_cols].fillna(0)
        y = df[target_col] if target_col in df.columns else None

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_cols
        )

        self.is_fitted = True
        logger.success(f"Feature engineering complete. Features: {len(feature_cols)}")

        return X_scaled, y, df

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")

        df = self.create_time_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categoricals(df, fit=False)

        X = df[self.feature_columns].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns
        )
        return X_scaled

    def save(self, path: str = None):
        """Save fitted pipeline to disk"""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or str(ARTIFACTS_DIR / "feature_engineer.pkl")
        joblib.dump(self, save_path)
        logger.success(f"Feature engineer saved to {save_path}")

    @classmethod
    def load(cls, path: str = None):
        """Load fitted pipeline from disk"""
        load_path = path or str(ARTIFACTS_DIR / "feature_engineer.pkl")
        fe = joblib.load(load_path)
        logger.info(f"Feature engineer loaded from {load_path}")
        return fe


def prepare_train_test_split(X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2,
                              val_size: float = 0.1,
                              random_state: int = 42):
    """Create train / validation / test splits"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    logger.info(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing_pipeline(raw_df: pd.DataFrame):
    """Execute full preprocessing and return splits + artifacts"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fe = TourismFeatureEngineer()
    X, y, df_engineered = fe.fit_transform(raw_df, target_col='high_tourist_destination')

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(X, y)

    # Save processed splits
    for name, data in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                        ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        data.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    fe.save()

    logger.success("Preprocessing pipeline complete. All splits and artifacts saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test, fe


if __name__ == "__main__":
    from src.data_ingestion import load_and_save_data
    df = load_and_save_data()
    X_train, X_val, X_test, y_train, y_val, y_test, fe = run_preprocessing_pipeline(df)
    print(f"\nTraining features: {X_train.shape}")
    print(f"Feature list: {fe.feature_columns}")
