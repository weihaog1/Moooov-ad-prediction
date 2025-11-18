"""
Optional Feature 2: XGBoost Revenue Prediction Model

Predicts future hourly revenue using machine learning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import joblib
from pathlib import Path


class RevenuePredictor:
    """
    Machine learning model for predicting hourly revenue.
    """

    def __init__(self):
        self.model = None
        self.feature_names = []

    def prepare_features(self, campaign_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction.

        Args:
            campaign_data: Raw campaign data

        Returns:
            (X features, y target)
        """
        df = campaign_data.copy()

        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month

        df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

        # Campaign encoding (simple label encoding)
        if 'campaign_name' in df.columns:
            campaign_map = {name: idx for idx, name in enumerate(df['campaign_name'].unique())}
            df['campaign_id'] = df['campaign_name'].map(campaign_map)

        # Select features
        feature_cols = [
            'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_business_hours'
        ]

        if 'day_of_month' in df.columns:
            feature_cols.extend(['day_of_month', 'month'])

        if 'campaign_id' in df.columns:
            feature_cols.append('campaign_id')

        # Add performance metrics as features (for revenue prediction)
        if 'impressions' in df.columns:
            feature_cols.extend(['impressions', 'clicks', 'spend', 'ctr', 'cpc'])

        # Filter available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Drop rows with missing values
        df = df.dropna(subset=feature_cols + ['revenue'])

        X = df[feature_cols]
        y = df['revenue']

        self.feature_names = feature_cols

        return X, y

    def train(
        self,
        campaign_data: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train XGBoost model.

        Args:
            campaign_data: Campaign performance data
            test_size: Proportion for testing

        Returns:
            Training results and metrics
        """
        print("\nTraining XGBoost revenue prediction model...")

        # Prepare data
        X, y = self.prepare_features(campaign_data)

        if len(X) < 50:
            return {
                'error': 'Insufficient data',
                'message': f'Need at least 50 samples, got {len(X)}',
                'n_samples': len(X)
            }

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # Train model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Metrics
        train_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'mae': float(mean_absolute_error(y_train, y_train_pred)),
            'r2': float(r2_score(y_train, y_train_pred))
        }

        test_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'mae': float(mean_absolute_error(y_test, y_test_pred)),
            'r2': float(r2_score(y_test, y_test_pred))
        }

        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"[OK] Model trained successfully")
        print(f"   Test R2: {test_metrics['r2']:.4f}")
        print(f"   Test MAE: ${test_metrics['mae']:.2f}")

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(top_features),
            'n_features': len(self.feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

    def save_model(self, filepath: str = 'outputs/revenue_model.joblib'):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, filepath)
        print(f"[OK] Model saved to {filepath}")

    def load_model(self, filepath: str = 'outputs/revenue_model.joblib'):
        """Load trained model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print(f"[OK] Model loaded from {filepath}")
