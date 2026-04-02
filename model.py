"""
GridSense AI — model.py
Machine Learning module: trains a Random Forest regressor to predict future load.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from utils import load_or_generate_dataset

# ✅ Proper paths
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "gridsense_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "gridsense_scaler.joblib")


class GridModel:

    def __init__(self, use_random_forest: bool = True):
        self.use_random_forest = use_random_forest
        self.model_name = "RandomForest" if use_random_forest else "LinearRegression"
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

        os.makedirs(MODEL_DIR, exist_ok=True)

        # Try loading existing model
        self._try_load()

        # If not trained → train automatically
        if not self.is_trained:
            print("⚡ No saved model found. Training new model...")
            df = load_or_generate_dataset()
            self.train(df)

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["day_of_week"] = df["datetime"].dt.dayofweek

        df["rolling_mean_5"] = df["global_active_power"].rolling(5, min_periods=1).mean()
        df["rolling_std_5"] = df["global_active_power"].rolling(5, min_periods=1).std().fillna(0)
        df["rolling_mean_10"] = df["global_active_power"].rolling(10, min_periods=1).mean()

        df["load_trend"] = df["global_active_power"].diff().fillna(0)

        df["sub_total"] = (
            df["sub_metering_1"] +
            df["sub_metering_2"] +
            df["sub_metering_3"]
        )

        df["power_factor"] = df["global_active_power"] / (
            df["global_reactive_power"].replace(0, 0.001) +
            df["global_active_power"] + 1e-6
        )

        return df

    def _build_X_y(self, df: pd.DataFrame):
        df = self._engineer_features(df)

        feature_cols = [
            "hour", "minute", "day_of_week",
            "global_active_power", "global_reactive_power",
            "voltage", "global_intensity",
            "rolling_mean_5", "rolling_std_5", "rolling_mean_10",
            "load_trend", "sub_total", "power_factor",
        ]

        self.feature_names = feature_cols

        HORIZON = 30
        df["target"] = df["global_active_power"].shift(-HORIZON)
        df.dropna(inplace=True)

        X = df[feature_cols].values
        y = df["target"].values

        return X, y

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        print(f"🧠 Training {self.model_name} model on {len(df)} rows...")

        X, y = self._build_X_y(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        if self.use_random_forest:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = LinearRegression()

        self.model.fit(X_train_s, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model trained — MAE: {mae:.4f} | R²: {r2:.4f}")

        self._save()

        return {"mae": mae, "r2": r2}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> float:

        if not self.is_trained:
            return features.get("global_active_power", 1.5) * 1.05

        # Ensure feature order is correct
        vec = np.array([
            [features.get(f, 0.0) for f in self.feature_names]
        ])

        vec_scaled = self.scaler.transform(vec)
        prediction = self.model.predict(vec_scaled)[0]

        return max(0.0, float(prediction))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"💾 Model saved → {MODEL_PATH}")

    def _try_load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_trained = True
                print(f"📂 Loaded existing model")
            except Exception as e:
                print(f"⚠️ Load failed: {e}")