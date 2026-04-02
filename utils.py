"""
GridSense AI — utils.py
Utility functions: risk detection, recommendation engine, dataset loading & preprocessing.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# ✅ Dataset path
DATASET_PATH = "data/household_power.csv"

# ─────────────────────────────────────────────────────────────
# ⚡ RISK DETECTION
# ─────────────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "SAFE":     (0,   75,  "#22c55e", "Operating within safe limits."),
    "WARNING":  (75,  90,  "#f59e0b", "Approaching capacity — monitor closely."),
    "CRITICAL": (90, 110,  "#ef4444", "Overload risk — immediate action required!"),
}

def detect_risk(load_percent: float) -> dict:
    """
    Determine risk level based on load percentage.
    """
    for level, (low, high, color, message) in RISK_THRESHOLDS.items():
        if low <= load_percent < high:
            return {
                "level": level,
                "color": color,
                "message": message
            }

    # fallback (if >110%)
    return {
        "level": "CRITICAL",
        "color": "#ef4444",
        "message": "Severe overload detected!"
    }


# ─────────────────────────────────────────────────────────────
# 🤖 RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────

def generate_recommendations(transformer_id: str, state: Dict, risk_level: str, all_states: Dict):
    """
    Generate intelligent suggestions based on transformer condition.
    """
    recs = []

    load_kw = state["load_kw"]
    capacity = state["capacity_kw"]
    load_percent = state["load_percent"]

    if risk_level == "CRITICAL":
        overload = max(0, load_kw - (0.85 * capacity))

        recs.append({
            "priority": "CRITICAL",
            "action": "Immediate Load Shedding",
            "detail": f"Reduce ~{int(overload)} kW immediately to prevent failure."
        })

        # Suggest best transformer for load shift
        target_id, target_state = min(
            all_states.items(),
            key=lambda x: x[1]["load_percent"]
        )

        if target_id != transformer_id:
            recs.append({
                "priority": "CRITICAL",
                "action": "Shift Load",
                "detail": f"Transfer load to {target_id} (currently {target_state['load_percent']:.1f}%)."
            })

        recs.append({
            "priority": "HIGH",
            "action": "Activate Backup Power",
            "detail": "Switch non-critical loads to backup supply (UPS/Diesel)."
        })

    elif risk_level == "WARNING":
        recs.append({
            "priority": "HIGH",
            "action": "Reduce Non-Critical Load",
            "detail": f"Reduce ~{int(load_kw * 0.1)} kW to stay under safe limits."
        })

        recs.append({
            "priority": "MEDIUM",
            "action": "Monitor Trends",
            "detail": "Observe load patterns for next 30 minutes."
        })

    else:
        recs.append({
            "priority": "INFO",
            "action": "Normal Operation",
            "detail": f"{transformer_id} is stable at {load_percent:.1f}% load."
        })

    return recs


# ─────────────────────────────────────────────────────────────
# 📊 DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_or_generate_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load dataset if exists, else generate synthetic dataset.
    """

    # ✅ Ensure folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"📂 Loading dataset from {path}")
        df = pd.read_csv(path, sep=";", low_memory=False, na_values=["?"])
    else:
        print("⚠ Dataset not found — generating synthetic dataset")
        df = generate_synthetic_dataset(path)

    return preprocess(df)


# ─────────────────────────────────────────────────────────────
# 🧪 SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(path: str, n_rows: int = 5000) -> pd.DataFrame:
    """
    Generate realistic synthetic smart grid dataset.
    """

    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1)

    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]
    hours = np.array([t.hour for t in timestamps])

    # Daily load pattern
    diurnal = 0.5 + 0.8 * np.sin(np.pi * (hours - 6) / 12) ** 2
    gap = np.clip(diurnal + rng.normal(0, 0.15, n_rows), 0.1, 5.0)

    df = pd.DataFrame({
        "date": [t.strftime("%d/%m/%Y") for t in timestamps],
        "time": [t.strftime("%H:%M:%S") for t in timestamps],
        "global_active_power": gap,
        "global_reactive_power": gap * 0.15,
        "voltage": 230 + rng.normal(0, 2, n_rows),
        "global_intensity": gap * 5,
        "sub_metering_1": gap * 100,
        "sub_metering_2": gap * 80,
        "sub_metering_3": gap * 60,
    })

    df.to_csv(path, sep=";", index=False)
    print(f"✅ Synthetic dataset created: {path}")

    return df


# ─────────────────────────────────────────────────────────────
# 🧹 PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare dataset for ML model.
    """

    df = df.copy()

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Combine date + time → datetime
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert numeric columns
    numeric_cols = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values safely
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Remove invalid datetime rows
    df.dropna(subset=["datetime"], inplace=True)

    print(f"✅ Dataset ready: {len(df)} rows")

    return df