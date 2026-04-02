"""
GridSense AI — Smart Grid Safety & Energy Optimization Platform
main.py: FastAPI application with all API endpoints
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime

from model import GridModel
from simulator import GridSimulator
from utils import detect_risk, generate_recommendations, load_or_generate_dataset

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GridSense AI",
    description="AI-based Smart Grid Safety & Energy Optimization Platform",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: Train model and init simulator ─────────────────────────────────
grid_model = GridModel()
simulator = GridSimulator()

@app.on_event("startup")
async def startup_event():
    """Load dataset, train model, and start simulator on app startup."""
    print("🔌 GridSense AI starting up...")
    df = load_or_generate_dataset()
    grid_model.train(df)
    simulator.start()
    print("✅ GridSense AI is ready.")


# ─── Request / Response Models ────────────────────────────────────────────────
class EnergyReading(BaseModel):
    global_active_power: float
    global_reactive_power: float
    voltage: float
    global_intensity: float
    sub_metering_1: float
    sub_metering_2: float
    sub_metering_3: float
    transformer_id: Optional[str] = "T1"

class ShiftRequest(BaseModel):
    from_transformer: str
    to_transformer: str
    load_amount: float


# ─── Endpoints ───────────────────────────────────────────────────────────────
import os
from fastapi.responses import FileResponse

@app.get("/")
def serve_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "static", "dashboard.html")
    return FileResponse(file_path)

@app.get("/status", tags=["Grid"])
def get_status():
    """Return current state of all transformers (load, risk level, metadata)."""
    states = simulator.get_transformer_states()
    enriched = {}
    for tid, state in states.items():
        risk = detect_risk(state["load_percent"])
        enriched[tid] = {
            **state,
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    return {"transformers": enriched, "total_transformers": len(enriched)}


@app.post("/data", tags=["Grid"])
def ingest_data(reading: EnergyReading):
    """Ingest a new energy reading and update the simulator state."""
    tid = reading.transformer_id
    if tid not in simulator.transformers:
        raise HTTPException(status_code=404, detail=f"Transformer {tid} not found.")

    load_kw = reading.global_active_power * 1000  # Convert kW to W equivalent
    simulator.update_transformer(tid, load_kw)

    risk = detect_risk(simulator.transformers[tid]["load_percent"])
    return {
        "status": "accepted",
        "transformer_id": tid,
        "load_kw": load_kw,
        "risk_level": risk["level"],
    }


@app.get("/predict", tags=["ML"])
def predict_load():
    """Use the trained ML model to predict next 30-minute load per transformer."""
    states = simulator.get_transformer_states()
    predictions = {}
    for tid, state in states.items():
        features = simulator.build_features(tid)
        pred = grid_model.predict(features)
        predictions[tid] = {
            "current_load_kw": round(state["load_kw"], 2),
            "predicted_load_30min_kw": round(float(pred), 2),
            "predicted_load_percent": round(float(pred) / state["capacity_kw"] * 100, 1),
            "trend": "increasing" if pred > state["load_kw"] else "decreasing",
        }
    return {"predictions": predictions, "model": grid_model.model_name, "horizon_minutes": 30}


@app.get("/alert", tags=["Safety"])
def get_alerts():
    """Return risk alerts for all transformers."""
    states = simulator.get_transformer_states()
    alerts = []
    for tid, state in states.items():
        risk = detect_risk(state["load_percent"])
        if risk["level"] != "SAFE":
            alerts.append({
                "transformer_id": tid,
                "risk_level": risk["level"],
                "load_percent": state["load_percent"],
                "message": risk["message"],
                "timestamp": datetime.utcnow().isoformat(),
            })
    return {
        "alerts": alerts,
        "alert_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["risk_level"] == "CRITICAL"),
        "warning_count": sum(1 for a in alerts if a["risk_level"] == "WARNING"),
    }


@app.get("/recommend", tags=["AI"])
def get_recommendations():
    """AI-powered recommendations based on current grid state."""
    states = simulator.get_transformer_states()
    recs = []
    for tid, state in states.items():
        risk = detect_risk(state["load_percent"])
        suggestions = generate_recommendations(tid, state, risk["level"], states)
        recs.append({
            "transformer_id": tid,
            "risk_level": risk["level"],
            "load_percent": state["load_percent"],
            "recommendations": suggestions,
        })
    # Sort: CRITICAL first, then WARNING, then SAFE
    priority = {"CRITICAL": 0, "WARNING": 1, "SAFE": 2}
    recs.sort(key=lambda x: priority[x["risk_level"]])
    return {"recommendations": recs, "generated_at": datetime.utcnow().isoformat()}


@app.post("/shift", tags=["Control"])
def shift_load(req: ShiftRequest):
    """Simulate transferring load from one transformer to another."""
    if req.from_transformer not in simulator.transformers:
        raise HTTPException(status_code=404, detail=f"{req.from_transformer} not found.")
    if req.to_transformer not in simulator.transformers:
        raise HTTPException(status_code=404, detail=f"{req.to_transformer} not found.")

    result = simulator.shift_load(req.from_transformer, req.to_transformer, req.load_amount)
    return result

@app.post("/auto-shift", tags=["Control"])
def auto_shift():
    """Automatically detect overloaded transformers and redistribute load."""
    result = simulator.auto_redistribute()
    return result



@app.get("/history/{transformer_id}", tags=["Grid"])
def get_history(transformer_id: str, limit: int = 20):
    """Return recent load history for a transformer."""
    if transformer_id not in simulator.transformers:
        raise HTTPException(status_code=404, detail=f"{transformer_id} not found.")
    history = simulator.get_history(transformer_id, limit)
    return {"transformer_id": transformer_id, "history": history}


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
