"""
GridSense AI — simulator.py
Real-time simulation engine: manages transformer state, generates realistic
load data, and handles load-shifting logic.
"""

import threading
import time
import random
import math
from datetime import datetime
from collections import deque
from typing import Dict, Any


# ─── Transformer Configuration ────────────────────────────────────────────────

TRANSFORMERS_CONFIG = {
    "T1": {"capacity_kw": 5000, "location": "Zone A – Industrial", "base_load": 0.55},
    "T2": {"capacity_kw": 3500, "location": "Zone B – Residential", "base_load": 0.45},
    "T3": {"capacity_kw": 4000, "location": "Zone C – Commercial", "base_load": 0.60},
}

HISTORY_SIZE = 60       # Keep last 60 readings per transformer
TICK_INTERVAL = 3       # Seconds between simulation ticks


class GridSimulator:
    """
    Simulates a network of power transformers with realistic load fluctuations.
    Runs a background thread that updates transformer loads every TICK_INTERVAL seconds.
    """

    def __init__(self):
        self.transformers: Dict[str, Dict[str, Any]] = {}
        self.history: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Initialize transformer state
        for tid, cfg in TRANSFORMERS_CONFIG.items():
            initial_load = cfg["capacity_kw"] * cfg["base_load"]
            self.transformers[tid] = {
                "capacity_kw": cfg["capacity_kw"],
                "location": cfg["location"],
                "load_kw": initial_load,
                "load_percent": round(cfg["base_load"] * 100, 1),
                "voltage": 230.0 + random.uniform(-5, 5),
                "current_a": round(initial_load / 0.23, 1),
                "frequency_hz": 50.0,
                "trend": 0.0,
                "phase": random.uniform(0, 2 * math.pi),  # Random phase offset
            }
            self.history[tid] = deque(maxlen=HISTORY_SIZE)

    # ── Background Simulation Thread ──────────────────────────────────────────

    def start(self):
        """Start the background simulation thread."""
        self._running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
        print(f"⚡ Simulator started — updating every {TICK_INTERVAL}s")

    def stop(self):
        """Gracefully stop the simulation."""
        self._running = False

    def _simulation_loop(self):
        """Core simulation loop — runs in background thread."""
        tick = 0
        while self._running:
            with self._lock:
                for tid in self.transformers:
                    self._tick_transformer(tid, tick)
            tick += 1
            time.sleep(TICK_INTERVAL)

    def _tick_transformer(self, tid: str, tick: int):
        """
        Update a single transformer's load using:
        - Diurnal sinusoidal pattern (peak at ~18:00)
        - Random Gaussian noise
        - Gradual trend drift
        """
        t = self.transformers[tid]
        cap = t["capacity_kw"]
        hour = datetime.now().hour + datetime.now().minute / 60.0

        # Diurnal pattern: low at night, peak in evening
        diurnal = 0.3 + 0.4 * math.sin(math.pi * (hour - 6) / 12) ** 2

        # Per-transformer phase shift for variety
        phase_shift = t["phase"]
        pattern = 0.5 + 0.3 * math.sin(2 * math.pi * tick / 40 + phase_shift)

        # Base load as fraction of capacity
        base_frac = TRANSFORMERS_CONFIG[tid]["base_load"]

        # Combine patterns
        target_frac = base_frac * 0.5 + diurnal * 0.3 + pattern * 0.2

        # Add Gaussian noise
        noise = random.gauss(0, 0.015)
        target_frac = max(0.1, min(1.05, target_frac + noise))

        # Smooth transition toward target (exponential smoothing)
        alpha = 0.15
        current_frac = t["load_kw"] / cap
        new_frac = alpha * target_frac + (1 - alpha) * current_frac

        new_load = new_frac * cap
        trend = new_load - t["load_kw"]

        # Update transformer state
        t["load_kw"] = round(new_load, 2)
        t["load_percent"] = round(new_frac * 100, 1)
        t["voltage"] = round(230.0 - (new_frac - 0.5) * 20 + random.gauss(0, 0.5), 2)
        t["current_a"] = round(new_load / (t["voltage"] * 0.001), 1)
        t["frequency_hz"] = round(50.0 - (new_frac - 0.5) * 0.4, 3)
        t["trend"] = round(trend, 3)

        # Record history
        self.history[tid].append({
            "timestamp": datetime.utcnow().isoformat(),
            "load_kw": t["load_kw"],
            "load_percent": t["load_percent"],
            "voltage": t["voltage"],
        })

    # ── External Data Ingestion ───────────────────────────────────────────────

    def update_transformer(self, tid: str, load_kw: float):
        """Force-update a transformer's load from an external API call."""
        with self._lock:
            t = self.transformers[tid]
            t["load_kw"] = round(load_kw, 2)
            t["load_percent"] = round(load_kw / t["capacity_kw"] * 100, 1)

    # ── Feature Builder for ML ────────────────────────────────────────────────

    def build_features(self, tid: str) -> dict:
        """
        Build a feature dict from the transformer's current state and history
        for passing to the ML model.
        """
        t = self.transformers[tid]
        hist = list(self.history[tid])
        loads = [h["load_kw"] for h in hist] or [t["load_kw"]]

        now = datetime.now()
        # Convert load_kw to approximate global_active_power (kW)
        gap = t["load_kw"] / 1000.0

        # Rolling stats
        loads_arr = loads[-10:] if len(loads) >= 10 else loads
        rolling_mean_5 = sum(loads_arr[-5:]) / min(5, len(loads_arr)) / 1000.0
        rolling_mean_10 = sum(loads_arr) / len(loads_arr) / 1000.0
        rolling_std_5 = (
            (sum((x / 1000.0 - rolling_mean_5) ** 2 for x in loads_arr[-5:]) / max(1, len(loads_arr[-5:]) - 1)) ** 0.5
            if len(loads_arr) > 1 else 0.0
        )
        trend = (loads[-1] - loads[-2]) / 1000.0 if len(loads) >= 2 else 0.0

        return {
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),
            "global_active_power": gap,
            "global_reactive_power": gap * 0.2,
            "voltage": t["voltage"],
            "global_intensity": t["current_a"] / 100.0,
            "rolling_mean_5": rolling_mean_5,
            "rolling_std_5": rolling_std_5,
            "rolling_mean_10": rolling_mean_10,
            "load_trend": trend,
            "sub_total": gap * 0.6,
            "power_factor": 0.9,
        }

    # ── Load Shifting ─────────────────────────────────────────────────────────

    def shift_load(self, from_id: str, to_id: str, amount_kw: float) -> dict:
        """
        Transfer `amount_kw` of load from one transformer to another.
        Validates that the receiving transformer has headroom.
        """
        with self._lock:
            src = self.transformers[from_id]
            dst = self.transformers[to_id]

            headroom = dst["capacity_kw"] - dst["load_kw"]
            actual_shift = min(amount_kw, headroom, src["load_kw"] * 0.3)

            if actual_shift <= 0:
                return {
                    "status": "failed",
                    "reason": f"{to_id} has no headroom ({headroom:.0f} kW available)",
                    "shifted_kw": 0,
                }

            # Apply the shift
            src["load_kw"] = round(src["load_kw"] - actual_shift, 2)
            src["load_percent"] = round(src["load_kw"] / src["capacity_kw"] * 100, 1)
            dst["load_kw"] = round(dst["load_kw"] + actual_shift, 2)
            dst["load_percent"] = round(dst["load_kw"] / dst["capacity_kw"] * 100, 1)

            return {
                "status": "success",
                "from": from_id,
                "to": to_id,
                "shifted_kw": round(actual_shift, 2),
                "new_load_from": src["load_kw"],
                "new_load_to": dst["load_kw"],
                "new_percent_from": src["load_percent"],
                "new_percent_to": dst["load_percent"],
                "timestamp": datetime.utcnow().isoformat(),
            }

    def auto_redistribute(self) -> dict:
        """
        Automatically find overloaded transformers and shift load to the
        least-loaded one with available headroom.
        """
        with self._lock:
            actions = []
            # Find overloaded (>85%) and underloaded transformers
            overloaded = {
                tid: t for tid, t in self.transformers.items() if t["load_percent"] > 85
            }
            if not overloaded:
                return {"status": "no_action_needed", "message": "All transformers are within safe limits.", "actions": []}

            for tid, t in overloaded.items():
                # Find best candidate to receive load
                candidates = [
                    (cid, ct) for cid, ct in self.transformers.items()
                    if cid != tid and ct["load_percent"] < 75
                ]
                if not candidates:
                    continue
                # Pick least loaded
                best_id, best = min(candidates, key=lambda x: x[1]["load_percent"])
                shift_amt = (t["load_kw"] - t["capacity_kw"] * 0.80)  # Bring to 80%
                result = self._shift_unlocked(tid, best_id, shift_amt)
                if result["status"] == "success":
                    actions.append(result)

            return {
                "status": "redistributed" if actions else "no_valid_targets",
                "actions": actions,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _shift_unlocked(self, from_id: str, to_id: str, amount_kw: float) -> dict:
        """Internal shift without acquiring lock (caller must hold lock)."""
        src = self.transformers[from_id]
        dst = self.transformers[to_id]
        headroom = dst["capacity_kw"] - dst["load_kw"]
        actual_shift = min(amount_kw, headroom, src["load_kw"] * 0.3)
        if actual_shift <= 0:
            return {"status": "failed", "reason": "No headroom", "shifted_kw": 0}
        src["load_kw"] = round(src["load_kw"] - actual_shift, 2)
        src["load_percent"] = round(src["load_kw"] / src["capacity_kw"] * 100, 1)
        dst["load_kw"] = round(dst["load_kw"] + actual_shift, 2)
        dst["load_percent"] = round(dst["load_kw"] / dst["capacity_kw"] * 100, 1)
        return {
            "status": "success",
            "from": from_id,
            "to": to_id,
            "shifted_kw": round(actual_shift, 2),
            "new_percent_from": src["load_percent"],
            "new_percent_to": dst["load_percent"],
        }

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_transformer_states(self) -> Dict[str, Any]:
        with self._lock:
            return {tid: dict(t) for tid, t in self.transformers.items()}

    def get_history(self, tid: str, limit: int = 20):
        with self._lock:
            hist = list(self.history.get(tid, []))
            return hist[-limit:]
