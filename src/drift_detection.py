"""
Lightweight data-drift detection using PSI (categorical) and z-score (numeric).
The baseline is computed from the training split and saved as data/baseline_stats.json
by src/train.py. The API calls check_drift() on each /drift/check request.
"""
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASELINE_PATHS = [
    "/app/data/baseline_stats.json",  # Docker
    "data/baseline_stats.json",       # local
]


def load_baseline() -> dict:
    for path in _BASELINE_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    logger.warning("No baseline_stats.json found; drift checks will be skipped")
    return {}


def check_drift(records: list[dict], baseline: dict) -> dict:
    """
    Compare a list of incoming prediction input dicts against the training baseline.
    Returns per-feature drift scores and an overall flag.
    """
    if not baseline or not records:
        return {"error": "baseline or records missing", "is_drifted": False}

    df = pd.DataFrame(records)
    results = {}

    for col, info in baseline.items():
        if col not in df.columns:
            continue

        if info["type"] == "numeric":
            incoming_mean = float(df[col].mean())
            score = abs(incoming_mean - info["mean"]) / max(info["std"], 1e-8)
            results[col] = {
                "type": "numeric",
                "drift_score": round(score, 4),
                "is_drifted": score > 2.0,
                "incoming_mean": round(incoming_mean, 4),
                "baseline_mean": round(info["mean"], 4),
            }
        else:
            baseline_dist = info["value_counts"]
            incoming_dist = df[col].value_counts(normalize=True).to_dict()
            psi = 0.0
            for cat, base_pct in baseline_dist.items():
                inc_pct = max(incoming_dist.get(cat, 1e-6), 1e-6)
                base_pct = max(base_pct, 1e-6)
                psi += (inc_pct - base_pct) * np.log(inc_pct / base_pct)
            psi_abs = abs(psi)
            results[col] = {
                "type": "categorical",
                "psi": round(psi_abs, 4),
                "is_drifted": psi_abs > 0.2,
                "incoming_distribution": {k: round(v, 4) for k, v in incoming_dist.items()},
            }

    any_drift = any(v.get("is_drifted", False) for v in results.values())
    return {"features": results, "is_drifted": any_drift, "n_records": len(records)}
