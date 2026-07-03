"""
Module 5 -- Prediction Engine
FastAPI endpoints.

Scope: historical/backtest predictions only. Refuses when weather
features are unavailable. Returns point estimates, prediction intervals,
confidence labels, model metadata. No explanation or narrative -- Module 6.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..services.prediction.prediction_service import (
    predict, predict_all_targets,
    get_prediction_history, get_deployed_models,
)

from ..config import DB_PATH
router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/models")
def list_models():
    """All deployed prediction models with CV metrics, interval offsets,
    and provenance. Includes both active and previous model versions."""
    return {"models": get_deployed_models(DB_PATH)}


@router.get("/{mine_code}")
def get_prediction(
    mine_code: int,
    date: str = Query(..., description="YYYY-MM-DD — must have real weather ingested"),
    target: str = Query("gcv", description="gcv | moisture | ash"),
    log: bool = Query(True, description="Persist to prediction history"),
):
    """
    Single prediction for one mine/date/target.

    Returns prediction_status='success' with point_estimate, interval,
    confidence_label, and actual_value (if a sampling record exists for
    this date, enabling immediate backtest comparison).

    Returns prediction_status='refused_*' with refusal_reason when
    weather data is unavailable, the mine is unmapped, or features are
    incomplete. A refusal is always explicit — the API never returns a
    fabricated estimate.

    confidence_label interpretation:
      'medium'   — GroupKFold R² ≥ 0.40 (use with care, not high confidence)
      'low'      — GroupKFold R² 0.25–0.39 (directional guidance only)
      'very_low' — GroupKFold R² < 0.25 (ash model — treat as indicative only)
    """
    if target not in ("gcv", "moisture", "ash"):
        raise HTTPException(status_code=400, detail="target must be 'gcv', 'moisture', or 'ash'")
    result = predict(DB_PATH, mine_code, date, target, log=log)
    return result.__dict__


@router.get("/{mine_code}/all")
def get_prediction_all_targets(
    mine_code: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    log: bool = Query(True),
):
    """Predict GCV, moisture, and ash in a single call."""
    return predict_all_targets(DB_PATH, mine_code, date, log=log)


@router.get("/history/log")
def prediction_history(
    mine_code: Optional[int] = Query(None),
    target: Optional[str] = Query(None, description="gcv | moisture | ash"),
    limit: int = Query(50, le=500),
):
    """Historical prediction log. Includes both successful predictions
    and refused attempts, with actual values for backtest comparison
    where sampling records exist."""
    return {
        "predictions": get_prediction_history(DB_PATH, mine_code, target, limit)
    }


@router.get("/backtest/summary")
def backtest_summary(
    target: str = Query("gcv", description="gcv | moisture | ash"),
    limit: int = Query(200, le=1000),
):
    """
    Backtest accuracy summary: for all logged successful predictions
    where an actual sampling value exists, compute mean absolute error,
    mean error (bias), and % of actuals falling within the prediction
    interval. Provides ground-truth evaluation of model performance
    on real production data.
    """
    import sqlite3, math
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT point_estimate, actual_value, interval_lower, interval_upper
            FROM predictions
            WHERE target_metric=? AND prediction_status='success'
              AND actual_value IS NOT NULL AND point_estimate IS NOT NULL
            ORDER BY requested_at DESC LIMIT ?
            """,
            (target, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return {"target": target, "n_samples": 0, "note": "No completed backtest predictions yet."}

        errors = [abs(r["point_estimate"] - r["actual_value"]) for r in rows]
        biases = [r["point_estimate"] - r["actual_value"] for r in rows]
        in_interval = sum(
            1 for r in rows
            if r["interval_lower"] <= r["actual_value"] <= r["interval_upper"]
        )
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))

        return {
            "target": target,
            "n_samples": len(rows),
            "mae": round(sum(errors) / len(errors), 3),
            "rmse": round(rmse, 3),
            "mean_bias": round(sum(biases) / len(biases), 3),
            "interval_coverage_pct": round(100 * in_interval / len(rows), 1),
            "interval_note": "10th-90th percentile OOF residual interval. Expected coverage ~80% if calibrated.",
        }
    finally:
        con.close()
