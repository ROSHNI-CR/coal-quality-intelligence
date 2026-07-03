"""
Module 6 -- Explainable AI & Decision Support
FastAPI endpoints.
"""
import os
from typing import Optional
from fastapi import APIRouter, Query
from ..services.explainability.explainability_service import (
    explain, get_mine_alerts, get_mine_insights
)

from ..config import DB_PATH
router = APIRouter(prefix="/api/explain", tags=["explainability"])


@router.get("/{mine_code}")
def get_explanation(
    mine_code: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    target: str = Query("moisture", description="gcv | moisture | ash"),
    log: bool = Query(True),
):
    """
    Full explanation for a mine/date/target combining:
    - Module 5 prediction (point estimate + interval)
    - Local feature attributions (marginal contribution or SHAP)
    - Module 3 Knowledge Base physical rationale
    - Module 3 Recommendation Engine alerts
    - Natural language narrative (what happened / why / what's next / what to do)
    """
    result = explain(DB_PATH, mine_code, date, target, log_explanation=log)
    return {
        "mine_code": result.mine_code,
        "mine_name": result.mine_name,
        "subsidiary": result.subsidiary,
        "date": result.date,
        "target_metric": result.target_metric,
        "prediction": {
            "status": result.prediction_status,
            "point_estimate": result.point_estimate,
            "interval_lower": result.interval_lower,
            "interval_upper": result.interval_upper,
            "confidence_label": result.confidence_label,
            "actual_value": result.actual_value,
            "historical_avg": result.historical_avg,
            "cv_r2": result.cv_r2,
        },
        "attributions": result.attributions,
        "alerts": result.alerts,
        "narrative": result.narrative,
        "refusal_reason": result.refusal_reason,
    }


@router.get("/alerts/active")
def get_active_alerts(
    mine_code: Optional[int] = Query(None),
    severity: Optional[str] = Query(None, description="info | warning | critical"),
    limit: int = Query(50, le=200),
):
    """All active operational alerts, optionally filtered by mine or severity."""
    return {"alerts": get_mine_alerts(DB_PATH, mine_code, severity, limit)}


@router.get("/insights/{mine_code}")
def get_insights(
    mine_code: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    target: str = Query("moisture", description="gcv | moisture | ash"),
):
    """Persisted narrative insights for a mine/date/target."""
    return {"insights": get_mine_insights(DB_PATH, mine_code, date, target)}
