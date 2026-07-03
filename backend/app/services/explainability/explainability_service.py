"""
Module 6 -- Explainable AI & Decision Support
Explainability service — public entry point.

Combines:
  - Module 5 prediction (point estimate + interval)
  - Local feature attributions (marginal contribution vs SHAP)
  - Module 3 Knowledge Base (physical meaning + operational interpretation)
  - Module 3 Recommendation Engine (triggered rules = operational alerts)
  - Natural language narrative (4 platform questions answered)

Returns a complete ExplainedPrediction for a mine/date/target.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional

import joblib
import pandas as pd

from ..prediction.prediction_service import predict, get_deployed_models, _load_features, _load_active_model
from ..recommendation.recommendation_engine import get_recommendations
from .local_explainer import explain_prediction, persist_explanations, FeatureAttribution
from .narrative_generator import generate_narrative, NarrativeInsights


@dataclass
class ExplainedPrediction:
    mine_code: int
    date: str
    target_metric: str
    mine_name: Optional[str]
    subsidiary: Optional[str]
    # Prediction (Module 5)
    prediction_status: str
    point_estimate: Optional[float]
    interval_lower: Optional[float]
    interval_upper: Optional[float]
    confidence_label: Optional[str]
    actual_value: Optional[float]
    historical_avg: Optional[float]
    cv_r2: Optional[float]
    # Explanations (Module 6)
    attributions: list
    # Alerts (Module 3 rules evaluated live)
    alerts: list
    # Narrative (4 questions)
    narrative: Optional[dict]
    refusal_reason: Optional[str]


def _get_mine_info(con: sqlite3.Connection, mine_code: int) -> dict:
    cur = con.cursor()
    cur.execute("SELECT mine_name, subsidiary FROM mine_master WHERE mine_code=?", (mine_code,))
    row = cur.fetchone()
    return {"mine_name": row[0], "subsidiary": row[1]} if row else {}


def _get_historical_avg(con: sqlite3.Connection, mine_code: int, target_metric: str) -> Optional[float]:
    col = {"gcv": "gcv", "moisture": "total_moisture_pct", "ash": "ash_pct"}[target_metric]
    validity = {
        "gcv": "gcv_valid=1 AND gcv BETWEEN 0 AND 9000",
        "moisture": "total_moisture_pct BETWEEN 0 AND 100",
        "ash": "ash_pct BETWEEN 0 AND 100",
    }[target_metric]
    cur = con.cursor()
    cur.execute(f"SELECT AVG({col}) FROM sampling_records WHERE mine_code=? AND {validity}", (mine_code,))
    row = cur.fetchone()
    return round(float(row[0]), 2) if row and row[0] else None


def explain(db_path: str, mine_code: int, date: str, target_metric: str,
            log_explanation: bool = True) -> ExplainedPrediction:
    """Full explainability pipeline for one mine/date/target."""
    con = sqlite3.connect(db_path)
    try:
        mine_info = _get_mine_info(con, mine_code)
        historical_avg = _get_historical_avg(con, mine_code, target_metric)

        # Step 1: Prediction (Module 5)
        pred = predict(db_path, mine_code, date, target_metric, log=log_explanation)

        if pred.prediction_status != "success":
            # Still run recommendation engine (weather-agnostic rules can still fire)
            rec_result = get_recommendations(db_path, mine_code, date, log_evaluation=False)
            alerts = [
                {k: v for k, v in r.items() if k not in ("not_yet_evaluable", "not_evaluable_reason")}
                for r in rec_result.get("triggered_recommendations", [])
            ]
            return ExplainedPrediction(
                mine_code=mine_code, date=date, target_metric=target_metric,
                mine_name=mine_info.get("mine_name"), subsidiary=mine_info.get("subsidiary"),
                prediction_status=pred.prediction_status,
                point_estimate=None, interval_lower=None, interval_upper=None,
                confidence_label=None, actual_value=pred.actual_value,
                historical_avg=historical_avg, cv_r2=pred.cv_r2,
                attributions=[], alerts=alerts,
                narrative={
                    "what_happened": f"No prediction possible: {pred.refusal_reason}",
                    "why": "Weather data required for attribution analysis.",
                    "whats_next": "Populate real weather data (Module 2 ingestion) to enable predictions.",
                    "what_to_do": "Check data quality log for ingestion status.",
                },
                refusal_reason=pred.refusal_reason,
            )

        # Step 2: Local feature attributions
        model_meta = _load_active_model(con, target_metric)
        features_df, status, _ = _load_features(con, mine_code, date, model_meta["feature_columns"])
        fitted_model = joblib.load(model_meta["model_artifact_path"])

        attributions = explain_prediction(
            db_path=db_path,
            prediction_id=0,  # not yet logged; persisted separately below
            mine_code=mine_code, date=date, target_metric=target_metric,
            features_df=features_df, fitted_model=fitted_model,
            model_name=model_meta["model_name"], top_n=8,
        )

        # Step 3: Recommendation engine (Module 3 rules + evidence)
        rec_result = get_recommendations(db_path, mine_code, date, log_evaluation=False)
        triggered_alerts = rec_result.get("triggered_recommendations", [])

        # Step 4: Natural language narrative
        narrative_obj = generate_narrative(
            mine_code=mine_code, date=date, target_metric=target_metric,
            point_estimate=pred.point_estimate,
            interval_lower=pred.interval_lower,
            interval_upper=pred.interval_upper,
            confidence_label=pred.confidence_label or "low",
            actual_value=pred.actual_value,
            historical_avg=historical_avg,
            attributions=attributions,
            active_rules=triggered_alerts,
            mine_name=mine_info.get("mine_name", "This mine"),
            cv_r2=pred.cv_r2 or 0.0,
        )

        # Step 5: Persist alerts and insights
        if log_explanation:
            _persist_alerts(con, mine_code, date, triggered_alerts)
            _persist_insights(con, mine_code, date, target_metric, narrative_obj)

        return ExplainedPrediction(
            mine_code=mine_code, date=date, target_metric=target_metric,
            mine_name=mine_info.get("mine_name"), subsidiary=mine_info.get("subsidiary"),
            prediction_status="success",
            point_estimate=pred.point_estimate,
            interval_lower=pred.interval_lower,
            interval_upper=pred.interval_upper,
            confidence_label=pred.confidence_label,
            actual_value=pred.actual_value,
            historical_avg=historical_avg,
            cv_r2=pred.cv_r2,
            attributions=[
                {
                    "rank": a.attribution_rank,
                    "feature": a.feature_name,
                    "attribution_score": a.attribution_score,
                    "method": a.attribution_method,
                    "feature_value": a.feature_value,
                    "kb_physical_meaning": a.kb_physical_meaning,
                    "kb_operational_interp": a.kb_operational_interp,
                    "kb_validation_status": a.kb_validation_status,
                }
                for a in attributions
            ],
            alerts=triggered_alerts,
            narrative=asdict(narrative_obj),
            refusal_reason=None,
        )
    finally:
        con.close()


def _persist_alerts(con: sqlite3.Connection, mine_code: int, date: str, alerts: list) -> None:
    cur = con.cursor()
    cur.execute("DELETE FROM operational_alerts WHERE mine_code=? AND date=?", (mine_code, date))
    for a in alerts:
        cur.execute(
            """INSERT INTO operational_alerts
               (mine_code, date, rule_name, severity, recommendation_category,
                recommendation_text, confidence_level, evidence_available,
                evidence_missing, recommendation_basis)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (mine_code, date,
             a.get("rule_name", ""), a.get("severity", "info"),
             a.get("recommendation_category", "general"),
             a.get("recommendation_text", ""), a.get("adjusted_confidence", "low"),
             json.dumps(a.get("evidence_available", [])),
             json.dumps(a.get("evidence_missing", [])),
             json.dumps(a.get("recommendation_basis", [])),
            )
        )
    con.commit()


def _persist_insights(con: sqlite3.Connection, mine_code: int, date: str,
                       target_metric: str, narrative) -> None:
    cur = con.cursor()
    cur.execute("DELETE FROM mine_insights WHERE mine_code=? AND date=? AND target_metric=?",
                (mine_code, date, target_metric))
    for itype, text in [
        ("what_happened", narrative.what_happened),
        ("why", narrative.why),
        ("whats_next", narrative.whats_next),
        ("what_to_do", narrative.what_to_do),
    ]:
        cur.execute(
            """INSERT INTO mine_insights
               (mine_code, date, insight_type, target_metric, insight_text,
                confidence_level, evidence_sources)
               VALUES (?,?,?,?,?,?,?)""",
            (mine_code, date, itype, target_metric, text,
             "medium", ",".join(narrative.evidence_sources))
        )
    con.commit()


def get_mine_alerts(db_path: str, mine_code: int = None,
                    severity: str = None, limit: int = 50) -> list:
    """Query persisted operational alerts."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        clauses, params = ["is_active=1"], []
        if mine_code:
            clauses.append("mine_code=?"); params.append(mine_code)
        if severity:
            clauses.append("severity=?"); params.append(severity)
        where = "WHERE " + " AND ".join(clauses)
        cur = con.cursor()
        cur.execute(f"SELECT * FROM operational_alerts {where} ORDER BY generated_at DESC LIMIT ?",
                    params + [limit])
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_mine_insights(db_path: str, mine_code: int, date: str,
                       target_metric: str = "moisture") -> list:
    """Retrieve persisted narrative insights for a mine/date."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """SELECT insight_type, insight_text, confidence_level, evidence_sources
               FROM mine_insights WHERE mine_code=? AND date=? AND target_metric=?
               ORDER BY id""",
            (mine_code, date, target_metric)
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
