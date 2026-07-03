"""
Module 3 -- Environmental Knowledge Base
MILESTONE 5: FastAPI endpoints.

Thin HTTP layer only -- every endpoint below calls directly into the
already-implemented and already-tested service layers from Module 2 and
Module 3 (environmental_service.py, recommendation_engine.py). No business
logic lives in this file: if behaviour needs to change, change the service
function, not the route.

NOTE ON TESTING: This sandbox has no network egress, so `pip install
fastapi` could not be run here and these routes could not be exercised with
TestClient in this session. The functions they call (get_daily_environmental_
snapshot, get_recommendations, get_rule_catalog, etc.) were already unit-
and integration-tested directly in Milestones 2-3. This file has been
syntax-checked (ast.parse) but NOT runtime-tested against a live FastAPI
app. Recommend running `pytest`/manual smoke test once this is merged into
an environment with FastAPI installed, before relying on it in production.

DB_PATH should be wired to the project's actual database path/config at
integration time -- it is read from an environment variable here with a
sensible default so this file can be dropped into the existing FastAPI app
without modification, assuming the existing app already has a similar
pattern for its other routers.
"""

import os
import sqlite3
from datetime import date as date_cls, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..services.environmental import environmental_service as es
from ..services.recommendation.recommendation_engine import get_recommendations, get_rule_catalog

from ..config import DB_PATH

router = APIRouter(prefix="/api/environmental", tags=["environmental"])


# ---------------------------------------------------------------------------
# Module 2 -- raw/derived weather data endpoints
# ---------------------------------------------------------------------------

@router.get("/ingestion-status")
def ingestion_status():
    """Coverage/health summary of the Environmental Variable Layer (Module 2).
    Use this before trusting any other environmental endpoint's completeness."""
    return es.get_ingestion_status(DB_PATH)


@router.get("/mines/{mine_code}/snapshot")
def daily_snapshot(mine_code: int, date: str = Query(..., description="YYYY-MM-DD")):
    """Full single-day environmental picture for one mine: raw weather +
    same-day features + rolling/derived features. Returns 404 if not yet
    ingested for this mine/date (no fabricated data is ever returned)."""
    result = es.get_daily_environmental_snapshot(DB_PATH, mine_code, date)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No weather data ingested yet for mine_code={mine_code} on {date}. "
                    "Production Open-Meteo ingestion may be pending.",
        )
    return result


@router.get("/mines/{mine_code}/timeseries")
def timeseries(mine_code: int,
               start_date: str = Query(..., description="YYYY-MM-DD"),
               end_date: str = Query(..., description="YYYY-MM-DD")):
    """Daily joined environmental timeseries for one mine over a date range."""
    rows = es.get_environmental_timeseries(DB_PATH, mine_code, start_date, end_date)
    return {"mine_code": mine_code, "start_date": start_date, "end_date": end_date,
            "count": len(rows), "data": rows}


@router.get("/mines/{mine_code}/latest-risk")
def latest_risk(mine_code: int):
    """Most recent day's environmental risk snapshot for a mine. Returns 404
    if no weather data has been ingested for this mine at all."""
    result = es.get_latest_environmental_risk(DB_PATH, mine_code)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No weather data ingested yet for mine_code={mine_code}.")
    return result


@router.get("/risk-overview")
def risk_overview():
    """Latest ERI snapshot for every mapped mine -- feeds National Map /
    risk distribution / high-risk-mines table on the Overview page."""
    return {"mines": es.get_environmental_risk_for_mapped_mines(DB_PATH)}


@router.get("/mines/{mine_code}/dominant-driver")
def dominant_driver(mine_code: int, date: str = Query(..., description="YYYY-MM-DD")):
    """Rule-based (NOT SHAP) dominant environmental driver for a mine/date --
    a fast fallback until Module 5 (Explainable AI) provides a SHAP-based one."""
    result = es.get_dominant_environmental_driver(DB_PATH, mine_code, date)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient weather history to compute a dominant driver for mine_code={mine_code} on {date}.",
        )
    return result


# ---------------------------------------------------------------------------
# Module 3 -- Environmental Knowledge Base endpoints
# ---------------------------------------------------------------------------

def _kb_query(sql_filter: str = "", params: tuple = ()) -> list[dict]:
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM environmental_knowledge_base {sql_filter} ORDER BY variable_name", params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


@router.get("/knowledge-base")
def knowledge_base_list(category: Optional[str] = Query(
        None, description="Filter by variable_category: raw_weather | same_day_derived | rolling_derived")):
    """Generic, mine-agnostic variable encyclopedia. No mine-specific data
    is ever returned from this endpoint -- it is pure scientific reference
    content (Module 3's defining property)."""
    if category:
        return {"variables": _kb_query("WHERE variable_category = ?", (category,))}
    return {"variables": _kb_query()}


@router.get("/knowledge-base/{variable_name}")
def knowledge_base_detail(variable_name: str):
    """Single variable's full Knowledge Base entry, plus its influence
    hypotheses against gcv/moisture/ash."""
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM environmental_knowledge_base WHERE variable_name = ?", (variable_name,))
        kb_row = cur.fetchone()
        if kb_row is None:
            raise HTTPException(status_code=404, detail=f"No Knowledge Base entry for variable '{variable_name}'.")
        cur.execute(
            "SELECT * FROM environmental_variable_influence WHERE variable_name = ? ORDER BY target_metric",
            (variable_name,),
        )
        influences = [dict(r) for r in cur.fetchall()]
        result = dict(kb_row)
        result["influence"] = influences
        return result
    finally:
        con.close()


@router.get("/influence")
def influence_lookup(variable_name: Optional[str] = None, target_metric: Optional[str] = None):
    """Query the generic influence hypothesis register. Every row returned
    has validation_status='pending' until Module 4 (Influence Quantification
    Engine) statistically validates it against real mine-specific data --
    that field is always included so callers never mistake a hypothesis
    for a proven result."""
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        clauses, params = [], []
        if variable_name:
            clauses.append("variable_name = ?")
            params.append(variable_name)
        if target_metric:
            clauses.append("target_metric = ?")
            params.append(target_metric)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(f"SELECT * FROM environmental_variable_influence {where} ORDER BY variable_name, target_metric", params)
        return {"influence": [dict(r) for r in cur.fetchall()]}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Module 3 -- Recommendation Engine endpoints
# ---------------------------------------------------------------------------

@router.get("/recommendations/{mine_code}")
def recommendations(mine_code: int, date: str = Query(..., description="YYYY-MM-DD")):
    """
    Run the Recommendation Engine for a mine/date. Returns:
      - triggered_recommendations: rules whose conditions were met
      - not_yet_evaluable_rules: rules that COULD apply but lack data right now
        (e.g. weather not yet ingested) -- never silently dropped
      - evidence_summary: which evidence categories were available this call

    No machine learning, SHAP, or statistical validation is used here --
    every recommendation is rule-based, evidence-graded, and traceable to
    the Module 3 Knowledge Base.
    """
    return get_recommendations(DB_PATH, mine_code, date)


@router.get("/recommendation-rules")
def recommendation_rules_catalog():
    """Full rule library with metadata -- for an admin/Settings page or for
    documenting the rule set independent of any specific mine/date."""
    return {"rules": get_rule_catalog(DB_PATH)}


# ---------------------------------------------------------------------------
# ERI aggregate stats — for the ERI gauge and risk distribution donut
# ---------------------------------------------------------------------------
@router.get("/eri-summary")
def eri_summary():
    """National ERI statistics computed from derived_environmental_features.
    Powers the ERI gauge (national average) and risk distribution donut."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT
                ROUND(AVG(environmental_risk_index), 1)   AS national_avg_eri,
                ROUND(MIN(environmental_risk_index), 1)   AS min_eri,
                ROUND(MAX(environmental_risk_index), 1)   AS max_eri,
                COUNT(*)                                  AS total_mine_days,
                SUM(CASE WHEN environmental_risk_index < 34  THEN 1 ELSE 0 END) AS low_days,
                SUM(CASE WHEN environmental_risk_index BETWEEN 34 AND 66 THEN 1 ELSE 0 END) AS moderate_days,
                SUM(CASE WHEN environmental_risk_index > 66  THEN 1 ELSE 0 END) AS high_days
            FROM derived_environmental_features
            WHERE environmental_risk_index IS NOT NULL
        """)
        row = dict(cur.fetchone())
        row["tier_counts"] = {
            "low":      row.pop("low_days"),
            "moderate": row.pop("moderate_days"),
            "high":     row.pop("high_days"),
        }
        eri = row["national_avg_eri"] or 0
        row["eri_label"] = "High" if eri > 66 else "Moderate" if eri > 33 else "Low"
        return row
    finally:
        con.close()
