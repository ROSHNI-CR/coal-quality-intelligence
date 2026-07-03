"""
Module 9 -- Reports
Exports: CSV (always available), JSON summary, PDF (when reportlab installed).
Provides mine-level, national summary, and influence quantification exports.
"""
import csv
import io
import json
import os
import sqlite3
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse

from ..config import DB_PATH
router = APIRouter(prefix="/api/reports", tags=["reports"])


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@router.get("/national-summary")
def national_summary_report(fmt: str = Query("json", description="json | csv")):
    """National coal quality summary: per-subsidiary averages and totals."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT m.subsidiary,
                   COUNT(DISTINCT m.mine_code)                                              as mines,
                   COUNT(s.gcv)                                                             as samples,
                   ROUND(AVG(CASE WHEN s.gcv_valid=1 THEN s.gcv END),0)                    as avg_gcv,
                   ROUND(MIN(CASE WHEN s.gcv_valid=1 THEN s.gcv END),0)                    as min_gcv,
                   ROUND(MAX(CASE WHEN s.gcv_valid=1 THEN s.gcv END),0)                    as max_gcv,
                   ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100
                             THEN s.total_moisture_pct END),2)                              as avg_moisture,
                   ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100
                             THEN s.ash_pct END),2)                                         as avg_ash,
                   MIN(substr(s.date,1,10))                                                 as data_from,
                   MAX(substr(s.date,1,10))                                                 as data_to
            FROM mine_master m
            LEFT JOIN sampling_records s ON s.mine_code=m.mine_code
            WHERE m.is_mapped=1
            GROUP BY m.subsidiary ORDER BY avg_gcv DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=national_summary_{datetime.now().strftime('%Y%m%d')}.csv"})
    return {"generated_at": datetime.now().isoformat(), "subsidiaries": rows}


@router.get("/mine-profiles")
def mine_profiles_report(
    subsidiary: str = Query(None),
    min_samples: int = Query(10),
    fmt: str = Query("json", description="json | csv"),
):
    """Per-mine coal quality profile export for all (or filtered) mapped mines."""
    con = _connect()
    try:
        cur = con.cursor()
        sub_clause = "AND m.subsidiary=?" if subsidiary else ""
        params = [subsidiary] if subsidiary else []
        params.append(min_samples)
        cur.execute(f"""
            SELECT m.mine_code, m.mine_name, m.subsidiary,
                   ROUND(m.latitude,6) as latitude, ROUND(m.longitude,6) as longitude,
                   COUNT(s.gcv)                                              as samples,
                   ROUND(AVG(CASE WHEN s.gcv_valid=1 THEN s.gcv END),0)     as avg_gcv,
                   ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100
                             THEN s.total_moisture_pct END),2)               as avg_moisture,
                   ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100
                             THEN s.ash_pct END),2)                          as avg_ash,
                   MIN(substr(s.date,1,10)) as first_sample,
                   MAX(substr(s.date,1,10)) as last_sample
            FROM mine_master m
            LEFT JOIN sampling_records s ON s.mine_code=m.mine_code
            WHERE m.is_mapped=1 {sub_clause}
            GROUP BY m.mine_code HAVING samples >= ?
            ORDER BY avg_gcv DESC
        """, params)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mine_profiles_{datetime.now().strftime('%Y%m%d')}.csv"})
    return {"generated_at": datetime.now().isoformat(), "mine_count": len(rows), "mines": rows}


@router.get("/influence-rankings")
def influence_rankings_report(
    target: str = Query("gcv", description="gcv | moisture | ash"),
    fmt: str = Query("json", description="json | csv"),
):
    """Module 4 environmental influence rankings for a target metric."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT id FROM module4_run_metadata WHERE run_completed_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return JSONResponse({"error": "No completed Module 4 run found"}, status_code=404)
        run_id = row[0]
        cur.execute("""
            SELECT ml_importance_rank, variable_name, ml_importance_score,
                   ml_direction, pearson_r, spearman_rho, best_lag_days,
                   agreement_with_kb, validation_status_assigned
            FROM environmental_influence_quantification
            WHERE run_id=? AND target_metric=?
            ORDER BY ml_importance_rank
        """, (run_id, target))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=influence_{target}_{datetime.now().strftime('%Y%m%d')}.csv"})
    return {"target": target, "run_id": run_id, "rankings": rows}


@router.get("/prediction-backtest")
def prediction_backtest_report(fmt: str = Query("json", description="json | csv")):
    """All logged predictions with actuals, for backtest accuracy analysis."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT mine_code, date, target_metric, prediction_status,
                   point_estimate, interval_lower, interval_upper,
                   confidence_label, actual_value,
                   CASE WHEN actual_value IS NOT NULL AND point_estimate IS NOT NULL
                        THEN ROUND(ABS(point_estimate - actual_value), 3) END as abs_error,
                   CASE WHEN actual_value IS NOT NULL AND interval_lower IS NOT NULL
                        THEN CASE WHEN actual_value BETWEEN interval_lower AND interval_upper
                             THEN 1 ELSE 0 END END as in_interval,
                   requested_at
            FROM predictions
            WHERE prediction_status='success'
            ORDER BY requested_at DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    if fmt == "csv":
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=backtest_{datetime.now().strftime('%Y%m%d')}.csv"})
    return {"generated_at": datetime.now().isoformat(), "total_predictions": len(rows), "predictions": rows}
