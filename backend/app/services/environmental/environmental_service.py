"""
Module 2 — Environmental Variable Layer
Environmental Service — the ONLY interface future modules should use to read
weather/environmental data. Nothing here depends on ML or SHAP.

PRODUCTION-ONLY DATA CONTRACT: every row this service can return originates
from the real Open-Meteo Archive API (ingestion.py). There is no synthetic
data path anywhere in Module 2. If weather_records / environmental_features
/ derived_environmental_features are empty or partially populated for a
mine/date, the functions below return None / empty lists / partial results
— callers (including future modules) MUST treat an absent row as
"not yet ingested", never assume or backfill a value. Use
get_ingestion_status() to check current coverage before relying on this
service for a given mine or date range.

Future modules (Environmental Knowledge Base, Influence Quantification,
Prediction, Explainable AI, Scenario Simulator, Smart Insights) should import
from this file rather than querying weather_records / environmental_features /
derived_environmental_features directly — this keeps the schema details
encapsulated and gives Module 2 a stable contract.

All functions return plain dict / list[dict] (JSON-serialisable), suitable
for direct use as FastAPI response bodies.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def is_mine_environmentally_mapped(db_path: str, mine_code: int) -> bool:
    """True if this mine has coordinates and therefore participates in the
    Environmental Variable Layer. Use this gate before calling the other
    functions below, so callers degrade gracefully for unmapped mines."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT is_mapped FROM mine_master WHERE mine_code = ?", (mine_code,))
        row = cur.fetchone()
        return bool(row and row["is_mapped"] == 1)
    finally:
        con.close()


def get_daily_environmental_snapshot(db_path: str, mine_code: int, date: str) -> Optional[dict]:
    """
    Full single-day environmental picture for one mine: raw weather +
    same-day features + rolling/derived features, joined into one record.
    Returns None if no data exists for that mine/date (e.g. unmapped mine,
    or date outside ingested range).
    """
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT w.*, e.temperature_range_c, e.dew_spread_c, e.thermal_stress_index,
                   d.drying_potential, d.environmental_risk_index, d.weather_stability_index,
                   d.consecutive_wet_days, d.consecutive_dry_days, d.moisture_accumulation_index,
                   d.rolling_rainfall_3d_mm, d.rolling_rainfall_7d_mm,
                   d.rolling_humidity_7d_pct, d.rolling_solar_radiation_7d_mj_m2
            FROM weather_records w
            LEFT JOIN environmental_features e ON e.mine_code = w.mine_code AND e.date = w.date
            LEFT JOIN derived_environmental_features d ON d.mine_code = w.mine_code AND d.date = w.date
            WHERE w.mine_code = ? AND w.date = ?
            """,
            (mine_code, date),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def get_environmental_timeseries(db_path: str, mine_code: int, start_date: str, end_date: str) -> list[dict]:
    """Full joined daily timeseries for one mine over a date range — the
    primary input for charts (Weather Intelligence page), correlation
    analysis (Module 3), and SHAP feature tables (Module 5)."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT w.mine_code, w.date,
                   w.temperature_max_c, w.temperature_min_c, w.temperature_mean_c,
                   w.relative_humidity_mean_pct, w.rainfall_mm, w.dew_point_mean_c,
                   w.wind_speed_mean_kmh, w.wind_gust_max_kmh, w.surface_pressure_mean_hpa,
                   w.cloud_cover_mean_pct, w.visibility_mean_km, w.solar_radiation_mj_m2,
                   w.weather_code, w.source, w.is_synthetic,
                   e.temperature_range_c, e.dew_spread_c, e.thermal_stress_index,
                   d.drying_potential, d.environmental_risk_index, d.weather_stability_index,
                   d.consecutive_wet_days, d.consecutive_dry_days, d.moisture_accumulation_index,
                   d.rolling_rainfall_3d_mm, d.rolling_rainfall_7d_mm,
                   d.rolling_humidity_7d_pct, d.rolling_solar_radiation_7d_mj_m2
            FROM weather_records w
            LEFT JOIN environmental_features e ON e.mine_code = w.mine_code AND e.date = w.date
            LEFT JOIN derived_environmental_features d ON d.mine_code = w.mine_code AND d.date = w.date
            WHERE w.mine_code = ? AND w.date BETWEEN ? AND ?
            ORDER BY w.date
            """,
            (mine_code, start_date, end_date),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_latest_environmental_risk(db_path: str, mine_code: int) -> Optional[dict]:
    """Most recent day's risk snapshot for a mine — used by Overview KPI
    strip / Mine Intelligence panel ('Environmental Status' block)."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT d.mine_code, d.date, d.environmental_risk_index, d.drying_potential,
                   d.weather_stability_index, d.consecutive_wet_days, d.consecutive_dry_days,
                   w.relative_humidity_mean_pct, w.rainfall_mm
            FROM derived_environmental_features d
            JOIN weather_records w ON w.mine_code = d.mine_code AND w.date = d.date
            WHERE d.mine_code = ?
            ORDER BY d.date DESC
            LIMIT 1
            """,
            (mine_code,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def get_environmental_risk_for_mapped_mines(db_path: str) -> list[dict]:
    """Latest ERI snapshot for every mapped mine — feeds the National Map /
    risk distribution / high-risk-mines table on the Overview page."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT m.mine_code, m.mine_name, m.subsidiary, m.latitude, m.longitude,
                   latest.date, latest.environmental_risk_index, latest.drying_potential,
                   latest.weather_stability_index
            FROM mine_master m
            JOIN (
                SELECT d.mine_code, d.date, d.environmental_risk_index, d.drying_potential,
                       d.weather_stability_index,
                       ROW_NUMBER() OVER (PARTITION BY d.mine_code ORDER BY d.date DESC) AS rn
                FROM derived_environmental_features d
            ) latest ON latest.mine_code = m.mine_code AND latest.rn = 1
            WHERE m.is_mapped = 1
            ORDER BY latest.environmental_risk_index DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_dominant_environmental_driver(db_path: str, mine_code: int, date: str) -> Optional[dict]:
    """
    Lightweight, non-ML 'dominant driver' for a given mine/day: ranks the
    same-day environmental signals by deviation from that mine's own 90-day
    trailing average, and returns the variable with the largest deviation.

    This is intentionally simple rule-based logic (NOT SHAP) — it exists so
    the dashboard can show a 'Dominant Driver' chip even before Module 5
    (Explainable AI) is built. Module 5 should supersede this with a proper
    SHAP-based driver once ML predictions exist, but this function remains
    useful as a fast, always-available fallback.
    """
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT date, relative_humidity_mean_pct AS humidity, rainfall_mm AS rainfall,
                   cloud_cover_mean_pct AS cloud_cover, solar_radiation_mj_m2 AS solar_radiation,
                   wind_speed_mean_kmh AS wind_speed, dew_point_mean_c AS dew_point
            FROM weather_records
            WHERE mine_code = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 90
            """,
            (mine_code, date),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if len(rows) < 2:
            return None

        today = rows[0]
        history = rows[1:]
        variables = ["humidity", "rainfall", "cloud_cover", "solar_radiation", "wind_speed", "dew_point"]

        best_var, best_z = None, -1.0
        for v in variables:
            vals = [r[v] for r in history if r[v] is not None]
            if len(vals) < 5 or today[v] is None:
                continue
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = variance ** 0.5
            if std == 0:
                continue
            z = abs(today[v] - mean) / std
            if z > best_z:
                best_z = z
                best_var = v

        if best_var is None:
            return None

        return {
            "mine_code": mine_code,
            "date": date,
            "dominant_driver": best_var,
            "z_score": round(best_z, 2),
            "method": "rule_based_zscore_90d (Module 2 fallback — superseded by SHAP in Module 5)",
        }
    finally:
        con.close()


def get_ingestion_status(db_path: str) -> dict:
    """Coverage/health summary of the Environmental Variable Layer — useful
    for an admin/Settings page and for Module 3+ to sanity-check before
    running correlation/ML jobs."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM weather_records")
        total_weather_rows = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(DISTINCT mine_code) AS c FROM weather_records")
        mines_with_weather = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM mine_master WHERE is_mapped = 1")
        total_mapped_mines = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM weather_records WHERE is_synthetic = 1")
        synthetic_rows = cur.fetchone()["c"]

        cur.execute("SELECT MIN(date) AS lo, MAX(date) AS hi FROM weather_records")
        date_row = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS c FROM environmental_features")
        env_feature_rows = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM derived_environmental_features")
        derived_feature_rows = cur.fetchone()["c"]

        cur.execute(
            "SELECT status, COUNT(*) AS c FROM weather_api_metadata GROUP BY status"
        )
        ingestion_status_counts = {r["status"]: r["c"] for r in cur.fetchall()}

        return {
            "total_mapped_mines": total_mapped_mines,
            "mines_with_weather_data": mines_with_weather,
            "weather_records_rows": total_weather_rows,
            "synthetic_rows": synthetic_rows,
            "real_api_rows": total_weather_rows - synthetic_rows,
            "date_range": {"start": date_row["lo"], "end": date_row["hi"]},
            "environmental_features_rows": env_feature_rows,
            "derived_environmental_features_rows": derived_feature_rows,
            "ingestion_run_status_counts": ingestion_status_counts,
        }
    finally:
        con.close()
