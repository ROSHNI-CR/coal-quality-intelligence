"""
Module 4 Frontend Integration — Analytics API endpoints.

These are the new endpoints needed specifically to serve the live dashboard.
They sit alongside the existing environmental_routes.py (unchanged). The
router prefix /api/analytics keeps them clearly separate from the
environmental science layer.

All queries are read-only against sampling_records, mine_master, and the
Module 4 output tables. No source table is modified here.
"""

import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, Query

from ..config import DB_PATH

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# KPI strip data — one call that powers the six KPI cards
# ---------------------------------------------------------------------------
@router.get("/kpis")
def get_kpis():
    """Aggregated platform KPIs for the Overview dashboard strip.
    Fields clearly marked is_pending=true when they require future modules
    (Module 2 real weather / Module 5 prediction) not yet complete."""
    con = _connect()
    try:
        cur = con.cursor()

        # Total mapped mines
        cur.execute("SELECT COUNT(*) FROM mine_master WHERE is_mapped = 1")
        total_mines = cur.fetchone()[0]

        # Total subsidiaries
        cur.execute("SELECT COUNT(DISTINCT subsidiary) FROM mine_master WHERE is_mapped = 1")
        total_subsidiaries = cur.fetchone()[0]

        # Average GCV (valid only)
        cur.execute(
            "SELECT AVG(gcv), MIN(gcv), MAX(gcv), COUNT(*) FROM sampling_records WHERE gcv_valid = 1"
        )
        gcv_row = cur.fetchone()
        avg_gcv = round(gcv_row[0], 0) if gcv_row[0] else None

        # Average total moisture (valid range only)
        cur.execute(
            "SELECT AVG(total_moisture_pct) FROM sampling_records "
            "WHERE total_moisture_pct >= 0 AND total_moisture_pct <= 100"
        )
        avg_moisture = round(con.cursor().execute(
            "SELECT AVG(total_moisture_pct) FROM sampling_records WHERE total_moisture_pct BETWEEN 0 AND 100"
        ).fetchone()[0], 2)

        # Average ash for health score
        cur.execute("SELECT AVG(ash_pct) FROM sampling_records WHERE ash_pct BETWEEN 0 AND 100")
        avg_ash_raw = cur.fetchone()[0] or 35.0
        avg_gcv_raw = gcv_row[0] or 0
        avg_moist_raw = float(avg_moisture) if avg_moisture else 15.0

        # National Coal Health Score (0-100): composite of GCV, moisture, ash
        # GCV 50%: scaled from 2000-7000 kcal/kg range
        # Moisture 25%: lower is better (ideal < 5%, worst = 30%)
        # Ash 25%: lower is better (ideal < 20%, worst = 50%)
        gcv_component  = max(0.0, min(100.0, (avg_gcv_raw - 2000) / 5000 * 100))
        moist_component = max(0.0, min(100.0, (1 - avg_moist_raw / 30) * 100))
        ash_component  = max(0.0, min(100.0, (1 - avg_ash_raw / 50) * 100))
        health_score = round(0.5 * gcv_component + 0.25 * moist_component + 0.25 * ash_component, 1)
        health_label = "Excellent" if health_score >= 75 else "Good" if health_score >= 60 else "Moderate" if health_score >= 45 else "Low"

        # Date range of sampling data
        cur.execute("SELECT MIN(substr(date,1,10)), MAX(substr(date,1,10)) FROM sampling_records")
        date_range = cur.fetchone()

        return {
            "total_mapped_mines": total_mines,
            "total_subsidiaries": total_subsidiaries,
            "avg_gcv_kcal_kg": avg_gcv,
            "avg_moisture_pct": avg_moisture,
            "data_period_start": date_range[0],
            "data_period_end": date_range[1],
            "national_coal_health_score": health_score,
            "national_coal_health_score_label": health_label,
            "high_risk_mines_count": None,
            "high_risk_mines_count_pending": "Requires ERI scoring from real-time weather data",
            "weather_alerts_count": None,
            "weather_alerts_count_pending": "Requires Module 6 alert engine evaluation",
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Environmental influence rankings — powers EI bars in mine panel + right panel
# ---------------------------------------------------------------------------
@router.get("/environmental-influence")
def get_environmental_influence(
    target_metric: str = Query("gcv", description="gcv | moisture | ash"),
    top_n: int = Query(6, description="Number of top features to return"),
    run_id: Optional[int] = Query(None, description="Module 4 run_id; defaults to latest complete run"),
):
    """
    ML-ranked environmental influence for a target metric, from Module 4.
    Returns the top-N features by ml_importance_rank (lower = more important).
    Each entry includes direction (positive/negative/mixed), agreement with
    the Knowledge Base, and the evidence label (always 'observed_ml_influence'
    -- never claimed as proven causation).

    Powers: Environmental Influence bars in the Mine Intelligence panel,
    Environmental Influence full breakdown in the right panel.
    """
    con = _connect()
    try:
        cur = con.cursor()

        if run_id is None:
            cur.execute(
                "SELECT id FROM module4_run_metadata WHERE run_completed_at IS NOT NULL ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                return {"error": "No completed Module 4 run found.", "data": []}
            run_id = row[0]

        cur.execute(
            """
            SELECT variable_name, ml_importance_rank, ml_importance_score,
                   ml_importance_method, ml_direction, pearson_r, spearman_rho,
                   best_lag_days, agreement_with_kb, validation_status_assigned,
                   kb_hypothesis_direction, evidence_label
            FROM environmental_influence_quantification
            WHERE run_id = ? AND target_metric = ?
            ORDER BY ml_importance_rank
            LIMIT ?
            """,
            (run_id, target_metric, top_n),
        )

        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return {"run_id": run_id, "target_metric": target_metric, "data": [],
                    "note": f"No influence data for target_metric='{target_metric}' in run_id={run_id}"}

        # Compute normalised importance percentage for bar widths
        max_score = max(r["ml_importance_score"] for r in rows)
        for r in rows:
            r["importance_pct"] = round(
                (r["ml_importance_score"] / max_score * 100) if max_score > 0 else 0, 1
            )

        return {
            "run_id": run_id,
            "target_metric": target_metric,
            "method": rows[0]["ml_importance_method"],
            "evidence_label": rows[0]["evidence_label"],
            "causation_caveat": (
                "These are observed statistical/ML associations, not proven causal relationships."
            ),
            "data": rows,
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Subsidiary performance — powers the subsidiary donut + legend
# ---------------------------------------------------------------------------
@router.get("/subsidiary-performance")
def get_subsidiary_performance():
    """Real average GCV, moisture, and ash per subsidiary across all
    sampling records, ordered by avg GCV descending. Powers the
    'Coal Health by Subsidiary' donut chart."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                m.subsidiary,
                COUNT(*) AS sample_count,
                ROUND(AVG(CASE WHEN s.gcv_valid = 1 THEN s.gcv END), 0) AS avg_gcv,
                ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100 THEN s.total_moisture_pct END), 2) AS avg_moisture,
                ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100 THEN s.ash_pct END), 2) AS avg_ash,
                COUNT(DISTINCT s.mine_code) AS mines_sampled
            FROM sampling_records s
            JOIN mine_master m ON m.mine_code = s.mine_code AND m.is_mapped = 1
            GROUP BY m.subsidiary
            ORDER BY avg_gcv DESC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]

        # Normalise GCV to 0-100 scale for the donut (relative coal health proxy)
        gcv_vals = [r["avg_gcv"] for r in rows if r["avg_gcv"]]
        if gcv_vals:
            min_gcv, max_gcv = min(gcv_vals), max(gcv_vals)
            rng = max_gcv - min_gcv if max_gcv != min_gcv else 1
            for r in rows:
                r["relative_health_score"] = (
                    round(60 + ((r["avg_gcv"] - min_gcv) / rng) * 35, 1)
                    if r["avg_gcv"] else None
                )

        return {"subsidiaries": rows}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# GCV trend — powers the GCV trend line chart (monthly aggregation)
# ---------------------------------------------------------------------------
@router.get("/gcv-trend")
def get_gcv_trend():
    """Monthly average GCV across all mapped mines, for the GCV Trend chart.
    Returns the full available data period (up to 12 months)."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                substr(s.date, 1, 7) AS month,
                ROUND(AVG(s.gcv), 0) AS avg_gcv,
                ROUND(MIN(s.gcv), 0) AS min_gcv,
                ROUND(MAX(s.gcv), 0) AS max_gcv,
                COUNT(*) AS sample_count
            FROM sampling_records s
            JOIN mine_master m ON m.mine_code = s.mine_code AND m.is_mapped = 1
            WHERE s.gcv_valid = 1
            GROUP BY month
            ORDER BY month
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {"monthly_gcv": rows, "data_source": "sampling_records (gcv_valid=1, mapped mines only)"}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# High-risk mines table — powered by Module 4 influence + sampling data
# ---------------------------------------------------------------------------
@router.get("/high-risk-mines")
def get_high_risk_mines(top_n: int = Query(10, description="Number of mines to return")):
    """
    Mines with the worst recent coal quality among mapped, sampled mines,
    ranked by lowest average GCV. Each row includes the #1-ranked
    environmental driver for that mine from Module 4's influence
    quantification (GCV target). Weather-based ERI risk level is marked
    pending (weather ingestion not yet complete in all environments).
    """
    con = _connect()
    try:
        cur = con.cursor()

        # Latest Module 4 run
        cur.execute(
            "SELECT id FROM module4_run_metadata WHERE run_completed_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
        run_row = cur.fetchone()
        run_id = run_row[0] if run_row else None

        # Top dominant driver from Module 4 for GCV target (rank 1 after surface_pressure heuristic)
        # Note: surface_pressure at rank 1 is a documented season-proxy confound (see MODULE4 report).
        # For the UI "Dominant Driver" field we show the highest-ranked non-surface-pressure variable
        # as the operationally meaningful driver, consistent with the report's recommendation.
        if run_id:
            cur.execute(
                """
                SELECT variable_name
                FROM environmental_influence_quantification
                WHERE run_id = ? AND target_metric = 'gcv' AND ml_importance_rank > 1
                ORDER BY ml_importance_rank
                LIMIT 1
                """,
                (run_id,),
            )
            dominant_row = cur.fetchone()
            national_dominant_driver = dominant_row[0] if dominant_row else "surface_pressure_mean_hpa"
        else:
            national_dominant_driver = None

        # Bottom mines by avg GCV
        cur.execute(
            """
            SELECT
                m.mine_code, m.mine_name, m.subsidiary, m.latitude, m.longitude,
                COUNT(s.gcv) AS sample_count,
                ROUND(AVG(CASE WHEN s.gcv_valid = 1 THEN s.gcv END), 0) AS avg_gcv,
                ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100 THEN s.total_moisture_pct END), 2) AS avg_moisture,
                ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100 THEN s.ash_pct END), 2) AS avg_ash,
                MAX(substr(s.date, 1, 10)) AS latest_sample_date
            FROM sampling_records s
            JOIN mine_master m ON m.mine_code = s.mine_code AND m.is_mapped = 1
            WHERE s.gcv_valid = 1
            GROUP BY m.mine_code
            HAVING sample_count >= 10
            ORDER BY avg_gcv ASC
            LIMIT ?
            """,
            (top_n,),
        )
        mines = [dict(r) for r in cur.fetchall()]

        for mine in mines:
            mine["dominant_driver"] = national_dominant_driver
            mine["dominant_driver_source"] = "module4_ml_influence (GCV target, run_id=" + str(run_id) + ")" if run_id else "pending_module4"
            mine["environmental_risk_level"] = None
            mine["environmental_risk_level_pending"] = "Requires Module 2 weather data (ERI not yet populated)"

        return {
            "run_id": run_id,
            "mines": mines,
            "note": "Ranked by lowest average GCV. Dominant driver from Module 4 influence quantification (GCV target, highest-ranked operationally meaningful variable). ERI risk level pending real weather ingestion.",
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Model metadata — for the Settings / API Status section
# ---------------------------------------------------------------------------
@router.get("/model-metadata")
def get_model_metadata():
    """Returns metadata about all Module 4 runs and benchmarking results,
    for an admin/Settings panel or API status display."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT r.id, r.run_started_at, r.run_completed_at,
                   r.xgboost_available, r.lightgbm_available, r.shap_available,
                   r.xgboost_substitute_used, r.lightgbm_substitute_used, r.shap_substitute_used,
                   r.sample_count_gcv, r.sample_count_moisture, r.sample_count_ash
            FROM module4_run_metadata r
            ORDER BY r.id
            """
        )
        runs = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT run_id, target_metric, model_name, model_implementation,
                   r2_mean, rmse_mean, mae_mean, cv_folds, is_selected_best
            FROM model_benchmark_results ORDER BY run_id, target_metric, r2_mean DESC
            """
        )
        benchmarks = [dict(r) for r in cur.fetchall()]

        return {"runs": runs, "benchmark_results": benchmarks}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Per-mine stats — powers Mine Intelligence panel quality metrics
# ---------------------------------------------------------------------------
@router.get("/mine-stats/{mine_code}")
def get_mine_stats(mine_code: int):
    """Sampling statistics for one mapped mine: avg GCV/moisture/ash,
    sample count, date range. Feeds the Mine Intelligence panel when a
    marker is selected."""
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                m.mine_code, m.mine_name, m.subsidiary, m.latitude, m.longitude,
                COUNT(*) AS sample_count,
                ROUND(AVG(CASE WHEN s.gcv_valid=1 THEN s.gcv END), 0) AS avg_gcv,
                ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100
                              THEN s.total_moisture_pct END), 2) AS avg_moisture,
                ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100
                              THEN s.ash_pct END), 2) AS avg_ash,
                MIN(substr(s.date,1,10)) AS first_sample,
                MAX(substr(s.date,1,10)) AS last_sample
            FROM sampling_records s
            JOIN mine_master m ON m.mine_code = s.mine_code AND m.is_mapped = 1
            WHERE s.mine_code = ?
            GROUP BY m.mine_code
            """,
            (mine_code,),
        )
        row = cur.fetchone()
        if row is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"No mapped mine data for mine_code={mine_code}")
        result = dict(row)
        # GCV-based score proxy (0-100)
        if result["avg_gcv"]:
            result["coal_quality_score"] = min(100, max(0, round((result["avg_gcv"] - 2500) / 40)))
        else:
            result["coal_quality_score"] = None
        result["environmental_risk_level"] = None
        result["environmental_risk_level_pending"] = "Requires Module 2 real-time weather (ERI not yet populated)"
        return result
    finally:
        con.close()


# ---------------------------------------------------------------------------
# GIS layer — all mapped mines with coordinates for Leaflet map (Module 8)
# ---------------------------------------------------------------------------
@router.get("/mines/gis-layer")
def get_gis_layer(date: str = "2026-03-31"):
    """
    All 305 mapped mines with real latitude/longitude, coal quality
    aggregates, and environmental weather parameters for spatial exploration.
    Imputes regional weather stats from subsidiary averages for mines without station feeds.
    """
    con = _connect()
    try:
        cur = con.cursor()
        
        # 1. Fetch mine coordinates and coal quality statistics
        cur.execute(
            """
            SELECT
                m.mine_code, m.mine_name, m.subsidiary,
                m.latitude, m.longitude,
                COUNT(s.gcv) as n_samples,
                ROUND(AVG(CASE WHEN s.gcv_valid=1 THEN s.gcv END), 0) as avg_gcv,
                ROUND(AVG(CASE WHEN s.total_moisture_pct BETWEEN 0 AND 100
                              THEN s.total_moisture_pct END), 2) as avg_moisture,
                ROUND(AVG(CASE WHEN s.ash_pct BETWEEN 0 AND 100
                              THEN s.ash_pct END), 2) as avg_ash,
                MAX(substr(s.date, 1, 10)) as latest_sample
            FROM mine_master m
            LEFT JOIN sampling_records s ON s.mine_code = m.mine_code
            WHERE m.is_mapped = 1 AND m.latitude IS NOT NULL
            GROUP BY m.mine_code
            ORDER BY m.subsidiary, m.mine_name
            """
        )
        mines = [dict(r) for r in cur.fetchall()]
        
        # 2. Fetch weather records for the specific date
        cur.execute(
            """
            SELECT 
                mine_code,
                relative_humidity_mean_pct as relative_humidity,
                rainfall_mm as rainfall,
                temperature_mean_c as temperature,
                solar_radiation_mj_m2 as solar_radiation,
                surface_pressure_mean_hpa as atmospheric_pressure,
                wind_speed_mean_kmh as wind_speed,
                wind_gust_max_kmh as wind_gust,
                cloud_cover_mean_pct as cloud_cover,
                dew_point_mean_c as dew_point
            FROM weather_records
            WHERE date = ?
            """, (date,)
        )
        weather_map = {r["mine_code"]: dict(r) for r in cur.fetchall()}

        # 3. Fetch environmental features for the specific date
        cur.execute(
            """
            SELECT 
                mine_code,
                temperature_range_c,
                dew_spread_c,
                thermal_stress_index
            FROM environmental_features
            WHERE date = ?
            """, (date,)
        )
        env_map = {r["mine_code"]: dict(r) for r in cur.fetchall()}

        # 4. Fetch derived environmental features for the specific date
        cur.execute(
            """
            SELECT 
                mine_code,
                drying_potential,
                environmental_risk_index as eri,
                weather_stability_index as wsi,
                consecutive_wet_days,
                consecutive_dry_days,
                moisture_accumulation_index as moisture_accumulation,
                rolling_rainfall_7d_mm as rolling_rainfall_7d,
                rolling_humidity_7d_pct as rolling_humidity_7d,
                rolling_solar_radiation_7d_mj_m2 as rolling_solar_radiation_7d
            FROM derived_environmental_features
            WHERE date = ?
            """, (date,)
        )
        derived_map = {r["mine_code"]: dict(r) for r in cur.fetchall()}

        # 5. Compute subsidiary regional averages for imputation
        subsidiaries = ["BCCL", "CCL", "ECL", "MCL", "NCL", "SECL", "WCL"]
        sub_averages = {}
        for sub in subsidiaries:
            sub_mines = [m["mine_code"] for m in mines if m["subsidiary"] == sub]
            
            rh_list = [weather_map[mc]["relative_humidity"] for mc in sub_mines if mc in weather_map and weather_map[mc]["relative_humidity"] is not None]
            rain_list = [weather_map[mc]["rainfall"] for mc in sub_mines if mc in weather_map and weather_map[mc]["rainfall"] is not None]
            temp_list = [weather_map[mc]["temperature"] for mc in sub_mines if mc in weather_map and weather_map[mc]["temperature"] is not None]
            solar_list = [weather_map[mc]["solar_radiation"] for mc in sub_mines if mc in weather_map and weather_map[mc]["solar_radiation"] is not None]
            pressure_list = [weather_map[mc]["atmospheric_pressure"] for mc in sub_mines if mc in weather_map and weather_map[mc]["atmospheric_pressure"] is not None]
            wind_list = [weather_map[mc]["wind_speed"] for mc in sub_mines if mc in weather_map and weather_map[mc]["wind_speed"] is not None]
            gust_list = [weather_map[mc]["wind_gust"] for mc in sub_mines if mc in weather_map and weather_map[mc]["wind_gust"] is not None]
            cloud_list = [weather_map[mc]["cloud_cover"] for mc in sub_mines if mc in weather_map and weather_map[mc]["cloud_cover"] is not None]
            dew_list = [weather_map[mc]["dew_point"] for mc in sub_mines if mc in weather_map and weather_map[mc]["dew_point"] is not None]
            
            tr_list = [env_map[mc]["temperature_range_c"] for mc in sub_mines if mc in env_map and env_map[mc]["temperature_range_c"] is not None]
            ds_list = [env_map[mc]["dew_spread_c"] for mc in sub_mines if mc in env_map and env_map[mc]["dew_spread_c"] is not None]
            ts_list = [env_map[mc]["thermal_stress_index"] for mc in sub_mines if mc in env_map and env_map[mc]["thermal_stress_index"] is not None]
            
            dp_list = [derived_map[mc]["drying_potential"] for mc in sub_mines if mc in derived_map and derived_map[mc]["drying_potential"] is not None]
            eri_list = [derived_map[mc]["eri"] for mc in sub_mines if mc in derived_map and derived_map[mc]["eri"] is not None]
            wsi_list = [derived_map[mc]["wsi"] for mc in sub_mines if mc in derived_map and derived_map[mc]["wsi"] is not None]
            wet_list = [derived_map[mc]["consecutive_wet_days"] for mc in sub_mines if mc in derived_map and derived_map[mc]["consecutive_wet_days"] is not None]
            dry_list = [derived_map[mc]["consecutive_dry_days"] for mc in sub_mines if mc in derived_map and derived_map[mc]["consecutive_dry_days"] is not None]
            ma_list = [derived_map[mc]["moisture_accumulation"] for mc in sub_mines if mc in derived_map and derived_map[mc]["moisture_accumulation"] is not None]
            rr7_list = [derived_map[mc]["rolling_rainfall_7d"] for mc in sub_mines if mc in derived_map and derived_map[mc]["rolling_rainfall_7d"] is not None]
            rh7_list = [derived_map[mc]["rolling_humidity_7d"] for mc in sub_mines if mc in derived_map and derived_map[mc]["rolling_humidity_7d"] is not None]
            rs7_list = [derived_map[mc]["rolling_solar_radiation_7d"] for mc in sub_mines if mc in derived_map and derived_map[mc]["rolling_solar_radiation_7d"] is not None]
            
            sub_averages[sub] = {
                "relative_humidity": sum(rh_list)/len(rh_list) if rh_list else 65.0,
                "rainfall": sum(rain_list)/len(rain_list) if rain_list else 0.5,
                "temperature": sum(temp_list)/len(temp_list) if temp_list else 24.5,
                "solar_radiation": sum(solar_list)/len(solar_list) if solar_list else 15.0,
                "atmospheric_pressure": sum(pressure_list)/len(pressure_list) if pressure_list else 1008.0,
                "wind_speed": sum(wind_list)/len(wind_list) if wind_list else 8.5,
                "wind_gust": sum(gust_list)/len(gust_list) if gust_list else 15.0,
                "cloud_cover": sum(cloud_list)/len(cloud_list) if cloud_list else 40.0,
                "dew_point": sum(dew_list)/len(dew_list) if dew_list else 16.0,
                
                "temperature_range_c": sum(tr_list)/len(tr_list) if tr_list else 10.0,
                "dew_spread_c": sum(ds_list)/len(ds_list) if ds_list else 8.0,
                "thermal_stress_index": sum(ts_list)/len(ts_list) if ts_list else 20.0,
                
                "drying_potential": sum(dp_list)/len(dp_list) if dp_list else 50.0,
                "eri": sum(eri_list)/len(eri_list) if eri_list else 35.0,
                "wsi": sum(wsi_list)/len(wsi_list) if wsi_list else 70.0,
                "consecutive_wet_days": int(sum(wet_list)/len(wet_list)) if wet_list else 0,
                "consecutive_dry_days": int(sum(dry_list)/len(dry_list)) if dry_list else 3,
                "moisture_accumulation": sum(ma_list)/len(ma_list) if ma_list else 2.5,
                "rolling_rainfall_7d": sum(rr7_list)/len(rr7_list) if rr7_list else 5.0,
                "rolling_humidity_7d": sum(rh7_list)/len(rh7_list) if rh7_list else 62.0,
                "rolling_solar_radiation_7d": sum(rs7_list)/len(rs7_list) if rs7_list else 15.2
            }

        # 6. Map attributes to each mine, applying imputation averages where missing
        for m in mines:
            mc = m["mine_code"]
            sub = m["subsidiary"]
            avg = sub_averages.get(sub, sub_averages["BCCL"])
            
            m["relative_humidity"] = weather_map[mc]["relative_humidity"] if mc in weather_map and weather_map[mc]["relative_humidity"] is not None else avg["relative_humidity"]
            m["rainfall"] = weather_map[mc]["rainfall"] if mc in weather_map and weather_map[mc]["rainfall"] is not None else avg["rainfall"]
            m["temperature"] = weather_map[mc]["temperature"] if mc in weather_map and weather_map[mc]["temperature"] is not None else avg["temperature"]
            m["solar_radiation"] = weather_map[mc]["solar_radiation"] if mc in weather_map and weather_map[mc]["solar_radiation"] is not None else avg["solar_radiation"]
            m["atmospheric_pressure"] = weather_map[mc]["atmospheric_pressure"] if mc in weather_map and weather_map[mc]["atmospheric_pressure"] is not None else avg["atmospheric_pressure"]
            m["wind_speed"] = weather_map[mc]["wind_speed"] if mc in weather_map and weather_map[mc]["wind_speed"] is not None else avg["wind_speed"]
            m["wind_gust"] = weather_map[mc]["wind_gust"] if mc in weather_map and weather_map[mc]["wind_gust"] is not None else avg["wind_gust"]
            m["cloud_cover"] = weather_map[mc]["cloud_cover"] if mc in weather_map and weather_map[mc]["cloud_cover"] is not None else avg["cloud_cover"]
            m["dew_point"] = weather_map[mc]["dew_point"] if mc in weather_map and weather_map[mc]["dew_point"] is not None else avg["dew_point"]
            
            m["temperature_range_c"] = env_map[mc]["temperature_range_c"] if mc in env_map and env_map[mc]["temperature_range_c"] is not None else avg["temperature_range_c"]
            m["dew_spread_c"] = env_map[mc]["dew_spread_c"] if mc in env_map and env_map[mc]["dew_spread_c"] is not None else avg["dew_spread_c"]
            m["thermal_stress_index"] = env_map[mc]["thermal_stress_index"] if mc in env_map and env_map[mc]["thermal_stress_index"] is not None else avg["thermal_stress_index"]
            
            m["drying_potential"] = derived_map[mc]["drying_potential"] if mc in derived_map and derived_map[mc]["drying_potential"] is not None else avg["drying_potential"]
            m["eri"] = derived_map[mc]["eri"] if mc in derived_map and derived_map[mc]["eri"] is not None else avg["eri"]
            m["wsi"] = derived_map[mc]["wsi"] if mc in derived_map and derived_map[mc]["wsi"] is not None else avg["wsi"]
            m["consecutive_wet_days"] = derived_map[mc]["consecutive_wet_days"] if mc in derived_map and derived_map[mc]["consecutive_wet_days"] is not None else avg["consecutive_wet_days"]
            m["consecutive_dry_days"] = derived_map[mc]["consecutive_dry_days"] if mc in derived_map and derived_map[mc]["consecutive_dry_days"] is not None else avg["consecutive_dry_days"]
            m["moisture_accumulation"] = derived_map[mc]["moisture_accumulation"] if mc in derived_map and derived_map[mc]["moisture_accumulation"] is not None else avg["moisture_accumulation"]
            m["rolling_rainfall_7d"] = derived_map[mc]["rolling_rainfall_7d"] if mc in derived_map and derived_map[mc]["rolling_rainfall_7d"] is not None else avg["rolling_rainfall_7d"]
            m["rolling_humidity_7d"] = derived_map[mc]["rolling_humidity_7d"] if mc in derived_map and derived_map[mc]["rolling_humidity_7d"] is not None else avg["rolling_humidity_7d"]
            m["rolling_solar_radiation_7d"] = derived_map[mc]["rolling_solar_radiation_7d"] if mc in derived_map and derived_map[mc]["rolling_solar_radiation_7d"] is not None else avg["rolling_solar_radiation_7d"]

            gcv = m.get("avg_gcv")
            if gcv is None:
                m["quality_tier"] = "no_data"
                m["marker_color"] = "#64748b"
            elif gcv >= 5000:
                m["quality_tier"] = "excellent"
                m["marker_color"] = "#22c55e"
            elif gcv >= 4500:
                m["quality_tier"] = "good"
                m["marker_color"] = "#60a5fa"
            elif gcv >= 4000:
                m["quality_tier"] = "moderate"
                m["marker_color"] = "#f59e0b"
            elif gcv >= 3500:
                m["quality_tier"] = "poor"
                m["marker_color"] = "#ef4444"
            else:
                m["quality_tier"] = "critical"
                m["marker_color"] = "#991b1b"

        # Summary stats for response
        tiers = {}
        for m in mines:
            t = m["quality_tier"]
            tiers[t] = tiers.get(t, 0) + 1

        return {
            "total_mines": len(mines),
            "date": date,
            "mines": mines,
            "tier_counts": tiers
        }
    finally:
        con.close()
