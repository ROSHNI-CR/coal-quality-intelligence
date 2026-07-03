"""
Module 5 -- Prediction Engine
Prediction service.

Public interface consumed by FastAPI endpoints and the frontend.
Scope (per instruction): historical/backtest predictions only.
Refuses predictions when required weather features are unavailable.
Returns point estimate, prediction interval, confidence label,
and full model/data provenance metadata. No explanation or
recommendation narrative -- that is Module 6.

Design rules enforced here:
  - No synthetic data, no fallback values. If weather features
    for a mine/date are not in the database, the prediction is
    refused with a clear reason, not silently approximated.
  - Confidence label derives from the model's GroupKFold CV R²
    (honest, leakage-free estimate), not training-set performance.
  - Every prediction (success or refusal) is logged to the
    `predictions` table for audit and backtest review.
  - Prediction intervals use honest OOF residual quantiles
    stored in prediction_models at deployment time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..influence_quantification.data_assembly import FEATURE_COLUMNS

MODELS_DIR = None  # resolved from DB at runtime


@dataclass
class PredictionResult:
    mine_code: int
    date: str
    target_metric: str
    prediction_status: str          # 'success' | 'refused_missing_weather' | 'refused_unmapped_mine' | 'refused_incomplete_features'
    point_estimate: Optional[float]
    interval_lower: Optional[float]
    interval_upper: Optional[float]
    confidence_label: Optional[str] # 'high' | 'medium' | 'low'
    actual_value: Optional[float]   # from sampling_records if available (backtest)
    model_id: Optional[int]
    model_name: Optional[str]
    model_implementation: Optional[str]
    cv_r2: Optional[float]
    refusal_reason: Optional[str]
    features_used: Optional[list]
    is_backtest: bool = True


def _confidence_label(r2: float) -> str:
    if r2 >= 0.40:
        return "medium"   # R²=0.34-0.46 range -- honest about limits
    if r2 >= 0.25:
        return "low"
    return "very_low"


def _load_active_model(con: sqlite3.Connection, target_metric: str) -> Optional[dict]:
    import os
    cur = con.cursor()
    cur.execute(
        """SELECT id, target_metric, model_name, model_implementation, model_artifact_path,
                  feature_columns_json, cv_r2_mean, cv_rmse_mean, cv_mae_mean,
                  interval_lower_quantile, interval_upper_quantile,
                  residual_lower_offset, residual_upper_offset, random_state
           FROM prediction_models WHERE target_metric=? AND is_active=1 ORDER BY id DESC LIMIT 1""",
        (target_metric,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    m = dict(zip(cols, row))
    m["feature_columns"] = json.loads(m["feature_columns_json"])

    # Resolve local path if database path is not valid
    path = m["model_artifact_path"]
    if not os.path.exists(path):
        filename = os.path.basename(path)
        local_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
        local_path = os.path.join(local_models_dir, filename)
        if os.path.exists(local_path):
            m["model_artifact_path"] = local_path

    return m


def _load_features(con: sqlite3.Connection, mine_code: int, date: str, feature_columns: list) -> Optional[pd.DataFrame]:
    """Load weather + derived features for a specific mine/date."""
    cur = con.cursor()
    # Check mine is mapped
    cur.execute("SELECT is_mapped FROM mine_master WHERE mine_code=?", (mine_code,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, "refused_unmapped_mine", "Mine is not mapped (no coordinates) and therefore has no weather data."

    # Join weather + environmental_features + derived_environmental_features
    raw_weather = [c for c in feature_columns if c in [
        "temperature_max_c","temperature_min_c","temperature_mean_c",
        "relative_humidity_mean_pct","relative_humidity_max_pct","relative_humidity_min_pct",
        "rainfall_mm","dew_point_mean_c","wind_speed_mean_kmh","wind_gust_max_kmh",
        "surface_pressure_mean_hpa","cloud_cover_mean_pct","solar_radiation_mj_m2",
    ]]
    env_features = [c for c in feature_columns if c in [
        "temperature_range_c","dew_spread_c","thermal_stress_index",
    ]]
    derived = [c for c in feature_columns if c in [
        "drying_potential","environmental_risk_index","weather_stability_index",
        "consecutive_wet_days","consecutive_dry_days","moisture_accumulation_index",
        "rolling_rainfall_3d_mm","rolling_rainfall_7d_mm","rolling_humidity_7d_pct",
        "rolling_solar_radiation_7d_mj_m2",
    ]]

    w_sel = ", ".join(f"w.{c}" for c in raw_weather) if raw_weather else "1"
    e_sel = ", ".join(f"e.{c}" for c in env_features) if env_features else "1"
    d_sel = ", ".join(f"d.{c}" for c in derived) if derived else "1"

    query = f"""
        SELECT {w_sel}, {e_sel}, {d_sel}
        FROM weather_records w
        LEFT JOIN environmental_features e ON e.mine_code=w.mine_code AND e.date=w.date
        LEFT JOIN derived_environmental_features d ON d.mine_code=w.mine_code AND d.date=w.date
        WHERE w.mine_code=? AND w.date=?
    """
    cur.execute(query, (mine_code, date))
    row = cur.fetchone()
    if not row:
        return None, "refused_missing_weather", (
            f"No weather data for mine_code={mine_code} on {date}. "
            "Production Open-Meteo ingestion must cover this mine/date before prediction is possible. "
            "No synthetic fallback will be used."
        )

    all_cols = raw_weather + env_features + derived
    values = list(row)[:len(all_cols)]
    df = pd.DataFrame([values], columns=all_cols)

    # Reorder to match training feature column order exactly
    df = df.reindex(columns=feature_columns)

    # Check for missing features
    missing = df.columns[df.isnull().any()].tolist()
    if missing:
        return None, "refused_incomplete_features", (
            f"Feature(s) {missing} are NULL for mine_code={mine_code} on {date}. "
            "Cannot produce a reliable prediction with incomplete feature set."
        )

    return df, "ok", None


def _get_actual(con: sqlite3.Connection, mine_code: int, date: str, target_metric: str) -> Optional[float]:
    """Look up actual observed value from sampling_records if it exists."""
    target_col = {"gcv": "gcv", "moisture": "total_moisture_pct", "ash": "ash_pct"}[target_metric]
    cur = con.cursor()
    validity_clause = {
        "gcv": "AND gcv_valid=1 AND gcv BETWEEN 0 AND 9000",
        "moisture": "AND total_moisture_pct BETWEEN 0 AND 100",
        "ash": "AND ash_pct BETWEEN 0 AND 100",
    }[target_metric]
    cur.execute(
        f"SELECT AVG({target_col}) FROM sampling_records "
        f"WHERE mine_code=? AND substr(date,1,10)=? {validity_clause}",
        (mine_code, date),
    )
    row = cur.fetchone()
    return round(float(row[0]), 2) if row and row[0] is not None else None


def _log_prediction(con: sqlite3.Connection, result: PredictionResult) -> int:
    cur = con.cursor()
    cur.execute(
        """INSERT INTO predictions
           (mine_code, date, target_metric, prediction_model_id, prediction_status,
            point_estimate, interval_lower, interval_upper, confidence_label,
            actual_value, is_backtest, refusal_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (result.mine_code, result.date, result.target_metric, result.model_id,
         result.prediction_status, result.point_estimate, result.interval_lower,
         result.interval_upper, result.confidence_label, result.actual_value,
         int(result.is_backtest), result.refusal_reason),
    )
    con.commit()
    return cur.lastrowid


def predict(db_path: str, mine_code: int, date: str, target_metric: str,
            log: bool = True) -> PredictionResult:
    """
    Generate a prediction for a specific mine/date/target.
    Historical/backtest only: refuses if weather data unavailable.
    """
    if target_metric not in ("gcv", "moisture", "ash"):
        raise ValueError(f"target_metric must be 'gcv', 'moisture', or 'ash', got {target_metric!r}")

    con = sqlite3.connect(db_path)
    try:
        model_meta = _load_active_model(con, target_metric)
        if not model_meta:
            result = PredictionResult(
                mine_code=mine_code, date=date, target_metric=target_metric,
                prediction_status="refused_no_model",
                point_estimate=None, interval_lower=None, interval_upper=None,
                confidence_label=None,
                actual_value=_get_actual(con, mine_code, date, target_metric),
                model_id=None, model_name=None, model_implementation=None, cv_r2=None,
                refusal_reason=f"No active prediction model for target_metric='{target_metric}'. Run model deployment first.",
            )
            if log:
                _log_prediction(con, result)
            return result

        features_df, status, reason = _load_features(
            con, mine_code, date, model_meta["feature_columns"]
        )

        if status != "ok":
            result = PredictionResult(
                mine_code=mine_code, date=date, target_metric=target_metric,
                prediction_status=status,
                point_estimate=None, interval_lower=None, interval_upper=None,
                confidence_label=None,
                actual_value=_get_actual(con, mine_code, date, target_metric),
                model_id=model_meta["id"], model_name=model_meta["model_name"],
                model_implementation=model_meta["model_implementation"],
                cv_r2=model_meta["cv_r2_mean"], refusal_reason=reason,
                features_used=None,
            )
            if log:
                _log_prediction(con, result)
            return result

        fitted_model = joblib.load(model_meta["model_artifact_path"])
        point = float(fitted_model.predict(features_df)[0])

        # Apply OOF residual interval
        lower = round(point + model_meta["residual_lower_offset"], 2)
        upper = round(point + model_meta["residual_upper_offset"], 2)
        point = round(point, 2)

        # Clamp to physically valid ranges
        ranges = {"gcv": (0, 9000), "moisture": (0, 100), "ash": (0, 100)}
        lo, hi = ranges[target_metric]
        point = max(lo, min(hi, point))
        lower = max(lo, min(hi, lower))
        upper = max(lo, min(hi, upper))

        actual = _get_actual(con, mine_code, date, target_metric)
        confidence = _confidence_label(model_meta["cv_r2_mean"])

        result = PredictionResult(
            mine_code=mine_code, date=date, target_metric=target_metric,
            prediction_status="success",
            point_estimate=point, interval_lower=lower, interval_upper=upper,
            confidence_label=confidence,
            actual_value=actual,
            model_id=model_meta["id"], model_name=model_meta["model_name"],
            model_implementation=model_meta["model_implementation"],
            cv_r2=round(model_meta["cv_r2_mean"], 4),
            refusal_reason=None,
            features_used=model_meta["feature_columns"],
        )
        if log:
            _log_prediction(con, result)
        return result
    finally:
        con.close()


def predict_all_targets(db_path: str, mine_code: int, date: str, log: bool = True) -> dict:
    """Convenience: predict GCV, moisture, and ash in one call."""
    return {
        t: asdict(predict(db_path, mine_code, date, t, log=log))
        for t in ("gcv", "moisture", "ash")
    }


def get_prediction_history(db_path: str, mine_code: int = None,
                            target_metric: str = None, limit: int = 100) -> list:
    """Query the prediction history log."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        clauses, params = [], []
        if mine_code:
            clauses.append("mine_code=?"); params.append(mine_code)
        if target_metric:
            clauses.append("target_metric=?"); params.append(target_metric)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = con.cursor()
        cur.execute(
            f"SELECT * FROM predictions {where} ORDER BY requested_at DESC LIMIT ?",
            params + [limit],
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_deployed_models(db_path: str) -> list:
    """Return metadata for all deployed prediction models."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """SELECT id, target_metric, model_name, model_implementation,
                      cv_r2_mean, cv_rmse_mean, cv_mae_mean,
                      interval_lower_quantile, interval_upper_quantile,
                      residual_lower_offset, residual_upper_offset,
                      training_sample_size, random_state, is_active, trained_at, notes
               FROM prediction_models ORDER BY target_metric, id"""
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
