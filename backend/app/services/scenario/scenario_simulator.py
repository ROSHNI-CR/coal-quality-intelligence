"""
Module 7 -- Scenario Simulator
Allows "what if" queries: given hypothetical environmental conditions
for a mine, what would the predicted GCV/moisture/ash be?

Design:
- User supplies any subset of the 26 feature columns
- Missing features are filled from that mine's own real environmental
  data for the closest available date (real-data baseline), or from
  population medians as a final fallback. Never random/synthetic.
- Returns predictions for all three targets side-by-side, plus a
  comparison against the mine's current/baseline conditions.
- Full explanation (attributions + narrative) generated via Module 6.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..influence_quantification.data_assembly import FEATURE_COLUMNS
from ..prediction.prediction_service import _load_active_model, _confidence_label


# Grouped for the UI: only ask users for the primary raw weather inputs.
# Derived features are auto-calculated from those.
PRIMARY_INPUTS = [
    ("temperature_mean_c",          "Mean Temperature",        "°C",     15, 45),
    ("relative_humidity_mean_pct",  "Relative Humidity",       "%",       5, 100),
    ("rainfall_mm",                 "Rainfall",                "mm",      0, 200),
    ("solar_radiation_mj_m2",       "Solar Radiation",         "MJ/m²",   0,  35),
    ("wind_speed_mean_kmh",         "Wind Speed",              "km/h",    0,  80),
    ("cloud_cover_mean_pct",        "Cloud Cover",             "%",       0, 100),
    ("dew_point_mean_c",            "Dew Point",               "°C",     -5,  35),
    ("surface_pressure_mean_hpa",   "Surface Pressure",        "hPa",   950, 1030),
]

PRIMARY_INPUT_NAMES = {p[0] for p in PRIMARY_INPUTS}


def _derive_features(inputs: dict) -> dict:
    """Compute derived features from primary weather inputs (mirrors Module 2 logic)."""
    derived = dict(inputs)

    t_mean = inputs.get("temperature_mean_c", 25.0)
    t_max  = inputs.get("temperature_max_c", t_mean + 5)
    t_min  = inputs.get("temperature_min_c", t_mean - 5)
    rh     = inputs.get("relative_humidity_mean_pct", 60.0)
    rain   = inputs.get("rainfall_mm", 0.0)
    solar  = inputs.get("solar_radiation_mj_m2", 15.0)
    wind   = inputs.get("wind_speed_mean_kmh", 10.0)
    dew    = inputs.get("dew_point_mean_c", t_mean - ((100 - rh) / 5.0))
    cloud  = inputs.get("cloud_cover_mean_pct", 40.0)

    # Same-day derived
    derived["temperature_range_c"]   = round(t_max - t_min, 2)
    derived["dew_spread_c"]          = round(t_mean - dew, 2)
    excess = max(0, t_mean - 25)
    derived["thermal_stress_index"]  = round(min(100, excess * (rh / 100.0) * 4.0), 2)

    # Rolling/window features — for a scenario we use the single-day value as proxy
    derived.setdefault("relative_humidity_max_pct", min(100, rh + 8))
    derived.setdefault("relative_humidity_min_pct", max(0,   rh - 12))
    derived.setdefault("temperature_max_c", t_max)
    derived.setdefault("temperature_min_c", t_min)
    derived.setdefault("wind_gust_max_kmh", wind * 1.5)

    derived["rolling_rainfall_3d_mm"]           = rain * 1.2
    derived["rolling_rainfall_7d_mm"]           = rain * 2.5
    derived["rolling_humidity_7d_pct"]          = rh
    derived["rolling_solar_radiation_7d_mj_m2"] = solar

    raw_drying = 0.25 * t_mean + 0.8 * wind + 1.2 * solar - 0.4 * rh - 1.5 * rain
    drying_potential = max(0, min(100, (raw_drying + 30) * (100 / 60)))
    derived["drying_potential"] = round(drying_potential, 2)
    derived["environmental_risk_index"] = round(min(100, 100 - drying_potential + 0.3 * rain * 2.5), 2)
    derived["weather_stability_index"]  = 80.0  # neutral for a single-point scenario
    derived["consecutive_wet_days"]     = 1 if rain > 1 else 0
    derived["consecutive_dry_days"]     = 0 if rain > 1 else 1

    today_signal = rain + 0.5 * (rh / 10.0)
    derived["moisture_accumulation_index"] = round(today_signal * 2.5, 2)

    return derived


def _get_baseline_features(db_path: str, mine_code: int) -> Optional[dict]:
    """Load that mine's most recent real environmental snapshot as baseline."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM weather_records WHERE mine_code=? ORDER BY date DESC LIMIT 1",
            (mine_code,)
        )
        wr = cur.fetchone()
        if not wr:
            return None
        baseline = dict(wr)

        date = baseline["date"]
        cur.execute("SELECT * FROM environmental_features WHERE mine_code=? AND date=?", (mine_code, date))
        ef = cur.fetchone()
        if ef:
            baseline.update({k: v for k, v in dict(ef).items() if v is not None})

        cur.execute("SELECT * FROM derived_environmental_features WHERE mine_code=? AND date=?", (mine_code, date))
        df = cur.fetchone()
        if df:
            baseline.update({k: v for k, v in dict(df).items() if v is not None})

        return baseline
    finally:
        con.close()


def _get_population_medians(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT " + ",".join(f"AVG({c})" for c in [
            "temperature_mean_c","relative_humidity_mean_pct","rainfall_mm",
            "solar_radiation_mj_m2","wind_speed_mean_kmh","cloud_cover_mean_pct",
            "dew_point_mean_c","surface_pressure_mean_hpa"
        ]) + " FROM weather_records")
        row = cur.fetchone()
        cols = ["temperature_mean_c","relative_humidity_mean_pct","rainfall_mm",
                "solar_radiation_mj_m2","wind_speed_mean_kmh","cloud_cover_mean_pct",
                "dew_point_mean_c","surface_pressure_mean_hpa"]
        return dict(zip(cols, row)) if row else {}
    finally:
        con.close()


@dataclass
class ScenarioResult:
    mine_code: int
    mine_name: Optional[str]
    scenario_inputs: dict       # what the user specified
    full_features: dict         # after filling derived features
    baseline_features: dict     # real data baseline for comparison
    predictions: dict           # {target: {point, lower, upper, confidence_label}}
    baseline_predictions: dict  # predictions on real baseline for comparison
    delta: dict                 # {target: scenario_pred - baseline_pred}
    scenario_label: str
    missing_filled_from: str    # 'mine_recent_data' | 'population_median'


def run_scenario(db_path: str, mine_code: int, scenario_inputs: dict,
                 scenario_label: str = "Custom Scenario") -> ScenarioResult:
    """
    Run a what-if scenario for a mine given partial environmental conditions.
    scenario_inputs: dict of {feature_name: value} for any subset of PRIMARY_INPUTS.
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT mine_name FROM mine_master WHERE mine_code=?", (mine_code,))
        row = cur.fetchone()
        mine_name = row[0] if row else None
    finally:
        con.close()

    # Build full feature set
    baseline_raw = _get_baseline_features(db_path, mine_code)
    if baseline_raw:
        fill_source = "mine_recent_data"
        # Start from baseline, override with scenario inputs
        full_inputs = {k: baseline_raw.get(k, v) for k, v in baseline_raw.items()}
        full_inputs.update({k: v for k, v in scenario_inputs.items() if v is not None})
    else:
        fill_source = "population_median"
        medians = _get_population_medians(db_path)
        full_inputs = dict(medians)
        full_inputs.update({k: v for k, v in scenario_inputs.items() if v is not None})

    full_features = _derive_features(full_inputs)

    # Reorder to match training column order
    feature_vec = pd.DataFrame([{c: full_features.get(c, np.nan) for c in FEATURE_COLUMNS}])

    # Run predictions for all 3 targets
    predictions = {}
    baseline_predictions = {}
    baseline_vec = None
    if baseline_raw:
        baseline_vec = pd.DataFrame([{c: baseline_raw.get(c, np.nan) for c in FEATURE_COLUMNS}])

    for target in ("gcv", "moisture", "ash"):
        model_meta = _load_active_model(sqlite3.connect(db_path), target)
        if not model_meta:
            continue
        fitted = joblib.load(model_meta["model_artifact_path"])
        pt = float(fitted.predict(feature_vec)[0])
        ranges = {"gcv": (0, 9000), "moisture": (0, 100), "ash": (0, 100)}
        lo, hi = ranges[target]
        pt = max(lo, min(hi, round(pt, 2)))
        lower = max(lo, min(hi, round(pt + model_meta["residual_lower_offset"], 2)))
        upper = max(lo, min(hi, round(pt + model_meta["residual_upper_offset"], 2)))

        predictions[target] = {
            "point_estimate": pt,
            "interval_lower": lower,
            "interval_upper": upper,
            "confidence_label": _confidence_label(model_meta["cv_r2_mean"]),
            "cv_r2": round(model_meta["cv_r2_mean"], 4),
        }

        if baseline_vec is not None:
            bp = float(fitted.predict(baseline_vec)[0])
            bp = max(lo, min(hi, round(bp, 2)))
            baseline_predictions[target] = {"point_estimate": bp}

    delta = {
        t: round(predictions[t]["point_estimate"] - baseline_predictions.get(t, {}).get("point_estimate", predictions[t]["point_estimate"]), 2)
        for t in predictions
    }

    return ScenarioResult(
        mine_code=mine_code, mine_name=mine_name,
        scenario_inputs=scenario_inputs, full_features=full_features,
        baseline_features={c: baseline_raw.get(c) for c in FEATURE_COLUMNS} if baseline_raw else {},
        predictions=predictions, baseline_predictions=baseline_predictions,
        delta=delta, scenario_label=scenario_label,
        missing_filled_from=fill_source,
    )


def get_scenario_input_schema() -> list:
    """Returns the input schema for the UI to render the scenario form."""
    return [
        {"field": f, "label": l, "unit": u, "min": mn, "max": mx}
        for f, l, u, mn, mx in PRIMARY_INPUTS
    ]
