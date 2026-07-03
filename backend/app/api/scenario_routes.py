"""Module 7 -- Scenario Simulator API"""
import os
from fastapi import APIRouter, Query
from ..services.scenario.scenario_simulator import run_scenario, get_scenario_input_schema
from dataclasses import asdict

from ..config import DB_PATH
router = APIRouter(prefix="/api/scenario", tags=["scenario"])

@router.get("/schema")
def scenario_schema():
    """Input schema: field names, labels, units, min/max for the UI form."""
    return {"inputs": get_scenario_input_schema()}

@router.post("/{mine_code}")
def run(mine_code: int, body: dict, label: str = Query("Custom Scenario")):
    """Run a what-if scenario. body: {feature_name: value, ...}"""
    result = run_scenario(DB_PATH, mine_code, body, label)
    return {
        "mine_code": result.mine_code,
        "mine_name": result.mine_name,
        "scenario_label": result.scenario_label,
        "missing_filled_from": result.missing_filled_from,
        "scenario_inputs": result.scenario_inputs,
        "predictions": result.predictions,
        "baseline_predictions": result.baseline_predictions,
        "delta": result.delta,
    }

@router.get("/{mine_code}/presets")
def preset_scenarios(mine_code: int):
    """Run 3 built-in presets (Heavy Monsoon / Ideal Dry / Current Baseline)."""
    presets = {
        "heavy_monsoon": {
            "relative_humidity_mean_pct": 90.0,
            "rainfall_mm": 30.0,
            "solar_radiation_mj_m2": 8.0,
            "cloud_cover_mean_pct": 90.0,
            "temperature_mean_c": 28.0,
        },
        "ideal_drying": {
            "relative_humidity_mean_pct": 28.0,
            "rainfall_mm": 0.0,
            "solar_radiation_mj_m2": 28.0,
            "wind_speed_mean_kmh": 18.0,
            "cloud_cover_mean_pct": 15.0,
        },
        "current_baseline": {},  # empty = 100% real baseline data
    }
    results = {}
    for name, inputs in presets.items():
        r = run_scenario(DB_PATH, mine_code, inputs, name)
        results[name] = {
            "predictions": r.predictions,
            "delta": r.delta,
            "filled_from": r.missing_filled_from,
        }
    return {"mine_code": mine_code, "scenarios": results}
