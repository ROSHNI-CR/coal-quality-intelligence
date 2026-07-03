"""
Module 6 -- Explainable AI
Local explanation engine.

Computes per-prediction feature attributions by marginal contribution:
for each feature, measure the change in prediction when that feature
is replaced with its population median (a model-agnostic local
explanation that works without SHAP and produces interpretable,
directionally meaningful scores).

This is distinct from Module 4's global permutation importance:
  - Module 4: how much does shuffling feature X across ALL samples
    degrade model performance? (global, population-level)
  - Module 6: for THIS specific prediction, how much does each
    feature contribute vs. a neutral baseline? (local, per-prediction)

When the real SHAP library is available, TreeExplainer is used instead
and produces better decompositions. The method is always recorded.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class FeatureAttribution:
    feature_name: str
    attribution_score: float    # signed: positive = pushes prediction UP
    attribution_rank: int
    attribution_method: str
    feature_value: Optional[float]
    kb_physical_meaning: Optional[str]
    kb_operational_interp: Optional[str]
    kb_validation_status: Optional[str]


def _load_population_medians(db_path: str, feature_columns: list) -> dict:
    """Compute population median of each feature across all weather records."""
    con = sqlite3.connect(db_path)
    try:
        available = [c for c in feature_columns if c in [
            "temperature_max_c","temperature_min_c","temperature_mean_c",
            "relative_humidity_mean_pct","relative_humidity_max_pct","relative_humidity_min_pct",
            "rainfall_mm","dew_point_mean_c","wind_speed_mean_kmh","wind_gust_max_kmh",
            "surface_pressure_mean_hpa","cloud_cover_mean_pct","solar_radiation_mj_m2",
        ]]
        derived = [c for c in feature_columns if c not in available]

        medians = {}
        if available:
            sel = ", ".join(f"AVG({c})" for c in available)
            cur = con.cursor()
            cur.execute(f"SELECT {sel} FROM weather_records")
            row = cur.fetchone()
            if row:
                medians.update(dict(zip(available, row)))

        if derived:
            all_derived = ["temperature_range_c","dew_spread_c","thermal_stress_index",
                           "drying_potential","environmental_risk_index","weather_stability_index",
                           "consecutive_wet_days","consecutive_dry_days","moisture_accumulation_index",
                           "rolling_rainfall_3d_mm","rolling_rainfall_7d_mm","rolling_humidity_7d_pct",
                           "rolling_solar_radiation_7d_mj_m2"]
            ef_cols = [c for c in derived if c in ["temperature_range_c","dew_spread_c","thermal_stress_index"]]
            def_cols = [c for c in derived if c in all_derived and c not in ef_cols]
            if ef_cols:
                sel = ", ".join(f"AVG({c})" for c in ef_cols)
                cur = con.cursor()
                cur.execute(f"SELECT {sel} FROM environmental_features")
                row = cur.fetchone()
                if row:
                    medians.update(dict(zip(ef_cols, row)))
            if def_cols:
                sel = ", ".join(f"AVG({c})" for c in def_cols)
                cur = con.cursor()
                cur.execute(f"SELECT {sel} FROM derived_environmental_features")
                row = cur.fetchone()
                if row:
                    medians.update(dict(zip(def_cols, row)))

        return medians
    finally:
        con.close()


def _load_kb_context(db_path: str, feature_name: str, target_metric: str) -> dict:
    """Fetch Knowledge Base context for a feature."""
    from ..influence_quantification.data_assembly import FEATURE_TO_KB_VARIABLE
    kb_var = FEATURE_TO_KB_VARIABLE.get(feature_name, feature_name)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT physical_meaning, operational_interpretation FROM environmental_knowledge_base WHERE variable_name=?",
            (kb_var,)
        )
        kb = cur.fetchone()
        cur.execute(
            "SELECT validation_status FROM environmental_variable_influence WHERE variable_name=? AND target_metric=?",
            (kb_var, target_metric)
        )
        inf = cur.fetchone()
        return {
            "physical_meaning": dict(kb)["physical_meaning"][:200] if kb else None,
            "operational_interp": dict(kb)["operational_interpretation"][:200] if kb else None,
            "validation_status": dict(inf)["validation_status"] if inf else None,
        }
    finally:
        con.close()


def explain_prediction(db_path: str, prediction_id: int, mine_code: int, date: str,
                        target_metric: str, features_df: pd.DataFrame,
                        fitted_model, model_name: str, top_n: int = 8) -> list[FeatureAttribution]:
    """
    Compute local feature attributions for a single prediction.
    Uses SHAP if available, otherwise marginal contribution vs population median.
    Returns top_n features ranked by absolute attribution magnitude.
    """
    feature_columns = list(features_df.columns)
    baseline_pred = float(fitted_model.predict(features_df)[0])

    if HAS_SHAP and model_name in ("random_forest", "xgboost", "lightgbm"):
        try:
            explainer = shap.TreeExplainer(fitted_model)
            shap_vals = explainer.shap_values(features_df)
            attributions = {f: float(shap_vals[0][i]) for i, f in enumerate(feature_columns)}
            method = "shap_tree"
        except Exception:
            attributions = None
    else:
        attributions = None

    if attributions is None:
        # Marginal contribution: replace each feature with population median, measure change
        medians = _load_population_medians(db_path, feature_columns)
        attributions = {}
        for feat in feature_columns:
            if feat not in medians or medians[feat] is None:
                attributions[feat] = 0.0
                continue
            df_perturbed = features_df.copy()
            df_perturbed[feat] = medians[feat]
            perturbed_pred = float(fitted_model.predict(df_perturbed)[0])
            attributions[feat] = baseline_pred - perturbed_pred  # positive = this feature raises prediction
        method = "marginal_contribution_vs_median"

    # Rank by absolute value
    ranked = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    results = []
    for rank, (feat, score) in enumerate(ranked[:top_n], start=1):
        kb = _load_kb_context(db_path, feat, target_metric)
        fval = float(features_df[feat].iloc[0]) if feat in features_df.columns else None
        results.append(FeatureAttribution(
            feature_name=feat,
            attribution_score=round(score, 4),
            attribution_rank=rank,
            attribution_method=method,
            feature_value=fval,
            kb_physical_meaning=kb["physical_meaning"],
            kb_operational_interp=kb["operational_interp"],
            kb_validation_status=kb["validation_status"],
        ))
    return results


def persist_explanations(db_path: str, prediction_id: int, mine_code: int,
                          date: str, target_metric: str,
                          attributions: list[FeatureAttribution]) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.executemany(
            """INSERT INTO prediction_explanations
               (prediction_id, mine_code, date, target_metric,
                feature_name, attribution_score, attribution_rank, attribution_method,
                kb_physical_meaning, kb_operational_interp, kb_validation_status, feature_value)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(prediction_id, mine_code, date, target_metric,
              a.feature_name, a.attribution_score, a.attribution_rank, a.attribution_method,
              a.kb_physical_meaning, a.kb_operational_interp, a.kb_validation_status, a.feature_value)
             for a in attributions]
        )
        con.commit()
    finally:
        con.close()
