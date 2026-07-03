"""
Module 6 -- Explainable AI
Natural language narrative generator.

Produces the four platform questions as structured text:
  1. What happened?    (observed coal quality vs historical average)
  2. Why?             (top attribution features + KB physical rationale)
  3. What's next?     (directional outlook from prediction + interval)
  4. What to do?      (triggered recommendation rules)

All text is assembled from structured data — Module 4 influence rankings,
Module 5 predictions, Module 3 Knowledge Base, and live weather features.
No LLM is called; every sentence is grounded in a specific data source
that can be traced and cited. This makes the narratives auditable,
consistent, and scientifically defensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .local_explainer import FeatureAttribution

VARIABLE_DISPLAY = {
    "surface_pressure_mean_hpa": "atmospheric pressure",
    "rolling_humidity_7d_pct": "7-day rolling humidity",
    "rolling_solar_radiation_7d_mj_m2": "7-day rolling solar radiation",
    "moisture_accumulation_index": "accumulated moisture index",
    "consecutive_wet_days": "consecutive wet days",
    "consecutive_dry_days": "consecutive dry days",
    "drying_potential": "drying potential",
    "environmental_risk_index": "environmental risk index",
    "rainfall_mm": "rainfall",
    "relative_humidity_mean_pct": "relative humidity",
    "dew_point_mean_c": "dew point",
    "temperature_mean_c": "temperature",
    "dew_spread_c": "dew spread",
    "cloud_cover_mean_pct": "cloud cover",
    "solar_radiation_mj_m2": "solar radiation",
    "wind_speed_mean_kmh": "wind speed",
    "rolling_rainfall_7d_mm": "7-day rolling rainfall",
    "weather_stability_index": "weather stability",
    "thermal_stress_index": "thermal stress",
    "temperature_range_c": "temperature range",
}

SKIP_AS_CONFOUND = {"surface_pressure_mean_hpa"}  # documented season proxy


def _vname(feat: str) -> str:
    return VARIABLE_DISPLAY.get(feat, feat.replace("_", " "))


def _direction_phrase(score: float, target_metric: str) -> str:
    if target_metric == "gcv":
        return "raised GCV" if score > 0 else "suppressed GCV"
    if target_metric == "moisture":
        return "increased moisture" if score > 0 else "reduced moisture"
    return "raised ash" if score > 0 else "reduced ash"


@dataclass
class NarrativeInsights:
    mine_code: int
    date: str
    target_metric: str
    what_happened: str
    why: str
    whats_next: str
    what_to_do: str
    confidence_note: str
    evidence_sources: list


def generate_narrative(
    mine_code: int,
    date: str,
    target_metric: str,
    point_estimate: Optional[float],
    interval_lower: Optional[float],
    interval_upper: Optional[float],
    confidence_label: str,
    actual_value: Optional[float],
    historical_avg: Optional[float],
    attributions: list[FeatureAttribution],
    active_rules: list[dict],
    mine_name: str = "This mine",
    cv_r2: float = 0.0,
) -> NarrativeInsights:

    unit = {"gcv": "kcal/kg", "moisture": "%", "ash": "%"}[target_metric]
    metric_label = {"gcv": "GCV", "moisture": "moisture", "ash": "ash content"}[target_metric]

    # ── WHAT HAPPENED ─────────────────────────────────────────────────────
    if actual_value is not None and historical_avg is not None:
        diff = actual_value - historical_avg
        direction = "above" if diff > 0 else "below"
        what_happened = (
            f"{mine_name} recorded {metric_label} of {actual_value:,.2f} {unit} on {date}, "
            f"which is {abs(diff):,.2f} {unit} {direction} its historical average "
            f"of {historical_avg:,.2f} {unit}."
        )
    elif actual_value is not None:
        what_happened = (
            f"{mine_name} recorded {metric_label} of {actual_value:,.2f} {unit} on {date}."
        )
    elif point_estimate is not None:
        what_happened = (
            f"No sampling record found for {mine_name} on {date}. "
            f"The model estimates {metric_label} at {point_estimate:,.2f} {unit} "
            f"based on environmental conditions."
        )
    else:
        what_happened = f"No {metric_label} data or prediction available for {date}."

    # ── WHY ───────────────────────────────────────────────────────────────
    meaningful = [
        a for a in attributions
        if a.feature_name not in SKIP_AS_CONFOUND and abs(a.attribution_score) > 0
    ][:3]

    if meaningful:
        parts = []
        for a in meaningful:
            vn = _vname(a.feature_name)
            dir_phrase = _direction_phrase(a.attribution_score, target_metric)
            val_note = f" (current: {a.feature_value:.1f})" if a.feature_value is not None else ""
            kb_note = f" {a.kb_physical_meaning[:100]}..." if a.kb_physical_meaning else ""
            validated = " [KB-validated]" if a.kb_validation_status == "validated" else ""
            parts.append(f"{vn.capitalize()}{val_note} {dir_phrase}{validated}.{kb_note}")

        why = "The primary environmental drivers identified by the model are: " + " ".join(parts)
        if any(a.feature_name in SKIP_AS_CONFOUND for a in attributions[:2]):
            why += " Note: atmospheric pressure ranked highly but is likely a seasonal proxy rather than a direct physical driver."
    else:
        why = ("Environmental attribution is available but no single dominant driver "
               "stands out clearly for this mine/date combination.")

    # ── WHAT'S NEXT ───────────────────────────────────────────────────────
    if point_estimate is not None and interval_lower is not None:
        whats_next = (
            f"Based on current environmental conditions, {metric_label} is estimated at "
            f"{point_estimate:,.2f} {unit} (80% prediction interval: "
            f"{interval_lower:,.2f}–{interval_upper:,.2f} {unit}). "
        )
        if confidence_label == "medium":
            whats_next += "Use as directional guidance; the model explains a moderate share of variance (GroupKFold R²≈0.35)."
        elif confidence_label == "low":
            whats_next += "Treat as indicative only — the model has low predictive power for this target at unseen mines."
        else:
            whats_next += "Model confidence is very low for this target; interval is wide and should be treated with caution."
    else:
        whats_next = "Prediction unavailable — weather data required for this mine/date."

    # ── WHAT TO DO ────────────────────────────────────────────────────────
    if active_rules:
        critical = [r for r in active_rules if r.get("severity") == "critical"]
        warnings = [r for r in active_rules if r.get("severity") == "warning"]
        actions = critical + warnings + [r for r in active_rules if r not in critical + warnings]
        what_to_do = " | ".join(r["recommendation_text"][:120] for r in actions[:3])
    else:
        what_to_do = (
            "No specific environmental alerts triggered for this mine/date. "
            "Continue standard sampling and monitoring procedures."
        )

    # ── CONFIDENCE NOTE ───────────────────────────────────────────────────
    confidence_note = (
        f"Model: LightGBM, GroupKFold CV R²={cv_r2:.3f}. "
        f"Explanations via {'SHAP TreeExplainer' if any(a.attribution_method == 'shap_tree' for a in attributions) else 'marginal contribution vs. population median'}. "
        f"All outputs are observed associations, not proven causal relationships."
    )

    return NarrativeInsights(
        mine_code=mine_code,
        date=date,
        target_metric=target_metric,
        what_happened=what_happened,
        why=why,
        whats_next=whats_next,
        what_to_do=what_to_do,
        confidence_note=confidence_note,
        evidence_sources=["module2_weather", "module3_kb", "module4_ml", "module5_prediction"],
    )
