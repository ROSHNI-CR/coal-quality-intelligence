"""
Module 3 -- Environmental Knowledge Base
MILESTONE 3: Evidence providers.

The Recommendation Engine never reads raw tables itself -- it asks for
EVIDENCE through this module, which is a collection of small provider
functions, each responsible for ONE category of evidence. This is the
"modular, not fixed IF-ELSE" structure requested:

  Available today:
    - current_conditions   (Module 2: weather_records + environmental_features
                             + derived_environmental_features, via
                             environmental_service.py)
    - weather_history       (Module 2: trailing window, also via
                             environmental_service.py)
    - knowledge_base        (Module 3: environmental_knowledge_base +
                             environmental_variable_influence)
    - data_quality           (derived locally: is data present or pending?)

  Not yet available -- explicit stubs, always returning a clearly-marked
  "not_available" result rather than fabricating a value:
    - future_weather          (would come from a forecast API integration)
    - influence_quantification (Module 4 -- not built)
    - prediction_confidence    (Module 5 -- not built)
    - explainable_ai           (Module 5 -- not built)

EvidenceContext.available_sources is the single source of truth the rule
engine consults to decide whether a rule CAN be evaluated and how much to
discount its confidence if some of its declared evidence_sources are
missing. Adding a real implementation for any "not yet available" provider
later requires NO change to the rule engine or to existing rules -- only
this file changes, and rules whose evidence_sources already named that
category automatically start using it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..environmental import environmental_service as es  # Module 2 service layer


@dataclass
class EvidenceContext:
    mine_code: int
    date: str

    # --- available now ---
    current_conditions: Optional[dict] = None      # from get_daily_environmental_snapshot
    weather_history: Optional[list] = None         # from get_environmental_timeseries (trailing window)
    knowledge_base_loaded: bool = False
    data_quality_status: Optional[dict] = None      # {"weather_data_present": bool, "reason": str}

    # --- not yet available (explicit, always None until those modules exist) ---
    future_weather: Optional[dict] = None
    influence_quantification: Optional[dict] = None
    prediction_confidence: Optional[dict] = None
    explainable_ai: Optional[dict] = None

    available_sources: set = field(default_factory=set)
    unavailable_sources: set = field(default_factory=set)


ALL_KNOWN_EVIDENCE_SOURCES = {
    "current_conditions", "weather_history", "knowledge_base", "data_quality",
    "future_weather", "influence_quantification", "prediction_confidence", "explainable_ai",
}


def get_current_conditions(db_path: str, mine_code: int, date: str) -> Optional[dict]:
    """Module 2 same-day snapshot. Returns None if not yet ingested."""
    return es.get_daily_environmental_snapshot(db_path, mine_code, date)


def get_weather_history(db_path: str, mine_code: int, date: str, window_days: int = 14) -> list:
    """Module 2 trailing window. Returns [] if nothing ingested for the period."""
    from datetime import date as d, timedelta
    end = d.fromisoformat(date)
    start = (end - timedelta(days=window_days)).isoformat()
    return es.get_environmental_timeseries(db_path, mine_code, start, date)


def get_data_quality_status(db_path: str, mine_code: int, date: str, current_conditions: Optional[dict]) -> dict:
    """
    Local, always-available evidence: do we actually HAVE weather data for
    this mine/date, or is it pending production ingestion (per the Module 2
    production-only policy -- no synthetic substitute)?
    """
    if current_conditions is not None:
        return {"weather_data_present": True, "reason": "weather_records row exists for this mine/date"}
    return {
        "weather_data_present": False,
        "reason": "No weather_records row for this mine/date -- pending production Open-Meteo ingestion "
                  "(no synthetic substitute is used per platform policy)",
    }


# ---------------------------------------------------------------------------
# Not-yet-available providers -- explicit stubs.
# Each returns None and is never asked to fabricate a value. The comment on
# each documents exactly which future module supplies it.
# ---------------------------------------------------------------------------

def get_future_weather(mine_code: int, date: str) -> Optional[dict]:
    """Forecast weather (e.g. Open-Meteo Forecast API) -- not yet integrated.
    Reserved for when the platform adds forward-looking sampling guidance."""
    return None


def get_influence_quantification(mine_code: int, variable_name: str, target_metric: str) -> Optional[dict]:
    """Module 4 (Influence Quantification Engine) -- not yet built. Would
    return the statistically-validated, mine-specific influence strength
    for (variable_name, target_metric), upgrading the generic Knowledge
    Base hypothesis into an evidence-backed, mine-specific number."""
    return None


def get_prediction_confidence(mine_code: int, date: str) -> Optional[dict]:
    """Module 5 (ML Prediction layer) -- not yet built. Would return the
    model's confidence in its GCV/Moisture/Ash prediction for this mine/date."""
    return None


def get_explainable_ai_output(mine_code: int, date: str) -> Optional[dict]:
    """Module 5 (Explainable AI / SHAP) -- not yet built. Would return the
    mine-specific dominant-driver explanation combining ML output with the
    Knowledge Base's generic rationale."""
    return None


def build_evidence_context(db_path: str, mine_code: int, date: str) -> EvidenceContext:
    """
    Assembles everything CURRENTLY available into one EvidenceContext.
    This is the only function the rule engine calls -- it has no idea which
    individual providers exist behind it, so adding/upgrading a provider
    later never requires changing the engine.
    """
    current = get_current_conditions(db_path, mine_code, date)
    history = get_weather_history(db_path, mine_code, date)
    quality = get_data_quality_status(db_path, mine_code, date, current)

    ctx = EvidenceContext(
        mine_code=mine_code,
        date=date,
        current_conditions=current,
        weather_history=history,
        knowledge_base_loaded=True,  # Module 3 KB is always available (static, no ingestion dependency)
        data_quality_status=quality,
        future_weather=get_future_weather(mine_code, date),
        influence_quantification=None,  # category-level; per-variable lookups happen in the engine if needed
        prediction_confidence=get_prediction_confidence(mine_code, date),
        explainable_ai=get_explainable_ai_output(mine_code, date),
    )

    available = {"knowledge_base"}  # always available
    if current is not None:
        available.add("current_conditions")
    if history:
        available.add("weather_history")
    available.add("data_quality")  # the STATUS is always available even if it says "missing"
    if ctx.future_weather is not None:
        available.add("future_weather")
    if ctx.prediction_confidence is not None:
        available.add("prediction_confidence")
    if ctx.explainable_ai is not None:
        available.add("explainable_ai")
    # influence_quantification availability is checked per-rule/per-variable, not globally

    ctx.available_sources = available
    ctx.unavailable_sources = ALL_KNOWN_EVIDENCE_SOURCES - available
    return ctx
