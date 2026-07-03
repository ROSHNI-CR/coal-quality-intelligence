"""
Module 3 -- Environmental Knowledge Base
MILESTONE 3: Rule Engine.

Evaluates the rules stored in environmental_recommendation_rules /
environmental_recommendation_conditions against an EvidenceContext, and
produces Recommendation objects. This engine is deliberately NOT a chain
of fixed Python if/elif statements -- every rule's trigger logic, supporting
variables, rationale, priority, category and confidence are DATA (read from
the database), and this engine is a generic interpreter over that data.
Adding, editing, or disabling a rule never requires touching this file.

Confidence handling
--------------------
Each rule declares evidence_sources it conceptually depends on (e.g.
"current_conditions,weather_history,knowledge_base"). At evaluation time,
the engine checks how many of those sources were actually available in the
EvidenceContext and downgrades the rule's static confidence_level by one
step for each missing declared source (capped at 'low'). This is what makes
the framework "enrichable" -- a rule written today that lists
'influence_quantification' as a future dependency will automatically regain
full confidence the day that provider starts returning real data, with zero
code change here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .evidence_providers import EvidenceContext

_CONFIDENCE_ORDER = ["low", "medium", "high"]


@dataclass
class Recommendation:
    rule_id: int
    rule_name: str
    triggered: bool
    trigger_condition: str
    supporting_variables: list
    scientific_rationale: str
    recommendation_category: str
    recommendation_priority: str
    expected_operational_benefit: str
    recommendation_text: str
    base_confidence: str
    adjusted_confidence: str
    evidence_available: list
    evidence_missing: list
    depends_on_modules: list
    recommendation_basis: list = field(default_factory=list)
    # ^ Validation requirement (Milestone 6): every recommendation must clearly
    # report which evidence categories it draws on, classified into the
    # five buckets the platform recognises:
    #   'knowledge_base'        -- uses Module 3 KB/influence rationale
    #   'weather'                -- uses current_conditions / weather_history / future_weather
    #   'data_quality'           -- uses the data-availability check itself
    #   'ml_driven'              -- uses influence_quantification / prediction_confidence (Module 4/5, not yet built)
    #   'explainable_ai_driven'  -- uses explainable_ai output (Module 5, not yet built)
    # This is computed from the rule's DECLARED evidence_sources (not just
    # what happened to be available this call), so the basis reflects the
    # rule's design intent and stays stable even when data is missing.
    not_yet_evaluable: bool = False
    not_evaluable_reason: Optional[str] = None


_BASIS_MAP = {
    "knowledge_base": "knowledge_base",
    "current_conditions": "weather",
    "weather_history": "weather",
    "future_weather": "weather",
    "data_quality": "data_quality",
    "influence_quantification": "ml_driven",
    "prediction_confidence": "ml_driven",
    "explainable_ai": "explainable_ai_driven",
}


def _classify_basis(declared_sources: set) -> list:
    basis = sorted({_BASIS_MAP[s] for s in declared_sources if s in _BASIS_MAP})
    return basis


def _load_active_rules(con: sqlite3.Connection) -> list[dict]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, variable_name, rule_name, condition_summary, condition_field, condition_operator,
               condition_value, condition_value_secondary, severity, operational_stage,
               recommendation_text, recommendation_category, recommendation_priority,
               expected_operational_benefit, confidence_level, is_multi_variable,
               evidence_sources, depends_on_modules, scientific_references
        FROM environmental_recommendation_rules
        WHERE is_active = 1
        ORDER BY id
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_conditions(con: sqlite3.Connection, rule_id: int) -> list[dict]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT variable_name, condition_field, condition_operator, condition_value,
               condition_value_secondary, logical_connector, sequence_order
        FROM environmental_recommendation_conditions
        WHERE rule_id = ?
        ORDER BY sequence_order
        """,
        (rule_id,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _evaluate_single_condition(value: Optional[float], operator: str, threshold: Optional[float],
                                 threshold2: Optional[float]) -> bool:
    if value is None:
        return False
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    if operator == "between":
        return threshold <= value <= threshold2
    return False


def _get_field_value(field_name: str, ctx: EvidenceContext) -> Optional[float]:
    """Looks up a field's current value from the EvidenceContext's
    current_conditions snapshot. Returns None if unavailable (rule then
    cannot be evaluated, and is reported as not_yet_evaluable rather than
    silently treated as 'condition not met')."""
    if ctx.current_conditions is None:
        return None
    return ctx.current_conditions.get(field_name)


def _confidence_downgrade(base: str, n_steps: int) -> str:
    try:
        idx = _CONFIDENCE_ORDER.index(base)
    except ValueError:
        idx = 1  # default to medium if unrecognised
    idx = max(0, idx - n_steps)
    return _CONFIDENCE_ORDER[idx]


def evaluate_rule(rule: dict, conditions: list[dict], ctx: EvidenceContext) -> Recommendation:
    declared_sources = set((rule["evidence_sources"] or "").split(",")) if rule["evidence_sources"] else set()
    declared_sources = {s.strip() for s in declared_sources if s.strip()}

    available = declared_sources & ctx.available_sources
    missing = declared_sources - ctx.available_sources

    depends_on = [m.strip() for m in (rule["depends_on_modules"] or "").split(",") if m.strip()]

    # Special case: data_quality_caveat rules evaluate the ABSENCE of data
    # itself -- they are always evaluable, since "is data present?" is
    # always knowable.
    if rule["recommendation_category"] == "data_quality_caveat" and rule["condition_field"] == "data_availability":
        weather_present = ctx.data_quality_status.get("weather_data_present", False) if ctx.data_quality_status else False
        triggered = not weather_present
        return Recommendation(
            rule_id=rule["id"], rule_name=rule["rule_name"], triggered=triggered,
            trigger_condition=rule["condition_summary"],
            supporting_variables=[rule["variable_name"]],
            scientific_rationale=rule["recommendation_text"],
            recommendation_category=rule["recommendation_category"],
            recommendation_priority=rule["recommendation_priority"],
            expected_operational_benefit=rule["expected_operational_benefit"],
            recommendation_text=rule["recommendation_text"] if triggered else "",
            base_confidence=rule["confidence_level"], adjusted_confidence=rule["confidence_level"],
            evidence_available=sorted(available), evidence_missing=sorted(missing),
            depends_on_modules=depends_on,
            recommendation_basis=_classify_basis(declared_sources),
        )

    # Multi-variable rules: evaluate each condition row, combine via logical_connector (AND/OR, left-to-right)
    if rule["is_multi_variable"]:
        if not conditions:
            return Recommendation(
                rule_id=rule["id"], rule_name=rule["rule_name"], triggered=False,
                trigger_condition=rule["condition_summary"], supporting_variables=[],
                scientific_rationale="", recommendation_category=rule["recommendation_category"],
                recommendation_priority=rule["recommendation_priority"],
                expected_operational_benefit=rule["expected_operational_benefit"],
                recommendation_text="", base_confidence=rule["confidence_level"],
                adjusted_confidence=rule["confidence_level"], evidence_available=sorted(available),
                evidence_missing=sorted(missing), depends_on_modules=depends_on,
                recommendation_basis=_classify_basis(declared_sources),
                not_yet_evaluable=True, not_evaluable_reason="Rule marked multi-variable but has no conditions defined.",
            )

        if ctx.current_conditions is None:
            return Recommendation(
                rule_id=rule["id"], rule_name=rule["rule_name"], triggered=False,
                trigger_condition=rule["condition_summary"],
                supporting_variables=[c["variable_name"] for c in conditions],
                scientific_rationale="", recommendation_category=rule["recommendation_category"],
                recommendation_priority=rule["recommendation_priority"],
                expected_operational_benefit=rule["expected_operational_benefit"],
                recommendation_text="", base_confidence=rule["confidence_level"],
                adjusted_confidence=rule["confidence_level"], evidence_available=sorted(available),
                evidence_missing=sorted(missing), depends_on_modules=depends_on,
                recommendation_basis=_classify_basis(declared_sources),
                not_yet_evaluable=True,
                not_evaluable_reason="No current_conditions available for this mine/date (weather data pending).",
            )

        result = None
        for cond in conditions:
            value = _get_field_value(cond["condition_field"], ctx)
            this_result = _evaluate_single_condition(
                value, cond["condition_operator"], cond["condition_value"], cond["condition_value_secondary"]
            )
            if result is None:
                result = this_result
            elif cond["logical_connector"] == "OR":
                result = result or this_result
            else:  # default AND
                result = result and this_result

        triggered = bool(result)
    else:
        # single-variable legacy path
        if ctx.current_conditions is None:
            return Recommendation(
                rule_id=rule["id"], rule_name=rule["rule_name"], triggered=False,
                trigger_condition=rule["condition_summary"], supporting_variables=[rule["variable_name"]],
                scientific_rationale="", recommendation_category=rule["recommendation_category"],
                recommendation_priority=rule["recommendation_priority"],
                expected_operational_benefit=rule["expected_operational_benefit"],
                recommendation_text="", base_confidence=rule["confidence_level"],
                adjusted_confidence=rule["confidence_level"], evidence_available=sorted(available),
                evidence_missing=sorted(missing), depends_on_modules=depends_on,
                recommendation_basis=_classify_basis(declared_sources),
                not_yet_evaluable=True,
                not_evaluable_reason="No current_conditions available for this mine/date (weather data pending).",
            )
        value = _get_field_value(rule["condition_field"], ctx)
        triggered = _evaluate_single_condition(
            value, rule["condition_operator"], rule["condition_value"], rule["condition_value_secondary"]
        )

    n_missing = len(missing)
    adjusted_confidence = _confidence_downgrade(rule["confidence_level"], n_missing)

    supporting_vars = [c["variable_name"] for c in conditions] if conditions else [rule["variable_name"]]

    return Recommendation(
        rule_id=rule["id"], rule_name=rule["rule_name"], triggered=triggered,
        trigger_condition=rule["condition_summary"],
        supporting_variables=supporting_vars,
        scientific_rationale=rule["scientific_references"] or "",
        recommendation_category=rule["recommendation_category"],
        recommendation_priority=rule["recommendation_priority"],
        expected_operational_benefit=rule["expected_operational_benefit"],
        recommendation_text=rule["recommendation_text"] if triggered else "",
        base_confidence=rule["confidence_level"], adjusted_confidence=adjusted_confidence,
        evidence_available=sorted(available), evidence_missing=sorted(missing),
        depends_on_modules=depends_on,
        recommendation_basis=_classify_basis(declared_sources),
    )


def run_rules(db_path: str, ctx: EvidenceContext) -> list[Recommendation]:
    """Evaluate every active rule against the given EvidenceContext."""
    con = sqlite3.connect(db_path)
    try:
        rules = _load_active_rules(con)
        results = []
        for rule in rules:
            conditions = _load_conditions(con, rule["id"]) if rule["is_multi_variable"] else []
            results.append(evaluate_rule(rule, conditions, ctx))
        return results
    finally:
        con.close()
