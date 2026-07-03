"""
Module 3 -- Environmental Knowledge Base
MILESTONE 3: Recommendation Engine orchestrator.

This is the single public entry point future modules and FastAPI endpoints
(Milestone 5) should call. It:
  1. Builds an EvidenceContext for the requested mine/date (evidence_providers.py).
  2. Runs every active rule against it (rule_engine.py).
  3. Returns only the TRIGGERED recommendations (plus, separately, a list of
     rules that could not yet be evaluated due to missing data, so callers
     can surface "N recommendations pending data" rather than silence).
  4. Optionally logs the full evaluation (triggered or not) to
     environmental_recommendation_log for audit/Alerts use later.

Nothing here performs machine learning, SHAP, or statistical validation.
Every recommendation's confidence is either the rule's own declared
confidence, or that confidence downgraded for missing declared evidence
sources -- never a number invented at request time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from .evidence_providers import build_evidence_context
from .rule_engine import run_rules, Recommendation


def get_recommendations(db_path: str, mine_code: int, date: str, log_evaluation: bool = True) -> dict:
    """
    Returns:
        {
            "mine_code": ...,
            "date": ...,
            "evidence_summary": {"available": [...], "missing": [...]},
            "triggered_recommendations": [Recommendation as dict, ...],
            "not_yet_evaluable_rules": [Recommendation as dict, ...],   # data missing, could not run
            "not_triggered_count": int,
        }
    """
    ctx = build_evidence_context(db_path, mine_code, date)
    results = run_rules(db_path, ctx)

    triggered = [r for r in results if r.triggered and not r.not_yet_evaluable]
    not_evaluable = [r for r in results if r.not_yet_evaluable]
    not_triggered = [r for r in results if not r.triggered and not r.not_yet_evaluable]

    if log_evaluation:
        _log_results(db_path, mine_code, date, results)

    return {
        "mine_code": mine_code,
        "date": date,
        "evidence_summary": {
            "available": sorted(ctx.available_sources),
            "missing": sorted(ctx.unavailable_sources),
        },
        "triggered_recommendations": [asdict(r) for r in triggered],
        "not_yet_evaluable_rules": [asdict(r) for r in not_evaluable],
        "not_triggered_count": len(not_triggered),
    }


def _log_results(db_path: str, mine_code: int, date: str, results: list[Recommendation]) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        rows = []
        for r in results:
            rows.append((
                mine_code, date, r.rule_id, r.rule_name, int(r.triggered),
                r.base_confidence, r.adjusted_confidence,
                ",".join(r.evidence_available), ",".join(r.evidence_missing),
                r.recommendation_text,
            ))
        cur.executemany(
            """
            INSERT INTO environmental_recommendation_log
                (mine_code, date, rule_id, rule_name, triggered, base_confidence,
                 adjusted_confidence, evidence_available, evidence_missing, recommendation_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


def get_rule_catalog(db_path: str) -> list[dict]:
    """Read-only listing of every defined rule and its full metadata --
    useful for an admin/Settings page or for documenting the rule library,
    independent of any specific mine/date evaluation."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, rule_name, variable_name, condition_summary, recommendation_category,
                   recommendation_priority, expected_operational_benefit, confidence_level,
                   is_multi_variable, evidence_sources, depends_on_modules, knowledge_type,
                   scientific_references, is_active
            FROM environmental_recommendation_rules
            ORDER BY id
            """
        )
        rules = [dict(r) for r in cur.fetchall()]
        for rule in rules:
            cur.execute(
                """
                SELECT variable_name, condition_field, condition_operator, condition_value,
                       condition_value_secondary, logical_connector, sequence_order
                FROM environmental_recommendation_conditions
                WHERE rule_id = ? ORDER BY sequence_order
                """,
                (rule["id"],),
            )
            rule["conditions"] = [dict(c) for c in cur.fetchall()]
        return rules
    finally:
        con.close()
