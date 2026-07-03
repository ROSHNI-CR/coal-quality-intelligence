"""
Module 3 -- Environmental Knowledge Base
MILESTONE 3: Example recommendation rules.

These rules demonstrate the modular, multi-variable, multi-source
architecture requested:
  - Trigger logic can combine 2+ environmental variables (AND/OR), not a
    single fixed threshold.
  - Every rule explicitly records its supporting variables, scientific
    rationale (drawn from / consistent with the Knowledge Base), priority,
    category, expected operational benefit, confidence, and which
    evidence sources / future modules it depends on.
  - Rules are written to run TODAY using only what Module 2 (environmental
    layer) and Module 3 (knowledge base) provide, but explicitly declare
    where Module 4 (Influence Quantification), Module 5 (Explainable AI /
    prediction confidence), and a Data Quality service would enrich them
    -- so adding those modules later upgrades the rule's confidence/output
    without a schema or rule-table change.

This is a STARTER set (5 rules), not an exhaustive rule library --
intentionally small so the engine's mechanics can be reviewed before more
rules are added.
"""

import sqlite3

RULES = [
    dict(
        rule_name="extended_wet_spell_moisture_risk",
        variable_name="consecutive_wet_days",
        condition_summary="consecutive_wet_days >= 3 AND environmental_risk_index >= 60",
        is_multi_variable=1,
        # legacy single-condition columns left NULL when is_multi_variable=1
        condition_field='see_environmental_recommendation_conditions', condition_operator='multi', condition_value=None, condition_value_secondary=None,
        severity="warning",
        operational_stage="pre_sampling",
        recommendation_category="sampling_timing",
        recommendation_priority="high",
        recommendation_text="An extended wet spell combined with elevated overall environmental risk suggests "
                             "moisture is likely accumulating in exposed/stockpiled coal. Consider delaying "
                             "non-urgent sampling until conditions stabilise, or explicitly flag samples taken "
                             "during this window as having elevated expected moisture for downstream "
                             "interpretation.",
        expected_operational_benefit="Reduces the chance of moisture-driven GCV misinterpretation and avoids "
                                      "unnecessary re-sampling by setting correct expectations up front.",
        confidence_level="medium",
        evidence_sources="current_conditions,weather_history,knowledge_base",
        depends_on_modules="module2_environmental_layer,module3_knowledge_base",
        knowledge_type="operational_assumption",
        scientific_references="Osborne, D. (ed.), The Coal Handbook: Towards Cleaner Production, Vol. 1; "
                               "CIMFR literature on coal stockpile weathering",
        is_active=1,
        conditions=[
            dict(variable_name="consecutive_wet_days", condition_field="consecutive_wet_days",
                 condition_operator=">=", condition_value=3, condition_value_secondary=None,
                 logical_connector=None, sequence_order=1),
            dict(variable_name="environmental_risk_index", condition_field="environmental_risk_index",
                 condition_operator=">=", condition_value=60, condition_value_secondary=None,
                 logical_connector="AND", sequence_order=2),
        ],
    ),
    dict(
        rule_name="favourable_drying_window",
        variable_name="drying_potential",
        condition_summary="drying_potential >= 70 AND consecutive_dry_days >= 2",
        is_multi_variable=1,
        condition_field='see_environmental_recommendation_conditions', condition_operator='multi', condition_value=None, condition_value_secondary=None,
        severity="info",
        operational_stage="pre_sampling",
        recommendation_category="sampling_timing",
        recommendation_priority="medium",
        recommendation_text="Conditions are currently favourable for natural drying (high drying potential "
                             "sustained over multiple consecutive dry days). This is a reasonable window for "
                             "sampling if low moisture / higher GCV results are operationally desirable, since "
                             "weather is unlikely to be artificially inflating moisture right now.",
        expected_operational_benefit="Helps operators time sampling to periods more representative of the "
                                      "coal's 'dry' quality potential rather than a transient wet anomaly.",
        confidence_level="medium",
        evidence_sources="current_conditions,weather_history,knowledge_base",
        depends_on_modules="module2_environmental_layer,module3_knowledge_base",
        knowledge_type="project_specific_rule",
        scientific_references="Penman (1948) evaporation theory; FAO Penman-Monteith framework (conceptual basis)",
        is_active=1,
        conditions=[
            dict(variable_name="drying_potential", condition_field="drying_potential",
                 condition_operator=">=", condition_value=70, condition_value_secondary=None,
                 logical_connector=None, sequence_order=1),
            dict(variable_name="consecutive_dry_days", condition_field="consecutive_dry_days",
                 condition_operator=">=", condition_value=2, condition_value_secondary=None,
                 logical_connector="AND", sequence_order=2),
        ],
    ),
    dict(
        rule_name="overnight_condensation_risk",
        variable_name="dew_spread",
        condition_summary="dew_spread <= 2.0 AND relative_humidity_mean_pct >= 85",
        is_multi_variable=1,
        condition_field='see_environmental_recommendation_conditions', condition_operator='multi', condition_value=None, condition_value_secondary=None,
        severity="warning",
        operational_stage="stockpile_management",
        recommendation_category="stockpile_management",
        recommendation_priority="medium",
        recommendation_text="Narrow dew spread combined with high humidity indicates elevated overnight "
                             "condensation risk on exposed stockpile surfaces, independent of any forecast "
                             "rainfall. Consider covering or otherwise protecting near-surface coal if low "
                             "moisture is operationally critical for the next sampling cycle.",
        expected_operational_benefit="Pre-empts a moisture source (condensation) that would not be visible from "
                                      "rainfall data alone, reducing surprise moisture readings.",
        confidence_level="medium",
        evidence_sources="current_conditions,knowledge_base",
        depends_on_modules="module2_environmental_layer,module3_knowledge_base",
        knowledge_type="established_principle",
        scientific_references="Psychrometric principles (ASHRAE Handbook - Fundamentals, Psychrometrics chapter)",
        is_active=1,
        conditions=[
            dict(variable_name="dew_spread", condition_field="dew_spread_c",
                 condition_operator="<=", condition_value=2.0, condition_value_secondary=None,
                 logical_connector=None, sequence_order=1),
            dict(variable_name="relative_humidity", condition_field="relative_humidity_mean_pct",
                 condition_operator=">=", condition_value=85, condition_value_secondary=None,
                 logical_connector="AND", sequence_order=2),
        ],
    ),
    dict(
        rule_name="single_factor_heavy_rainfall_flag",
        variable_name="rainfall",
        condition_summary="rainfall_mm >= 25 (single-day)",
        is_multi_variable=0,
        condition_field="rainfall_mm", condition_operator=">=", condition_value=25, condition_value_secondary=None,
        severity="critical",
        operational_stage="sampling",
        recommendation_category="sampling_timing",
        recommendation_priority="critical",
        recommendation_text="A heavy single-day rainfall event was recorded. Any sample taken on or immediately "
                             "after this date should be treated as likely to show anomalously high moisture, not "
                             "representative of typical conditions for this mine.",
        expected_operational_benefit="Prevents a single extreme weather event from being misread as a baseline "
                                      "quality shift.",
        confidence_level="high",
        evidence_sources="current_conditions,knowledge_base",
        depends_on_modules="module2_environmental_layer,module3_knowledge_base",
        knowledge_type="established_principle",
        scientific_references="Osborne, D. (ed.), The Coal Handbook: Towards Cleaner Production, Vol. 1",
        is_active=1,
        conditions=[],  # single-variable rule: uses condition_field directly, no rows needed here
    ),
    dict(
        rule_name="weather_data_pending_caveat",
        variable_name="weather_records",
        condition_summary="No weather_records exist for this mine/date (production ingestion pending)",
        is_multi_variable=0,
        condition_field="data_availability", condition_operator="==", condition_value=0, condition_value_secondary=None,
        severity="info",
        operational_stage="pre_sampling",
        recommendation_category="data_quality_caveat",
        recommendation_priority="low",
        recommendation_text="No real weather observations are available yet for this mine/date (production "
                             "Open-Meteo ingestion is pending in the current environment). Any environmental "
                             "risk assessment for this period should be treated as unavailable rather than "
                             "assumed favourable or unfavourable -- do not infer conditions from absence of data.",
        expected_operational_benefit="Prevents silently treating missing weather data as 'no risk', which would "
                                      "be a misleading default.",
        confidence_level="high",
        evidence_sources="data_quality",
        depends_on_modules="module2_environmental_layer",
        knowledge_type="project_specific_rule",
        scientific_references=None,
        is_active=1,
        conditions=[],
    ),
]


def populate(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()

        rule_cols = [
            "variable_name", "rule_name", "condition_summary", "condition_field", "condition_operator",
            "condition_value", "condition_value_secondary", "severity", "operational_stage",
            "recommendation_text", "is_active", "knowledge_type", "scientific_references",
            "recommendation_category", "recommendation_priority", "expected_operational_benefit",
            "confidence_level", "is_multi_variable", "evidence_sources", "depends_on_modules",
        ]

        rules_inserted = 0
        conditions_inserted = 0

        for rule in RULES:
            # upsert by rule_name (UNIQUE)
            cur.execute("SELECT id FROM environmental_recommendation_rules WHERE rule_name = ?", (rule["rule_name"],))
            existing = cur.fetchone()

            values = [rule.get(c) for c in rule_cols]
            if existing:
                rule_id = existing[0]
                set_clause = ", ".join(f"{c}=?" for c in rule_cols)
                cur.execute(
                    f"UPDATE environmental_recommendation_rules SET {set_clause}, updated_at=datetime('now') WHERE id=?",
                    values + [rule_id],
                )
                cur.execute("DELETE FROM environmental_recommendation_conditions WHERE rule_id = ?", (rule_id,))
            else:
                placeholders = ",".join(["?"] * len(rule_cols))
                cur.execute(
                    f"INSERT INTO environmental_recommendation_rules ({','.join(rule_cols)}) VALUES ({placeholders})",
                    values,
                )
                rule_id = cur.lastrowid

            rules_inserted += 1

            for cond in rule["conditions"]:
                cur.execute(
                    """
                    INSERT INTO environmental_recommendation_conditions
                        (rule_id, sequence_order, logical_connector, variable_name,
                         condition_field, condition_operator, condition_value, condition_value_secondary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id, cond["sequence_order"], cond["logical_connector"], cond["variable_name"],
                        cond["condition_field"], cond["condition_operator"], cond["condition_value"],
                        cond["condition_value_secondary"],
                    ),
                )
                conditions_inserted += 1

        con.commit()
        return {"rules": rules_inserted, "conditions": conditions_inserted}
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    result = populate(db_file)
    print(f"Populated/updated {result['rules']} rules and {result['conditions']} conditions in {db_file}")
