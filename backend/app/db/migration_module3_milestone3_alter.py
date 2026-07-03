"""
Module 3 -- Environmental Knowledge Base
MILESTONE 3: Rule Engine -- schema extension.

Architectural requirement driving this design (per explicit instruction):
recommendations must NOT be a flat collection of fixed IF-ELSE thresholds.
The Recommendation Engine is a FRAMEWORK that:
  1. Can express multi-variable trigger logic (AND/OR composite conditions),
     not just a single field/operator/value.
  2. Records, for every rule: supporting variables, scientific rationale,
     priority, category, expected operational benefit, confidence, and
     explicit dependencies on future modules (Influence Quantification,
     Prediction, Explainable AI, Data Quality).
  3. Is designed to be enriched later -- a rule's confidence and behaviour
     can improve once Module 4 (Influence Quantification), Module 5
     (Explainable AI / prediction confidence), and a Data Quality service
     exist, WITHOUT schema changes -- those modules simply become
     additional "evidence sources" the engine consults.

Three additions, all additive (no existing table touched):

  environmental_recommendation_rules (ALTERED, columns added only)
      Adds the rule-level metadata: category, priority, expected benefit,
      confidence, multi-variable flag, evidence sources this rule
      consults, and explicit dependencies on not-yet-built modules.

  environmental_recommendation_conditions (NEW)
      One row per condition within a rule's trigger logic. A rule with
      is_multi_variable=1 has 2+ rows here, combined via logical_connector
      ('AND'/'OR') in sequence_order -- this is what lets a rule require
      e.g. "consecutive_wet_days >= 3 AND environmental_risk_index >= 60"
      instead of a single fixed threshold.

  environmental_recommendation_log (NEW, schema only for now)
      Audit table for recommendations actually generated/issued for a
      mine/date by the engine -- which rule fired, what evidence was
      available at the time, what confidence was assigned. Empty until the
      engine is run against real (non-empty) weather data; exists now so
      Operational Alerts / Smart Insights have a stable table to query
      later without another migration.
"""

import sqlite3


def _column_exists(cur, table, column):
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


ALTER_COLUMNS = [
    ("recommendation_category", "TEXT NOT NULL DEFAULT 'general'"),
    # 'sampling_timing' | 'stockpile_management' | 'dispatch_planning' | 'data_quality_caveat' | 'general'
    ("recommendation_priority", "TEXT NOT NULL DEFAULT 'medium'"),
    # 'low' | 'medium' | 'high' | 'critical'
    ("expected_operational_benefit", "TEXT"),
    ("confidence_level", "TEXT NOT NULL DEFAULT 'medium'"),
    # confidence in THIS rule's recommendation, independent of KB variable confidence
    ("is_multi_variable", "INTEGER NOT NULL DEFAULT 0"),
    # 1 => ignore condition_field/operator/value on this row, use
    # environmental_recommendation_conditions instead
    ("evidence_sources", "TEXT"),
    # comma-separated: which evidence categories this rule consults, e.g.
    # 'current_conditions,weather_history,knowledge_base'. Future values
    # this same column will hold once those modules exist:
    # 'influence_quantification,prediction_confidence,explainable_ai,data_quality'
    ("depends_on_modules", "TEXT"),
    # comma-separated module identifiers this rule's FULL design depends on,
    # e.g. 'module2_environmental_layer,module3_knowledge_base' (available now)
    # vs 'module4_influence_quantification,module5_explainable_ai' (future --
    # rule runs today in a degraded/lower-confidence form without them)
]

DDL_CONDITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS environmental_recommendation_conditions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id                     INTEGER NOT NULL,         -- logical FK -> environmental_recommendation_rules.id
    sequence_order              INTEGER NOT NULL,         -- evaluation order within the rule
    logical_connector           TEXT,                     -- 'AND' | 'OR' | NULL (NULL for the first condition)
    variable_name               TEXT NOT NULL,             -- logical FK -> environmental_knowledge_base.variable_name
    condition_field             TEXT NOT NULL,             -- actual Module 2 field evaluated, e.g. 'consecutive_wet_days'
    condition_operator          TEXT NOT NULL,             -- '>' | '>=' | '<' | '<=' | '==' | 'between'
    condition_value             REAL,
    condition_value_secondary   REAL,
    created_at                  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

DDL_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS environmental_recommendation_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mine_code               INTEGER NOT NULL,
    date                    TEXT NOT NULL,
    rule_id                 INTEGER NOT NULL,             -- logical FK -> environmental_recommendation_rules.id
    rule_name               TEXT NOT NULL,
    triggered               INTEGER NOT NULL,             -- 1 = conditions met, 0 = evaluated but not triggered
    base_confidence         TEXT,                          -- the rule's static confidence_level at evaluation time
    adjusted_confidence      TEXT,                          -- confidence after evidence-availability adjustment
    evidence_available      TEXT,                          -- comma-separated evidence sources actually available
    evidence_missing        TEXT,                          -- comma-separated evidence sources the rule wanted but lacked
    recommendation_text     TEXT,
    generated_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def run_migration(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for col_name, col_def in ALTER_COLUMNS:
            if not _column_exists(cur, "environmental_recommendation_rules", col_name):
                cur.execute(f"ALTER TABLE environmental_recommendation_rules ADD COLUMN {col_name} {col_def}")
        cur.execute(DDL_CONDITIONS_TABLE)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_erc_rule_id ON environmental_recommendation_conditions(rule_id)")
        cur.execute(DDL_LOG_TABLE)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_erl_mine_date ON environmental_recommendation_log(mine_code, date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_erl_rule_id ON environmental_recommendation_log(rule_id)")
        con.commit()
    finally:
        con.close()


def verify_migration(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        report = {}
        for col_name, _ in ALTER_COLUMNS:
            report[f"environmental_recommendation_rules.{col_name}"] = _column_exists(
                cur, "environmental_recommendation_rules", col_name
            )
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {r[0] for r in cur.fetchall()}
        for t in ["environmental_recommendation_conditions", "environmental_recommendation_log"]:
            report[t] = t in existing
        return report
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_migration(db_file)
    report = verify_migration(db_file)
    print("Module 3 Milestone 3 schema extension applied to:", db_file)
    for k, v in report.items():
        print(f"  {k}: {v}")
