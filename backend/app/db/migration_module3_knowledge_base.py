"""
Module 3 — Environmental Knowledge Base
MILESTONE 1 of 6: Database schema.

Purpose
-------
The Environmental Knowledge Base is the scientific reasoning layer of the
platform. It performs NO machine learning and NO prediction. It encodes,
for every raw and derived environmental variable already produced by
Module 2, the domain knowledge needed to explain *why* environmental
conditions matter for coal quality — definitions, units, physical meaning,
expected (not yet statistically proven) influence on GCV/Moisture/Ash,
relationship type, expected lag, operational interpretation, and a
confidence level. This is the foundation that Explainable AI, Smart
Insights, the Recommendation Engine, the Scenario Simulator, Operational
Alerts, and the Environmental Factors page will all read from later.

Tables created in this milestone (additive only — no changes to any
existing table, including Module 2's):

  environmental_knowledge_base
      One row per environmental variable (raw or derived). The static
      "encyclopedia" entry: what it is, its unit, its physical meaning, its
      operational interpretation, and how confident we are in that
      definition.

  environmental_variable_influence
      One row per (variable, target_metric) pair, where target_metric is
      one of 'gcv' | 'moisture' | 'ash'. Encodes the EXPECTED relationship
      (direction, strength, directness, lag) per domain knowledge. This is
      explicitly a hypothesis register, not a statistically validated
      result — validation_status starts at 'pending' for every row and is
      only ever updated by the future Influence Quantification Engine
      (Module 4), never by this module.

  environmental_recommendation_rules
      One row per operational recommendation rule tied to a variable. The
      actual rule *evaluation* logic is Milestone 3 (rule engine); this
      milestone only defines the schema that stores rule metadata and the
      human-readable recommendation text/condition description.

Design principles
------------------
1. variable_name is the stable join key used across all three tables and
   is intended to match Module 2's column-naming wherever a derived
   feature corresponds 1:1 to a column (e.g. 'environmental_risk_index'
   matches derived_environmental_features.environmental_risk_index). Raw
   weather variables use a normalised name (e.g. 'temperature' rather than
   the unit-suffixed column name 'temperature_mean_c') because a single
   conceptual variable (temperature) can map to several Module 2 columns
   (mean/max/min) — source_table/source_column on
   environmental_knowledge_base records that mapping explicitly.
2. No hard FOREIGN KEY from environmental_variable_influence /
   environmental_recommendation_rules to environmental_knowledge_base is
   declared with ON DELETE/UPDATE actions, but variable_name IS indexed in
   all three tables and uniqueness on environmental_knowledge_base.variable_name
   makes it usable as a logical FK for application-level joins.
3. Nothing in this schema stores a numeric prediction or a SHAP value —
   that is explicitly out of scope until Module 4 (Influence Quantification)
   and Module 5 (Explainable AI).
"""

import sqlite3
from pathlib import Path


DDL_STATEMENTS = [
    # ------------------------------------------------------------------
    # 1. environmental_knowledge_base — variable "encyclopedia" entries
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS environmental_knowledge_base (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        variable_name                   TEXT NOT NULL UNIQUE,   -- stable key, e.g. 'temperature', 'environmental_risk_index'
        display_name                    TEXT NOT NULL,          -- human-readable label, e.g. 'Temperature'
        variable_category               TEXT NOT NULL,          -- 'raw_weather' | 'same_day_derived' | 'rolling_derived'
        unit                            TEXT,                   -- e.g. '°C', '%', 'mm', 'km/h', 'hPa', 'km', 'MJ/m²', 'days', 'score (0-100)'
        scientific_definition           TEXT NOT NULL,
        physical_meaning                TEXT NOT NULL,
        operational_interpretation      TEXT NOT NULL,          -- what an engineer should read into this value
        source_table                    TEXT NOT NULL,          -- which Module 2 table this variable is computed/stored in
        source_column                   TEXT,                   -- column name(s) in that table, comma-separated if multiple (e.g. mean/max/min)
        confidence_level                TEXT NOT NULL,          -- 'high' | 'medium' | 'low' -- confidence in the DEFINITION itself, not the influence claims
        requires_statistical_validation INTEGER NOT NULL DEFAULT 1,  -- 1 = Module 4 should validate, 0 = definitional/no validation needed
        created_at                      TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at                      TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ekb_variable_category ON environmental_knowledge_base(variable_category);",

    # ------------------------------------------------------------------
    # 2. environmental_variable_influence — expected influence hypotheses
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS environmental_variable_influence (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        variable_name                   TEXT NOT NULL,          -- logical FK -> environmental_knowledge_base.variable_name
        target_metric                   TEXT NOT NULL,          -- 'gcv' | 'moisture' | 'ash'
        relationship_type               TEXT NOT NULL,          -- 'direct' | 'indirect'
        influence_direction             TEXT NOT NULL,          -- 'positive' | 'negative' | 'context_dependent'
        influence_strength              TEXT NOT NULL,          -- 'low' | 'medium' | 'high'
        expected_lag_days_min           INTEGER NOT NULL DEFAULT 0,
        expected_lag_days_max           INTEGER NOT NULL DEFAULT 0,
        lag_description                 TEXT,                   -- human-readable lag behaviour, e.g. "effect builds over 2-5 days after rainfall"
        rationale                       TEXT NOT NULL,           -- scientific/operational reasoning behind the expected influence
        confidence_level                TEXT NOT NULL,           -- 'high' | 'medium' | 'low' -- confidence in THIS influence hypothesis
        requires_statistical_validation INTEGER NOT NULL DEFAULT 1,
        validation_status               TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'validated' | 'partially_validated' | 'rejected'
                                                                            -- set ONLY by the future Influence Quantification Engine (Module 4)
        validated_at                    TEXT,                     -- populated by Module 4 when validation_status changes from 'pending'
        created_at                      TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at                      TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (variable_name, target_metric)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_evi_variable ON environmental_variable_influence(variable_name);",
    "CREATE INDEX IF NOT EXISTS idx_evi_target_metric ON environmental_variable_influence(target_metric);",
    "CREATE INDEX IF NOT EXISTS idx_evi_validation_status ON environmental_variable_influence(validation_status);",

    # ------------------------------------------------------------------
    # 3. environmental_recommendation_rules — rule metadata (schema only;
    #    evaluation logic arrives in Milestone 3)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS environmental_recommendation_rules (
        id                       INTEGER PRIMARY KEY AUTOINCREMENT,
        variable_name            TEXT NOT NULL,           -- logical FK -> environmental_knowledge_base.variable_name
        rule_name                TEXT NOT NULL UNIQUE,
        condition_summary        TEXT NOT NULL,           -- human-readable condition, e.g. "consecutive_wet_days >= 3"
        condition_field           TEXT NOT NULL,           -- the Module 2 field this rule actually evaluates, e.g. 'consecutive_wet_days'
        condition_operator        TEXT NOT NULL,           -- '>' | '>=' | '<' | '<=' | '==' | 'between'
        condition_value           REAL,                    -- threshold value (NULL if condition_operator = 'between')
        condition_value_secondary REAL,                    -- upper bound, only used when condition_operator = 'between'
        severity                  TEXT NOT NULL,           -- 'info' | 'warning' | 'critical'
        operational_stage         TEXT NOT NULL,           -- 'pre_sampling' | 'sampling' | 'stockpile_management' | 'dispatch'
        recommendation_text       TEXT NOT NULL,           -- the actual operator-facing recommendation
        is_active                 INTEGER NOT NULL DEFAULT 1,
        created_at                TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_err_variable ON environmental_recommendation_rules(variable_name);",
    "CREATE INDEX IF NOT EXISTS idx_err_severity ON environmental_recommendation_rules(severity);",
    "CREATE INDEX IF NOT EXISTS idx_err_is_active ON environmental_recommendation_rules(is_active);",
]


def run_migration(db_path: str) -> None:
    """Apply the Module 3 Milestone 1 schema migration to the given SQLite database file."""
    db_path = str(Path(db_path))
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for stmt in DDL_STATEMENTS:
            cur.execute(stmt)
        con.commit()
    finally:
        con.close()


def verify_migration(db_path: str) -> dict:
    """Return a dict confirming the three new tables exist and their row counts."""
    expected_tables = [
        "environmental_knowledge_base",
        "environmental_variable_influence",
        "environmental_recommendation_rules",
    ]
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {r[0] for r in cur.fetchall()}
        report = {}
        for t in expected_tables:
            if t in existing:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                report[t] = {"exists": True, "row_count": cur.fetchone()[0]}
            else:
                report[t] = {"exists": False, "row_count": None}
        return report
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_migration(db_file)
    report = verify_migration(db_file)
    print("Module 3 Milestone 1 schema migration applied to:", db_file)
    for table, info in report.items():
        print(f"  {table:38s} exists={info['exists']!s:5s} rows={info['row_count']}")
