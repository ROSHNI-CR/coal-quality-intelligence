"""
Module 6 -- Explainable AI & Decision Support
Schema migration (additive only).

Tables:
  prediction_explanations   -- local feature attributions per prediction
  mine_insights             -- generated natural-language insight rows per mine/date
  operational_alerts        -- triggered alerts from recommendation engine evaluation
"""
import sqlite3

DDL = [
    """
    CREATE TABLE IF NOT EXISTS prediction_explanations (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id           INTEGER NOT NULL,   -- FK -> predictions.id
        mine_code               INTEGER NOT NULL,
        date                    TEXT NOT NULL,
        target_metric           TEXT NOT NULL,
        feature_name            TEXT NOT NULL,
        attribution_score       REAL NOT NULL,       -- signed: positive=increases target, negative=decreases
        attribution_rank        INTEGER NOT NULL,    -- 1=most influential for this prediction
        attribution_method      TEXT NOT NULL,       -- 'local_permutation' | 'shap' | 'marginal_contribution'
        kb_physical_meaning     TEXT,                -- from environmental_knowledge_base.physical_meaning
        kb_operational_interp   TEXT,                -- from environmental_knowledge_base.operational_interpretation
        kb_validation_status    TEXT,                -- from environmental_variable_influence.validation_status_assigned
        feature_value           REAL,                -- actual value for this mine/date
        created_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_pe_prediction ON prediction_explanations(prediction_id);",
    "CREATE INDEX IF NOT EXISTS idx_pe_mine_date ON prediction_explanations(mine_code, date);",

    """
    CREATE TABLE IF NOT EXISTS mine_insights (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code           INTEGER NOT NULL,
        date                TEXT NOT NULL,
        insight_type        TEXT NOT NULL,       -- 'what_happened' | 'why' | 'whats_next' | 'what_to_do'
        target_metric       TEXT,                -- 'gcv' | 'moisture' | 'ash' | NULL (general)
        insight_text        TEXT NOT NULL,
        confidence_level    TEXT NOT NULL,       -- 'high' | 'medium' | 'low'
        evidence_sources    TEXT,                -- comma-separated: 'module4_ml','module3_kb','module2_weather'
        is_pending          INTEGER NOT NULL DEFAULT 0,
        pending_reason      TEXT,
        generated_at        TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_mi_mine_date ON mine_insights(mine_code, date);",

    """
    CREATE TABLE IF NOT EXISTS operational_alerts (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code           INTEGER NOT NULL,
        date                TEXT NOT NULL,
        rule_name           TEXT NOT NULL,
        severity            TEXT NOT NULL,       -- 'info' | 'warning' | 'critical'
        recommendation_category TEXT NOT NULL,
        recommendation_text TEXT NOT NULL,
        confidence_level    TEXT NOT NULL,
        evidence_available  TEXT,
        evidence_missing    TEXT,
        recommendation_basis TEXT,
        is_active           INTEGER NOT NULL DEFAULT 1,
        generated_at        TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_oa_mine_date ON operational_alerts(mine_code, date);",
    "CREATE INDEX IF NOT EXISTS idx_oa_severity ON operational_alerts(severity, is_active);",
]

def run_migration(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        for stmt in DDL:
            con.execute(stmt)
        con.commit()
    finally:
        con.close()

def verify_migration(db_path: str) -> dict:
    tables = ["prediction_explanations", "mine_insights", "operational_alerts"]
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {r[0] for r in cur.fetchall()}
        return {t: t in existing for t in tables}
    finally:
        con.close()

if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_migration(db)
    print("Module 6 schema applied:", verify_migration(db))
