"""
Module 3 — Environmental Knowledge Base
MILESTONE 2a: Schema extension (additive ALTER TABLE only).

Adds three fields to environmental_knowledge_base and
environmental_variable_influence so every entry can be scientifically
defensible and auditable:

  knowledge_type
      'established_principle'   -> well-documented meteorological / coal
                                    science / mining engineering fact
      'operational_assumption'  -> reasonable, widely-used operational
                                    heuristic, context-dependent, not a
                                    universal physical law
      'project_specific_rule'   -> a composite/derived metric or threshold
                                    defined specifically for this platform
                                    (e.g. Environmental Risk Index, Drying
                                    Potential) — grounded in real drivers
                                    but the exact formula/weights are a
                                    project design choice, not literature

  scientific_references
      Free-text citation list (book/standard/principle names — never
      reproduced verbatim text from any source, per citation policy).
      NULL where a relationship is purely a project-specific construction
      with no external literature to cite.

  confidence_rationale
      One or two sentences explaining WHY this row was assigned its
      confidence_level — makes the Knowledge Base self-documenting for
      the Explainable AI module to surface to end users.

environmental_recommendation_rules also gets knowledge_type +
scientific_references for the same reason, since recommendation rules are,
by nature, mostly project-specific operational thresholds.

This migration is idempotent: it checks for column existence before
attempting to add it, so it is safe to re-run.
"""

import sqlite3


def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


ALTER_SPECS = {
    "environmental_knowledge_base": [
        ("knowledge_type", "TEXT NOT NULL DEFAULT 'established_principle'"),
        ("scientific_references", "TEXT"),
        ("confidence_rationale", "TEXT"),
    ],
    "environmental_variable_influence": [
        ("knowledge_type", "TEXT NOT NULL DEFAULT 'established_principle'"),
        ("scientific_references", "TEXT"),
        ("confidence_rationale", "TEXT"),
    ],
    "environmental_recommendation_rules": [
        ("knowledge_type", "TEXT NOT NULL DEFAULT 'project_specific_rule'"),
        ("scientific_references", "TEXT"),
    ],
}


def run_migration(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for table, columns in ALTER_SPECS.items():
            for col_name, col_def in columns:
                if not _column_exists(cur, table, col_name):
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        con.commit()
    finally:
        con.close()


def verify_migration(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        report = {}
        for table, columns in ALTER_SPECS.items():
            cur.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cur.fetchall()}
            report[table] = {col_name: (col_name in existing_cols) for col_name, _ in columns}
        return report
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_migration(db_file)
    report = verify_migration(db_file)
    print("Module 3 Milestone 2a schema extension applied to:", db_file)
    for table, cols in report.items():
        print(f"  {table}:")
        for col, present in cols.items():
            print(f"    {col}: present={present}")
