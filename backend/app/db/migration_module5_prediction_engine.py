"""
Module 5 -- Prediction Engine
Schema migration.

Scope discipline (per explicit instruction): Module 5 is PREDICTION ONLY.
No explanation, narrative, or recommendation generation lives here -- that
is deferred to Module 6 (Explainable AI / Decision Support). This module
deploys the Module 4 GroupKFold-validated best model per target as a
persisted production artifact, serves point predictions + intervals +
confidence labels via a clean service API, and logs every prediction to a
history table. Nothing here writes to or modifies any Module 1-4 table.

Tables:

  prediction_models
      One row per persisted production model (one per target_metric,
      though the schema allows multiple versions over time via
      is_active). References the Module 4 run_id and
      model_benchmark_results row it was selected from -- the model
      FAMILY and hyperparameters are never re-decided here, only
      refit-and-persisted as a deployable artifact. Stores the on-disk
      path to the serialized model (joblib), the feature column order
      (critical -- predictions must present features in the exact order
      the model was trained on), and residual-quantile interval bounds
      computed from honest out-of-fold (GroupKFold) residuals -- not
      training-set residuals, which would understate uncertainty.

  predictions
      One row per prediction call (history log). Stores point estimate,
      interval bounds, confidence label, which prediction_models row was
      used, whether weather data was available (refusals are logged too,
      with prediction_status='refused_missing_data' and NULL estimates --
      a refusal is itself a recorded event, not a silent failure), and
      the actual observed value if a matching sampling_records row exists
      for that mine/date (enabling backtesting accuracy review without
      ever joining back into or modifying sampling_records itself).
"""

import sqlite3


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS prediction_models (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        target_metric               TEXT NOT NULL,           -- 'gcv' | 'moisture' | 'ash'
        source_run_id               INTEGER NOT NULL,         -- module4_run_metadata.id this model was selected from
        model_name                  TEXT NOT NULL,            -- 'lightgbm' | 'xgboost' | 'random_forest' | 'linear_regression'
        model_implementation        TEXT NOT NULL,            -- actual class used (honest substitute labeling preserved)
        model_artifact_path         TEXT NOT NULL,             -- on-disk path to the joblib-serialized fitted model
        feature_columns_json        TEXT NOT NULL,             -- JSON list, exact order used at training time
        cv_r2_mean                  REAL NOT NULL,             -- from model_benchmark_results (GroupKFold), informs confidence label
        cv_rmse_mean                REAL NOT NULL,
        cv_mae_mean                 REAL NOT NULL,
        interval_lower_quantile     REAL NOT NULL,             -- e.g. 0.10
        interval_upper_quantile     REAL NOT NULL,             -- e.g. 0.90
        residual_lower_offset       REAL NOT NULL,             -- additive offset to point estimate for lower bound
        residual_upper_offset       REAL NOT NULL,             -- additive offset to point estimate for upper bound
        training_sample_size        INTEGER NOT NULL,
        random_state                INTEGER NOT NULL,
        is_active                   INTEGER NOT NULL DEFAULT 1, -- only one active model per target_metric expected
        trained_at                  TEXT NOT NULL DEFAULT (datetime('now')),
        notes                       TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_pm_target_active ON prediction_models(target_metric, is_active);",

    """
    CREATE TABLE IF NOT EXISTS predictions (
        id                       INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code                INTEGER NOT NULL,
        date                     TEXT NOT NULL,               -- YYYY-MM-DD, the date predicted for
        target_metric            TEXT NOT NULL,
        prediction_model_id      INTEGER,                       -- NULL if refused before model lookup made sense
        prediction_status        TEXT NOT NULL,                 -- 'success' | 'refused_missing_weather' | 'refused_unmapped_mine' | 'refused_incomplete_features'
        point_estimate           REAL,                           -- NULL if refused
        interval_lower           REAL,
        interval_upper           REAL,
        confidence_label         TEXT,                           -- 'high' | 'medium' | 'low', derived from cv_r2_mean, NULL if refused
        actual_value             REAL,                           -- populated if a matching sampling_records value exists (backtest comparison), else NULL
        is_backtest               INTEGER NOT NULL DEFAULT 1,     -- this module supports historical/backtest predictions only
        refusal_reason            TEXT,                           -- human-readable reason when prediction_status != 'success'
        requested_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_pred_mine_date ON predictions(mine_code, date);",
    "CREATE INDEX IF NOT EXISTS idx_pred_target_status ON predictions(target_metric, prediction_status);",
]


def run_migration(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for stmt in DDL_STATEMENTS:
            cur.execute(stmt)
        con.commit()
    finally:
        con.close()


def verify_migration(db_path: str) -> dict:
    expected = ["prediction_models", "predictions"]
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {r[0] for r in cur.fetchall()}
        report = {}
        for t in expected:
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
    print("Module 5 schema migration applied to:", db_file)
    for t, info in report.items():
        print(f"  {t:25s} exists={info['exists']!s:5s} rows={info['row_count']}")
