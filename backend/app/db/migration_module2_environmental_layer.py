"""
Module 2 — Environmental Variable Layer
Schema migration script.

Creates four new, normalized tables that extend (never modify) the existing
production schema:

    weather_records                -> raw daily weather observations per mapped mine
    environmental_features         -> same-day derived environmental features
    derived_environmental_features -> rolling/temporal-window derived features
    weather_api_metadata           -> ingestion run log / provenance tracking

Design principles
------------------
1. Additive only. No ALTER/DROP on mine_master, mine_coordinates,
   sampling_records, dispatch_records, eie_metadata, or data_quality_log.
2. Keyed by (mine_code, date) so that:
     - any future module can join weather_records / environmental_features /
       derived_environmental_features directly onto sampling_records via
       (mine_code, date) <-> (mine_code, date)
     - mine_code is a SOFT reference to mine_master.mine_code. mine_master
       does not declare mine_code as PRIMARY KEY/UNIQUE in the existing
       production schema, and per the "do not modify existing tables" rule
       we cannot add that constraint retroactively. SQLite requires a FK
       target to be a unique/primary key, so a hard FOREIGN KEY to
       mine_master is not possible without altering it. We therefore
       enforce the relationship at the application layer (environmental_service.py)
       and add a plain index on mine_code for join performance instead.
       The FK between the new Module 2 tables themselves (e.g.
       environmental_features -> weather_records) IS enforced, since
       weather_records.(mine_code, date) is declared UNIQUE by us.
3. Architecture must tolerate mines being added to mine_coordinates later
   without any redesign -> nothing here is hardcoded to the current 45 mines.
4. is_synthetic flag is threaded through every observation-level table so
   that production code can always tell real API data apart from
   development placeholder data, and so synthetic rows can be bulk-deleted
   and replaced once production API access is available.
"""

import sqlite3
from pathlib import Path


DDL_STATEMENTS = [
    # ------------------------------------------------------------------
    # 1. weather_api_metadata — ingestion run log / provenance
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS weather_api_metadata (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code           INTEGER NOT NULL,
        source              TEXT    NOT NULL,              -- 'open-meteo-archive' | 'nasa-power' | 'synthetic'
        request_start_date  TEXT    NOT NULL,               -- YYYY-MM-DD
        request_end_date    TEXT    NOT NULL,               -- YYYY-MM-DD
        records_requested   INTEGER,
        records_fetched     INTEGER,
        status              TEXT    NOT NULL,               -- 'SUCCESS' | 'PARTIAL' | 'FAILED' | 'SKIPPED'
        http_status         INTEGER,
        is_synthetic        INTEGER NOT NULL DEFAULT 0,      -- 1 = dev placeholder, 0 = real API
        error_message       TEXT,
        fetched_at          TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_weather_api_metadata_mine ON weather_api_metadata(mine_code);",
    "CREATE INDEX IF NOT EXISTS idx_weather_api_metadata_status ON weather_api_metadata(status);",

    # ------------------------------------------------------------------
    # 2. weather_records — raw daily observations
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS weather_records (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code               INTEGER NOT NULL,
        date                    TEXT    NOT NULL,            -- YYYY-MM-DD, joins sampling_records.date
        temperature_max_c       REAL,
        temperature_min_c       REAL,
        temperature_mean_c      REAL,
        relative_humidity_mean_pct REAL,
        relative_humidity_max_pct REAL,
        relative_humidity_min_pct REAL,
        rainfall_mm             REAL,
        dew_point_mean_c        REAL,
        wind_speed_mean_kmh     REAL,
        wind_gust_max_kmh       REAL,
        surface_pressure_mean_hpa REAL,
        cloud_cover_mean_pct    REAL,
        visibility_mean_km      REAL,
        solar_radiation_mj_m2   REAL,                        -- shortwave radiation sum
        weather_code            INTEGER,                      -- WMO code (Open-Meteo convention)
        source                  TEXT    NOT NULL,             -- 'open-meteo-archive' | 'nasa-power' | 'synthetic'
        is_synthetic            INTEGER NOT NULL DEFAULT 0,
        ingested_at             TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (mine_code, date)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_weather_records_mine_date ON weather_records(mine_code, date);",
    "CREATE INDEX IF NOT EXISTS idx_weather_records_date ON weather_records(date);",

    # ------------------------------------------------------------------
    # 3. environmental_features — same-day derived features
    #    (computable directly from a single day's weather_records row,
    #     no temporal window required)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS environmental_features (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code           INTEGER NOT NULL,
        date                TEXT    NOT NULL,
        temperature_range_c REAL,                            -- temperature_max_c - temperature_min_c
        dew_spread_c        REAL,                            -- temperature_mean_c - dew_point_mean_c
        thermal_stress_index REAL,                            -- normalised heat+humidity load, 0-100
        computed_at         TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (mine_code, date),
        FOREIGN KEY (mine_code, date) REFERENCES weather_records(mine_code, date)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_environmental_features_mine_date ON environmental_features(mine_code, date);",

    # ------------------------------------------------------------------
    # 4. derived_environmental_features — rolling / temporal-window features
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS derived_environmental_features (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        mine_code                   INTEGER NOT NULL,
        date                        TEXT    NOT NULL,
        drying_potential            REAL,    -- composite: high temp/wind/solar + low humidity/rain -> high score
        environmental_risk_index    REAL,    -- 0-100, higher = more adverse for coal quality
        weather_stability_index     REAL,    -- 0-100, higher = more stable/consistent recent weather
        consecutive_wet_days        INTEGER, -- run length of rainfall > 1mm ending on `date`
        consecutive_dry_days        INTEGER, -- run length of rainfall <= 1mm ending on `date`
        moisture_accumulation_index REAL,    -- decayed cumulative rainfall + humidity signal
        rolling_rainfall_3d_mm      REAL,
        rolling_rainfall_7d_mm      REAL,
        rolling_humidity_7d_pct     REAL,
        rolling_solar_radiation_7d_mj_m2 REAL,
        computed_at                 TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (mine_code, date),
        FOREIGN KEY (mine_code, date) REFERENCES weather_records(mine_code, date)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_derived_env_features_mine_date ON derived_environmental_features(mine_code, date);",
]


def run_migration(db_path: str) -> None:
    """Apply the Module 2 schema migration to the given SQLite database file."""
    db_path = str(Path(db_path))
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        for stmt in DDL_STATEMENTS:
            cur.execute(stmt)
        con.commit()
    finally:
        con.close()


def verify_migration(db_path: str) -> dict:
    """Return a dict confirming the four new tables exist and their row counts."""
    expected_tables = [
        "weather_api_metadata",
        "weather_records",
        "environmental_features",
        "derived_environmental_features",
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
    print("Module 2 schema migration applied to:", db_file)
    for table, info in report.items():
        print(f"  {table:35s} exists={info['exists']!s:5s} rows={info['row_count']}")
