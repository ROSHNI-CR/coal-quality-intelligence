"""
Module 2 — Environmental Variable Layer
Ingestion orchestrator (PRODUCTION-ONLY).

For every mapped mine (mine_master.is_mapped = 1):
  1. Determine the date range needing weather, driven by that mine's own
     sampling_records date coverage (NOT hardcoded) — newly mapped mines or
     extended sampling periods are picked up automatically with no code
     changes.
  2. Skip dates already present in weather_records for that mine (idempotent,
     safe to re-run).
  3. Call the real Open-Meteo Archive API (open_meteo_client.py).
  4. If the API call succeeds, persist the rows to weather_records with
     is_synthetic = 0, source = 'open-meteo-archive'.
  5. If the API call fails for any reason (no network egress, rate limit,
     bad response, etc.), DO NOT generate placeholder data. The mine's
     weather_records rows for that period are simply left absent. The
     attempt — success or failure — is always logged in
     weather_api_metadata so the gap is visible and auditable, and so a
     later re-run from a network-enabled environment will pick up exactly
     where this one left off.

This file has NO synthetic data dependency of any kind. There is no dev
fallback path. If Open-Meteo cannot be reached, Module 2's weather tables
remain empty or partially populated, and every downstream module must treat
"no row" as "not yet available" rather than assume any value.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Optional

from .open_meteo_client import fetch_archive_weather, is_api_reachable

WEATHER_COLUMNS = [
    "temperature_max_c", "temperature_min_c", "temperature_mean_c",
    "relative_humidity_mean_pct", "relative_humidity_max_pct", "relative_humidity_min_pct",
    "rainfall_mm", "dew_point_mean_c", "wind_speed_mean_kmh", "wind_gust_max_kmh",
    "surface_pressure_mean_hpa", "cloud_cover_mean_pct", "visibility_mean_km",
    "solar_radiation_mj_m2", "weather_code",
]

OPEN_METEO_KEY_MAP = {
    "temperature_2m_max": "temperature_max_c",
    "temperature_2m_min": "temperature_min_c",
    "temperature_2m_mean": "temperature_mean_c",
    "relative_humidity_2m_mean": "relative_humidity_mean_pct",
    "relative_humidity_2m_max": "relative_humidity_max_pct",
    "relative_humidity_2m_min": "relative_humidity_min_pct",
    "precipitation_sum": "rainfall_mm",
    "dew_point_2m_mean": "dew_point_mean_c",
    "wind_speed_10m_max": "wind_speed_mean_kmh",
    "wind_gusts_10m_max": "wind_gust_max_kmh",
    "surface_pressure_mean": "surface_pressure_mean_hpa",
    "cloud_cover_mean": "cloud_cover_mean_pct",
    "visibility_mean": "visibility_mean_km",
    "shortwave_radiation_sum": "solar_radiation_mj_m2",
    "weather_code": "weather_code",
}


def _get_mapped_mines(con: sqlite3.Connection) -> list[dict]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT mine_code, latitude, longitude, mine_name, subsidiary
        FROM mine_master
        WHERE is_mapped = 1
        ORDER BY mine_code
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _get_required_date_range(con: sqlite3.Connection, mine_code: int) -> Optional[tuple[str, str]]:
    cur = con.cursor()
    cur.execute(
        "SELECT MIN(date), MAX(date) FROM sampling_records WHERE mine_code = ?",
        (mine_code,),
    )
    row = cur.fetchone()
    if row and row[0] and row[1]:
        return row[0][:10], row[1][:10]

    cur.execute("SELECT value FROM eie_metadata WHERE key='data_date_range_start'")
    start_row = cur.fetchone()
    cur.execute("SELECT value FROM eie_metadata WHERE key='data_date_range_end'")
    end_row = cur.fetchone()
    if start_row and end_row:
        return start_row[0], end_row[0]
    return None


def _existing_dates(con: sqlite3.Connection, mine_code: int) -> set[str]:
    cur = con.cursor()
    cur.execute("SELECT date FROM weather_records WHERE mine_code = ?", (mine_code,))
    return {r[0] for r in cur.fetchall()}


def _date_range_list(start_date: str, end_date: str) -> list[str]:
    s = date.fromisoformat(start_date)
    e = date.fromisoformat(end_date)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _insert_weather_rows(con: sqlite3.Connection, mine_code: int, rows: list[dict], source: str):
    cur = con.cursor()
    cols = ["mine_code", "date"] + WEATHER_COLUMNS + ["source", "is_synthetic"]
    placeholders = ",".join(["?"] * len(cols))
    sql = f"""
        INSERT OR IGNORE INTO weather_records ({",".join(cols)})
        VALUES ({placeholders})
    """
    payload = []
    for r in rows:
        values = [mine_code, r["date"]] + [r.get(c) for c in WEATHER_COLUMNS] + [source, 0]
        payload.append(values)
    cur.executemany(sql, payload)
    con.commit()
    return cur.rowcount


def _log_metadata(con: sqlite3.Connection, mine_code: int, source: str, start_date: str, end_date: str,
                   requested: int, fetched: int, status: str, http_status: Optional[int],
                   error_message: Optional[str]):
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO weather_api_metadata
            (mine_code, source, request_start_date, request_end_date,
             records_requested, records_fetched, status, http_status,
             is_synthetic, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (mine_code, source, start_date, end_date, requested, fetched, status,
         http_status, error_message),
    )
    con.commit()


def _transform_open_meteo_daily(daily_block: dict) -> list[dict]:
    dates = daily_block.get("time", [])
    rows = []
    for i, d in enumerate(dates):
        row = {"date": d}
        for om_key, col in OPEN_METEO_KEY_MAP.items():
            series = daily_block.get(om_key)
            row[col] = series[i] if series and i < len(series) else None
        rows.append(row)
    return rows


def ingest_weather_for_all_mapped_mines(db_path: str, verbose: bool = True) -> dict:
    """
    Main entry point. Production-only: calls the real Open-Meteo Archive API
    for every mapped mine. No synthetic data is ever generated.

    If the API is unreachable from the current environment, this function
    still runs to completion — it logs a SKIPPED/FAILED status per mine in
    weather_api_metadata and leaves weather_records untouched for those
    mines. Re-running this exact function later from a network-enabled
    environment will pick up exactly where it left off (idempotent).
    """
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")

    api_reachable = is_api_reachable()
    if verbose:
        print(f"[Module 2 Ingestion] Open-Meteo Archive API reachable: {api_reachable}")
        if not api_reachable:
            print("[Module 2 Ingestion] Production ingestion pipeline is implemented and idempotent, "
                  "but cannot execute in this environment due to lack of network egress to "
                  "archive-api.open-meteo.com. No synthetic data will be generated. "
                  "weather_records will remain empty/partial until re-run from a network-enabled host.")

    mines = _get_mapped_mines(con)
    summary = {
        "api_reachable": api_reachable,
        "mines_processed": 0,
        "mines_skipped_no_range": 0,
        "mines_already_complete": 0,
        "rows_inserted": 0,
        "real_api_success": 0,
        "failed": 0,
    }

    for mine in mines:
        mine_code = mine["mine_code"]
        date_range = _get_required_date_range(con, mine_code)
        if not date_range:
            summary["mines_skipped_no_range"] += 1
            continue
        start_date, end_date = date_range

        all_dates = _date_range_list(start_date, end_date)
        existing = _existing_dates(con, mine_code)
        missing = [d for d in all_dates if d not in existing]
        if not missing:
            if verbose:
                print(f"  mine {mine_code}: already fully ingested ({len(existing)} days), skipping")
            summary["mines_already_complete"] += 1
            summary["mines_processed"] += 1
            continue

        fetch_start, fetch_end = missing[0], missing[-1]

        if not api_reachable:
            _log_metadata(
                con, mine_code, "open-meteo-archive", fetch_start, fetch_end,
                requested=len(missing), fetched=0,
                status="SKIPPED", http_status=None,
                error_message="Open-Meteo Archive API unreachable in current execution environment "
                               "(no network egress). No data fetched. No synthetic fallback used. "
                               "Pending real ingestion run.",
            )
            if verbose:
                print(f"  mine {mine_code} ({mine['mine_name']}): SKIPPED — API unreachable, "
                      f"{len(missing)} days pending")
            summary["failed"] += 1
            summary["mines_processed"] += 1
            continue

        result = fetch_archive_weather(mine["latitude"], mine["longitude"], fetch_start, fetch_end)

        if result.success:
            rows = _transform_open_meteo_daily(result.daily)
            missing_set = set(missing)
            rows = [r for r in rows if r["date"] in missing_set]
            inserted = _insert_weather_rows(con, mine_code, rows, "open-meteo-archive")
            summary["rows_inserted"] += inserted
            summary["real_api_success"] += 1
            _log_metadata(
                con, mine_code, "open-meteo-archive", fetch_start, fetch_end,
                requested=len(missing), fetched=len(rows),
                status="SUCCESS", http_status=result.http_status, error_message=None,
            )
            if verbose:
                print(f"  mine {mine_code} ({mine['mine_name']}): {len(rows)} days ingested via Open-Meteo")
        else:
            summary["failed"] += 1
            _log_metadata(
                con, mine_code, "open-meteo-archive", fetch_start, fetch_end,
                requested=len(missing), fetched=0,
                status="FAILED", http_status=result.http_status, error_message=result.error_message,
            )
            if verbose:
                print(f"  mine {mine_code} ({mine['mine_name']}): FAILED — {result.error_message}")

        summary["mines_processed"] += 1

    con.close()
    return summary


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    summary = ingest_weather_for_all_mapped_mines(db_file)
    print("\n=== Module 2 Ingestion Summary (production-only) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
