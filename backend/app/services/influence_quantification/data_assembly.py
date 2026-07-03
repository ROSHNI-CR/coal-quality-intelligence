"""
Module 4 -- Environmental Influence Quantification Engine
Data assembly layer.

Builds the joined per-sample analytical dataset (sampling x weather x
same-day features x rolling features) used by the ML benchmarking and
statistical validation layers. This is the ONLY place the join-layer
fixes identified in pre-Module-4 validation are applied -- nothing here
writes back to or modifies mine_master, sampling_records, weather_records,
environmental_features, or derived_environmental_features. All fixes are
applied in-memory, at read time, every run.

Fixes applied here (per agreed plan):
  1. Date normalization: sampling_records.date is 'YYYY-MM-DD HH:MM:SS',
     weather_records.date is 'YYYY-MM-DD'. Joined via substr(date,1,10) on
     the sampling side -- never by altering stored values.
  2. Authoritative mapping source: mine selection uses mine_master.is_mapped,
     never the stale sampling_records.is_mapped snapshot column.
  3. Invalid target exclusion: rows with ash_pct outside [0,100] or
     total_moisture_pct outside [0,100] (or gcv_valid=0) are excluded from
     the analytical dataset returned to callers -- never deleted or
     modified in sampling_records itself. Excluded counts are always
     returned alongside the data so callers can report on them.
  4. Granularity: per-sample is the default unit of analysis (no daily
     aggregation) -- each sampling_records row becomes one ML observation,
     joined to that day's weather/derived features for its mine.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

FEATURE_COLUMNS = [
    # raw weather
    "temperature_max_c", "temperature_min_c", "temperature_mean_c",
    "relative_humidity_mean_pct", "relative_humidity_max_pct", "relative_humidity_min_pct",
    "rainfall_mm", "dew_point_mean_c", "wind_speed_mean_kmh", "wind_gust_max_kmh",
    "surface_pressure_mean_hpa", "cloud_cover_mean_pct", "solar_radiation_mj_m2",
    # NOTE: visibility_mean_km intentionally excluded -- 100% NULL in current dataset
    # (documented Module 2 limitation: Open-Meteo archive endpoint does not reliably
    # provide a daily visibility aggregate). Re-include if/when populated.

    # same-day derived
    "temperature_range_c", "dew_spread_c", "thermal_stress_index",

    # rolling derived
    "drying_potential", "environmental_risk_index", "weather_stability_index",
    "consecutive_wet_days", "consecutive_dry_days", "moisture_accumulation_index",
    "rolling_rainfall_3d_mm", "rolling_rainfall_7d_mm", "rolling_humidity_7d_pct",
    "rolling_solar_radiation_7d_mj_m2",
]

# Maps feature column -> the Module 3 Knowledge Base variable_name it corresponds to
# (used later to reconcile ML/statistical findings against the KB hypothesis register)
FEATURE_TO_KB_VARIABLE = {
    "temperature_mean_c": "temperature",
    "relative_humidity_mean_pct": "relative_humidity",
    "rainfall_mm": "rainfall",
    "dew_point_mean_c": "dew_point",
    "wind_speed_mean_kmh": "wind_speed",
    "wind_gust_max_kmh": "wind_gust",
    "surface_pressure_mean_hpa": "surface_pressure",
    "cloud_cover_mean_pct": "cloud_cover",
    "solar_radiation_mj_m2": "solar_radiation",
    "temperature_range_c": "temperature_range",
    "dew_spread_c": "dew_spread",
    "thermal_stress_index": "thermal_stress",
    "drying_potential": "drying_potential",
    "environmental_risk_index": "environmental_risk_index",
    "weather_stability_index": "weather_stability_index",
    "consecutive_wet_days": "consecutive_wet_days",
    "consecutive_dry_days": "consecutive_dry_days",
    "moisture_accumulation_index": "moisture_accumulation_index",
    # rolling humidity/rainfall/solar and the min/max temp/humidity variants
    # don't have their own 1:1 KB entry (KB covers the canonical daily-mean
    # variable); they are still used as ML features but map back to the same
    # KB variable_name as their canonical counterpart for reconciliation.
    "relative_humidity_max_pct": "relative_humidity",
    "relative_humidity_min_pct": "relative_humidity",
    "temperature_max_c": "temperature",
    "temperature_min_c": "temperature",
    "rolling_rainfall_3d_mm": "rainfall",
    "rolling_rainfall_7d_mm": "rainfall",
    "rolling_humidity_7d_pct": "relative_humidity",
    "rolling_solar_radiation_7d_mj_m2": "solar_radiation",
}

TARGET_COLUMN_MAP = {
    "gcv": "gcv",
    "moisture": "total_moisture_pct",
    "ash": "ash_pct",
}


@dataclass
class AssembledDataset:
    target_metric: str
    df: pd.DataFrame                 # feature columns + target column, one row per valid sample
    feature_columns: list
    n_total_joined: int               # rows after join, before target-validity filtering
    n_excluded_invalid_target: int
    n_final: int
    mines_covered: int


def _load_raw_joined(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Single SQL join applying fixes #1 (date normalization) and #2
    (authoritative mapping source via mine_master.is_mapped). Returns one
    row per sampling_records row that has a matching weather/feature row,
    for currently mapped mines only.
    """
    query = f"""
        SELECT
            s.mine_code,
            substr(s.date, 1, 10) AS sample_date,
            s.gcv, s.gcv_valid, s.moisture_pct, s.total_moisture_pct, s.ash_pct,
            s.declared_grade, s.sampling_grade, s.grade_category, s.subsidiary,
            {", ".join("w." + c for c in [
                "temperature_max_c", "temperature_min_c", "temperature_mean_c",
                "relative_humidity_mean_pct", "relative_humidity_max_pct", "relative_humidity_min_pct",
                "rainfall_mm", "dew_point_mean_c", "wind_speed_mean_kmh", "wind_gust_max_kmh",
                "surface_pressure_mean_hpa", "cloud_cover_mean_pct", "solar_radiation_mj_m2",
            ])},
            {", ".join("e." + c for c in ["temperature_range_c", "dew_spread_c", "thermal_stress_index"])},
            {", ".join("d." + c for c in [
                "drying_potential", "environmental_risk_index", "weather_stability_index",
                "consecutive_wet_days", "consecutive_dry_days", "moisture_accumulation_index",
                "rolling_rainfall_3d_mm", "rolling_rainfall_7d_mm", "rolling_humidity_7d_pct",
                "rolling_solar_radiation_7d_mj_m2",
            ])}
        FROM sampling_records s
        JOIN mine_master m
            ON m.mine_code = s.mine_code AND m.is_mapped = 1
        JOIN weather_records w
            ON w.mine_code = s.mine_code AND w.date = substr(s.date, 1, 10)
        LEFT JOIN environmental_features e
            ON e.mine_code = s.mine_code AND e.date = substr(s.date, 1, 10)
        LEFT JOIN derived_environmental_features d
            ON d.mine_code = s.mine_code AND d.date = substr(s.date, 1, 10)
    """
    return pd.read_sql_query(query, con)


def assemble_dataset(db_path: str, target_metric: str) -> AssembledDataset:
    """
    Build the analytical dataset for one target_metric ('gcv' | 'moisture' | 'ash').

    Applies fix #3 (invalid target exclusion) for THIS target only -- a row
    excluded for the 'ash' dataset (bad ash_pct) may still be valid and
    included in the 'gcv' or 'moisture' dataset, since each target is
    quality-gated independently. Applies fix #4 implicitly: no aggregation
    is performed, each sampling row stays one row.
    """
    if target_metric not in TARGET_COLUMN_MAP:
        raise ValueError(f"Unknown target_metric: {target_metric!r}. Must be one of {list(TARGET_COLUMN_MAP)}")

    con = sqlite3.connect(db_path)
    try:
        raw = _load_raw_joined(con)
    finally:
        con.close()

    n_total_joined = len(raw)
    mines_covered = raw["mine_code"].nunique()

    target_col = TARGET_COLUMN_MAP[target_metric]

    if target_metric == "gcv":
        valid_mask = (raw["gcv_valid"] == 1) & raw["gcv"].notna()
    elif target_metric == "moisture":
        valid_mask = raw["total_moisture_pct"].notna() & (raw["total_moisture_pct"] >= 0) & (raw["total_moisture_pct"] <= 100)
    elif target_metric == "ash":
        valid_mask = raw["ash_pct"].notna() & (raw["ash_pct"] >= 0) & (raw["ash_pct"] <= 100)
    else:
        valid_mask = pd.Series(True, index=raw.index)

    n_excluded = int((~valid_mask).sum())
    filtered = raw[valid_mask].copy()

    # also require all feature columns present (no NULL features) for a clean ML matrix
    feature_cols_available = [c for c in FEATURE_COLUMNS if c in filtered.columns]
    complete_mask = filtered[feature_cols_available].notna().all(axis=1)
    n_excluded_incomplete_features = int((~complete_mask).sum())
    filtered = filtered[complete_mask].copy()

    filtered = filtered.rename(columns={target_col: "target"})
    keep_cols = feature_cols_available + ["target", "mine_code", "sample_date", "grade_category", "subsidiary"]
    final_df = filtered[keep_cols].reset_index(drop=True)

    return AssembledDataset(
        target_metric=target_metric,
        df=final_df,
        feature_columns=feature_cols_available,
        n_total_joined=n_total_joined,
        n_excluded_invalid_target=n_excluded + n_excluded_incomplete_features,
        n_final=len(final_df),
        mines_covered=mines_covered,
    )


def get_assembly_summary(db_path: str) -> dict:
    """Quick summary across all three targets -- used for the run-metadata
    audit row and for sanity-checking before training."""
    summary = {}
    for target in TARGET_COLUMN_MAP:
        ds = assemble_dataset(db_path, target)
        summary[target] = {
            "n_total_joined": ds.n_total_joined,
            "n_excluded_invalid_or_incomplete": ds.n_excluded_invalid_target,
            "n_final": ds.n_final,
            "mines_covered": ds.mines_covered,
        }
    return summary


if __name__ == "__main__":
    import sys
    import json
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    print(json.dumps(get_assembly_summary(db_file), indent=2))
