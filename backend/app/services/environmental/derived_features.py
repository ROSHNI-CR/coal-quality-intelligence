"""
Module 2 — Environmental Variable Layer
Derived feature calculator.

Reads weather_records and computes two layers of derived features:

  environmental_features          -> same-day features (no temporal window)
  derived_environmental_features  -> rolling/temporal-window features

This module has NO machine learning dependency, per the architecture
requirement — it is pure deterministic feature engineering over the raw
weather observations, and is meant to be the stable foundation that the
ML/XAI modules (3+) consume later.

Feature definitions
--------------------
Same-day (environmental_features):
  temperature_range_c    = temperature_max_c - temperature_min_c
  dew_spread_c           = temperature_mean_c - dew_point_mean_c
                           (small/negative dew spread => air near saturation,
                            i.e. high condensation / moisture risk)
  thermal_stress_index   = a 0-100 heat+humidity load proxy, computed as a
                           humidity-weighted excess-temperature score:
                               excess_temp = max(0, temperature_mean_c - 25)
                               thermal_stress = excess_temp * (humidity/100) * 4
                           clipped to [0, 100]. Higher = more heat+moisture
                           stress on exposed coal stockpiles.

Rolling / window (derived_environmental_features), all computed per mine
ordered by date, using only past+current days (no look-ahead leakage):
  rolling_rainfall_3d_mm        = sum of rainfall over trailing 3 days
  rolling_rainfall_7d_mm        = sum of rainfall over trailing 7 days
  rolling_humidity_7d_pct       = mean of relative_humidity_mean_pct, trailing 7 days
  rolling_solar_radiation_7d_mj_m2 = mean of solar_radiation_mj_m2, trailing 7 days
  consecutive_wet_days          = run length of rainfall_mm > 1.0 ending today
  consecutive_dry_days          = run length of rainfall_mm <= 1.0 ending today
  moisture_accumulation_index   = exponentially-decayed cumulative moisture
                                   signal: today = rainfall_mm + 0.5*humidity_mean_pct/10
                                   plus 0.85 * yesterday's index (decay factor 0.85
                                   chosen so ~7-day half-life of accumulated wetness)
  drying_potential              = composite 0-100 score combining temperature,
                                   wind, solar radiation (favourable) against
                                   humidity and rainfall (unfavourable):
                                       drying_potential =
                                         clip(
                                           0.25*temp_mean_c
                                         + 0.8*wind_speed_mean_kmh
                                         + 1.2*solar_radiation_mj_m2
                                         - 0.4*relative_humidity_mean_pct
                                         - 1.5*rainfall_mm
                                         , 0, 100, after rescaling)
  environmental_risk_index      = inverse-flavoured composite, 0-100, higher =
                                   worse for coal quality (wetter/more humid/
                                   less drying):
                                       eri = clip(100 - drying_potential
                                             + 0.3*rolling_rainfall_7d_mm, 0, 100)
  weather_stability_index       = 0-100, higher = more stable/consistent
                                   recent weather, computed as
                                   100 - normalised std-dev of temperature_mean_c
                                   and rainfall_mm over trailing 7 days.

All formulas are intentionally transparent/explainable (no black-box
weighting) so the Explainable AI module can describe them in plain language
without needing SHAP on Module 2 itself — SHAP is reserved for the ML
prediction layer in Module 4+, exactly as scoped.
"""

from __future__ import annotations

import sqlite3

import pandas as pd


def _load_weather_df(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT mine_code, date, temperature_max_c, temperature_min_c, temperature_mean_c,
               relative_humidity_mean_pct, rainfall_mm, dew_point_mean_c,
               wind_speed_mean_kmh, solar_radiation_mj_m2
        FROM weather_records
        ORDER BY mine_code, date
        """,
        con,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _clip(series: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


def compute_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same-day features. No grouping/window needed."""
    out = pd.DataFrame()
    out["mine_code"] = df["mine_code"]
    out["date"] = df["date"].dt.strftime("%Y-%m-%d")
    out["temperature_range_c"] = (df["temperature_max_c"] - df["temperature_min_c"]).round(2)
    out["dew_spread_c"] = (df["temperature_mean_c"] - df["dew_point_mean_c"]).round(2)

    excess_temp = (df["temperature_mean_c"] - 25).clip(lower=0)
    thermal_stress = excess_temp * (df["relative_humidity_mean_pct"] / 100.0) * 4.0
    out["thermal_stress_index"] = _clip(thermal_stress).round(2)
    return out


def compute_derived_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling/window features, computed independently per mine_code group,
    strictly causal (trailing windows only, no look-ahead)."""
    df = df.sort_values(["mine_code", "date"]).reset_index(drop=True)
    results = []

    for mine_code, g in df.groupby("mine_code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        rainfall = g["rainfall_mm"].fillna(0.0)
        humidity = g["relative_humidity_mean_pct"]
        temp_mean = g["temperature_mean_c"]
        wind = g["wind_speed_mean_kmh"]
        solar = g["solar_radiation_mj_m2"]

        rolling_rainfall_3d = rainfall.rolling(window=3, min_periods=1).sum()
        rolling_rainfall_7d = rainfall.rolling(window=7, min_periods=1).sum()
        rolling_humidity_7d = humidity.rolling(window=7, min_periods=1).mean()
        rolling_solar_7d = solar.rolling(window=7, min_periods=1).mean()

        # consecutive wet/dry day run-lengths (causal, per-row)
        is_wet = (rainfall > 1.0).astype(int)
        consecutive_wet = []
        consecutive_dry = []
        wet_run, dry_run = 0, 0
        for w in is_wet:
            if w == 1:
                wet_run += 1
                dry_run = 0
            else:
                dry_run += 1
                wet_run = 0
            consecutive_wet.append(wet_run)
            consecutive_dry.append(dry_run)

        # moisture accumulation index: decayed cumulative wetness signal
        decay = 0.85
        moisture_idx = []
        prev = 0.0
        for r, h in zip(rainfall, humidity.ffill().fillna(50.0)):
            today_signal = r + 0.5 * (h / 10.0)
            cur_val = today_signal + decay * prev
            moisture_idx.append(round(cur_val, 2))
            prev = cur_val

        # drying potential: favourable (temp, wind, solar) vs unfavourable (humidity, rain)
        raw_drying = (
            0.25 * temp_mean.fillna(0)
            + 0.8 * wind.fillna(0)
            + 1.2 * solar.fillna(0)
            - 0.4 * humidity.fillna(0)
            - 1.5 * rainfall.fillna(0)
        )
        # rescale raw_drying (~ -30..+30 typical range observed) into 0-100
        drying_potential = _clip((raw_drying + 30) * (100 / 60))

        # environmental risk index: inverse of drying potential + rolling wetness penalty
        eri = _clip(100 - drying_potential + 0.3 * rolling_rainfall_7d)

        # weather stability index: 100 - normalised 7d std-dev of temp & rainfall
        temp_std_7d = temp_mean.rolling(window=7, min_periods=2).std().fillna(0)
        rain_std_7d = rainfall.rolling(window=7, min_periods=2).std().fillna(0)
        # normalise: temp std of ~6C and rain std of ~15mm treated as "very unstable" (=100 penalty)
        instability = _clip((temp_std_7d / 6.0) * 50 + (rain_std_7d / 15.0) * 50)
        wsi = _clip(100 - instability)

        out = pd.DataFrame({
            "mine_code": mine_code,
            "date": g["date"].dt.strftime("%Y-%m-%d"),
            "drying_potential": drying_potential.round(2),
            "environmental_risk_index": eri.round(2),
            "weather_stability_index": wsi.round(2),
            "consecutive_wet_days": consecutive_wet,
            "consecutive_dry_days": consecutive_dry,
            "moisture_accumulation_index": moisture_idx,
            "rolling_rainfall_3d_mm": rolling_rainfall_3d.round(2),
            "rolling_rainfall_7d_mm": rolling_rainfall_7d.round(2),
            "rolling_humidity_7d_pct": rolling_humidity_7d.round(2),
            "rolling_solar_radiation_7d_mj_m2": rolling_solar_7d.round(2),
        })
        results.append(out)

    return pd.concat(results, ignore_index=True)


def persist_features(con: sqlite3.Connection, env_df: pd.DataFrame, derived_df: pd.DataFrame) -> dict:
    cur = con.cursor()

    env_cols = ["mine_code", "date", "temperature_range_c", "dew_spread_c", "thermal_stress_index"]
    cur.executemany(
        f"""
        INSERT OR REPLACE INTO environmental_features ({",".join(env_cols)})
        VALUES ({",".join(["?"]*len(env_cols))})
        """,
        env_df[env_cols].itertuples(index=False, name=None),
    )

    derived_cols = [
        "mine_code", "date", "drying_potential", "environmental_risk_index",
        "weather_stability_index", "consecutive_wet_days", "consecutive_dry_days",
        "moisture_accumulation_index", "rolling_rainfall_3d_mm", "rolling_rainfall_7d_mm",
        "rolling_humidity_7d_pct", "rolling_solar_radiation_7d_mj_m2",
    ]
    cur.executemany(
        f"""
        INSERT OR REPLACE INTO derived_environmental_features ({",".join(derived_cols)})
        VALUES ({",".join(["?"]*len(derived_cols))})
        """,
        derived_df[derived_cols].itertuples(index=False, name=None),
    )
    con.commit()
    return {"environmental_features_rows": len(env_df), "derived_environmental_features_rows": len(derived_df)}


def run_feature_computation(db_path: str, verbose: bool = True) -> dict:
    con = sqlite3.connect(db_path)
    try:
        df = _load_weather_df(con)
        if df.empty:
            if verbose:
                print("No weather_records found — run ingestion.py first.")
            return {"environmental_features_rows": 0, "derived_environmental_features_rows": 0}

        env_df = compute_environmental_features(df)
        derived_df = compute_derived_environmental_features(df)
        summary = persist_features(con, env_df, derived_df)

        if verbose:
            print(f"Computed environmental_features for {df['mine_code'].nunique()} mines, "
                  f"{summary['environmental_features_rows']} rows.")
            print(f"Computed derived_environmental_features for {df['mine_code'].nunique()} mines, "
                  f"{summary['derived_environmental_features_rows']} rows.")
        return summary
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_feature_computation(db_file)
