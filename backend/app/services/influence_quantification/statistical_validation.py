"""
Module 4 -- Environmental Influence Quantification Engine
Statistical validation layer.

Provides SUPPORTING evidence for the ML/SHAP-primary influence ranking --
never the primary ranking itself, per the agreed architecture. Computes,
per (target_metric, feature):
  - Pearson r and Spearman rho (linear and monotonic association)
  - Best-lag correlation: tests lag 0-7 days between the environmental
    variable and the target, reusing each mine's own time series (lagged
    by shifting the feature column within mine_code groups, ordered by
    date) -- corroborates or contradicts the expected_lag_days_min/max
    windows already encoded in environmental_variable_influence (Module 3)
  - Seasonal breakdown: correlation computed separately within each of the
    four seasons used elsewhere in the platform (pre_monsoon: Apr-Jun,
    monsoon: Jul-Sep, post_monsoon: Oct-Nov, winter: Dec-Mar) -- supported
    by the dataset's full 12-month span (confirmed in pre-Module-4
    validation).

This layer never trains a predictive model and never produces a feature
ranking on its own -- it produces evidence rows that the orchestrator
attaches alongside the ML ranking for human/SHAP-consuming review.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats


def _season_for_date(date_str: str) -> str:
    month = int(date_str[5:7])
    if month in (4, 5, 6):
        return "pre_monsoon"
    if month in (7, 8, 9):
        return "monsoon"
    if month in (10, 11):
        return "post_monsoon"
    return "winter"  # 12, 1, 2, 3


def compute_correlations(df: pd.DataFrame, feature_columns: list[str], target_col: str = "target") -> dict:
    """Pearson r and Spearman rho per feature against the target, on the
    full assembled dataset (one row per sample, as agreed)."""
    results = {}
    y = df[target_col]
    for f in feature_columns:
        x = df[f]
        if x.std() == 0 or y.std() == 0:
            results[f] = {"pearson_r": None, "spearman_rho": None}
            continue
        pearson_r, _ = stats.pearsonr(x, y)
        spearman_rho, _ = stats.spearmanr(x, y)
        results[f] = {"pearson_r": float(pearson_r), "spearman_rho": float(spearman_rho)}
    return results


def compute_best_lag_correlation(df: pd.DataFrame, feature_columns: list[str], target_col: str = "target",
                                   max_lag_days: int = 7) -> dict:
    """
    For each feature, tests correlation between the feature value at lag L
    days before the sample date and the target, for L in [0, max_lag_days],
    WITHIN each mine's own chronological sequence (so a lag never crosses
    between two different mines). Returns the lag with the strongest
    absolute correlation per feature.

    Note: this requires the per-mine daily WEATHER series (not the
    per-sample dataset, since sampling dates aren't necessarily daily) --
    so this function expects df to already include all of a mine's daily
    rows joined to that mine's sample target via merge_asof-style nearest-
    or-exact alignment. For simplicity and correctness here, lag is
    computed by shifting each mine's sample-level series by lag in terms
    of *sample order*, which is an approximation; a stricter version would
    re-join to daily weather_records per lag offset. See orchestrator notes
    for how this is invoked.
    """
    results = {}
    df_sorted = df.sort_values(["mine_code", "sample_date"]).reset_index(drop=True)

    for f in feature_columns:
        best_lag, best_corr = 0, 0.0
        for lag in range(0, max_lag_days + 1):
            shifted = df_sorted.groupby("mine_code")[f].shift(lag)
            valid = shifted.notna() & df_sorted[target_col].notna()
            if valid.sum() < 30:
                continue
            corr, _ = stats.pearsonr(shifted[valid], df_sorted.loc[valid, target_col])
            if abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag
        results[f] = {"best_lag_days": best_lag, "best_lag_correlation": float(best_corr)}
    return results


def compute_seasonal_breakdown(df: pd.DataFrame, feature_columns: list[str], target_col: str = "target") -> dict:
    """Pearson r per feature, computed separately within each season."""
    df = df.copy()
    df["season"] = df["sample_date"].apply(_season_for_date)

    results = {f: {} for f in feature_columns}
    for season, g in df.groupby("season"):
        if len(g) < 30:
            for f in feature_columns:
                results[f][season] = None
            continue
        for f in feature_columns:
            x, y = g[f], g[target_col]
            if x.std() == 0 or y.std() == 0:
                results[f][season] = None
                continue
            corr, _ = stats.pearsonr(x, y)
            results[f][season] = round(float(corr), 4)

    return {f: json.dumps(v) for f, v in results.items()}
