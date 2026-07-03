"""
Module 4 -- Environmental Influence Quantification Engine
Orchestrator.

Ties together: data assembly (fixes applied, no source-table writes) ->
model benchmarking (4 model families, cross-validated, best selected per
target) -> SHAP/permutation-importance explainability on the selected
model, refit on full data -> statistical validation (correlation, lag,
seasonal -- supporting evidence only) -> reconciliation against the
Module 3 Knowledge Base hypothesis register -> persistence to the Module 4
output tables.

This is the single public entry point: run_influence_quantification(db_path).

Architecture discipline enforced here (per explicit instruction):
  - ML (SHAP-primary or its honest fallback) determines ml_importance_rank.
    Statistics NEVER override the ML ranking -- they are stored alongside
    as corroborating/contradicting evidence only.
  - The Knowledge Base is consulted for `agreement_with_kb` reconciliation
    and to update environmental_variable_influence.validation_status, but
    the KB's pre-existing hypothesis never determines the ML ranking or is
    silently treated as ground truth.
  - Every output row carries `evidence_label` ('observed_ml_influence' or
    'observed_statistical_association') and a mandatory `causation_caveat`
    -- this engine NEVER asserts proven causation anywhere.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from .data_assembly import assemble_dataset, FEATURE_TO_KB_VARIABLE
from .model_benchmarking import benchmark_models, select_best_model, refit_on_full_data, get_library_availability
from .explainability import compute_feature_importance, get_shap_availability
from .statistical_validation import compute_correlations, compute_best_lag_correlation, compute_seasonal_breakdown

CAUSATION_CAVEAT = (
    "This result reflects an OBSERVED statistical/ML association in the available data, not a proven causal "
    "relationship. Confounding factors (e.g. seasonal correlation between multiple weather variables, mine-specific "
    "operational practices, sampling timing) may contribute to this finding. Use alongside the Environmental "
    "Knowledge Base's physical rationale and domain judgement before acting on it operationally."
)


def _get_kb_hypothesis(con: sqlite3.Connection, variable_name: str, target_metric: str) -> dict | None:
    cur = con.cursor()
    cur.execute(
        """SELECT influence_direction, confidence_level FROM environmental_variable_influence
           WHERE variable_name = ? AND target_metric = ?""",
        (variable_name, target_metric),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"influence_direction": row[0], "confidence_level": row[1]}


def _reconcile_with_kb(ml_direction: str, kb_hypothesis: dict | None) -> tuple[str, str]:
    """Returns (agreement_with_kb, validation_status_assigned)."""
    if kb_hypothesis is None:
        return "kb_had_no_directional_hypothesis", "inconclusive"

    kb_dir = kb_hypothesis["influence_direction"]
    if kb_dir == "context_dependent":
        return "kb_had_no_directional_hypothesis", "inconclusive"

    if ml_direction == "mixed":
        return "partial", "partially_validated"

    if ml_direction == kb_dir:
        return "agrees", "validated"

    return "disagrees", "rejected"


def _insert_run_metadata(con: sqlite3.Connection, assembly_summary: dict) -> int:
    lib = get_library_availability()
    shap_lib = get_shap_availability()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO module4_run_metadata
            (xgboost_available, lightgbm_available, shap_available,
             xgboost_substitute_used, lightgbm_substitute_used, shap_substitute_used,
             sample_count_gcv, sample_count_moisture, sample_count_ash,
             rows_excluded_ash_invalid, rows_excluded_moisture_invalid, rows_excluded_gcv_invalid,
             notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(lib["xgboost_available"]), int(lib["lightgbm_available"]), int(shap_lib["shap_available"]),
            lib["xgboost_substitute_used"], lib["lightgbm_substitute_used"], shap_lib["shap_substitute_used"],
            assembly_summary["gcv"]["n_final"], assembly_summary["moisture"]["n_final"], assembly_summary["ash"]["n_final"],
            assembly_summary["ash"]["n_excluded_invalid_or_incomplete"],
            assembly_summary["moisture"]["n_excluded_invalid_or_incomplete"],
            assembly_summary["gcv"]["n_excluded_invalid_or_incomplete"],
            "Model benchmarking CV pass used GroupKFold(groups=mine_code, n_splits=5) -- every fold contains "
            "completely unseen mines, no mine_code appears in both train and validation within any fold. A "
            "15,000-row random subsample (group membership preserved) was used for the cross-validated "
            "model-family comparison pass only (full dataset is computationally large for 4-model x 3-target x "
            "5-fold CV in this environment); the SELECTED best model per target was then refit on the FULL "
            "target-specific dataset before SHAP/importance extraction. Linear Regression is wrapped in a "
            "Pipeline(StandardScaler, LinearRegression); tree-based models are left unscaled. See "
            "model_benchmark_results.training_sample_size and .cv_folds per row.",
        ),
    )
    con.commit()
    return cur.lastrowid


def start_run(db_path: str) -> int:
    """Creates the module4_run_metadata row and returns the new run_id.
    Call this once per run, then run_target() once per target_metric
    (each independently resumable/retriable), then complete_run() at the end."""
    con = sqlite3.connect(db_path)
    try:
        from .data_assembly import get_assembly_summary
        assembly_summary = get_assembly_summary(db_path)
        return _insert_run_metadata(con, assembly_summary)
    finally:
        con.close()


def run_target(db_path: str, run_id: int, target_metric: str, cv_folds: int = 5,
                benchmark_sample_size: int = 15000, verbose: bool = True) -> dict:
    """Runs the full benchmarking -> explainability -> statistical validation
    -> KB reconciliation -> persistence pipeline for ONE target_metric,
    under an already-started run_id. Safe to call once per target_metric;
    does not touch module4_run_metadata.run_completed_at (see complete_run)."""
    con = sqlite3.connect(db_path)
    try:
        if verbose:
            print(f"\n=== Target: {target_metric} ===")
        ds = assemble_dataset(db_path, target_metric)
        X = ds.df[ds.feature_columns]
        y = ds.df["target"]
        groups = ds.df["mine_code"]

        if verbose:
            print(f"  Dataset: {len(X)} samples, {len(ds.feature_columns)} features, {ds.mines_covered} mines")

        bench_results = benchmark_models(X, y, groups, cv_folds=cv_folds, benchmark_sample_size=benchmark_sample_size)
        best = select_best_model(bench_results)
        if verbose:
            for r in bench_results:
                flag = " <- SELECTED" if r.model_name == best.model_name else ""
                print(f"  {r.model_name:18s} R2={r.r2_mean:.3f}+-{r.r2_std:.3f} "
                      f"RMSE={r.rmse_mean:.2f} ({r.model_implementation}){flag}")

        cur = con.cursor()
        for r in bench_results:
            cur.execute(
                """INSERT INTO model_benchmark_results
                   (run_id, target_metric, model_name, model_implementation, cv_folds,
                    rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std,
                    is_selected_best, training_sample_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, target_metric, r.model_name, r.model_implementation, r.cv_folds,
                 r.rmse_mean, r.rmse_std, r.mae_mean, r.mae_std, r.r2_mean, r.r2_std,
                 int(r.model_name == best.model_name), r.training_sample_size),
            )
        con.commit()

        fitted_model, impl_label = refit_on_full_data(best.model_name, X, y)

        importance = compute_feature_importance(fitted_model, best.model_name, X, y)
        if verbose:
            print(f"  Explainability method: {importance.method}")

        correlations = compute_correlations(ds.df, ds.feature_columns)
        lags = compute_best_lag_correlation(ds.df, ds.feature_columns)
        seasonal = compute_seasonal_breakdown(ds.df, ds.feature_columns)

        ranked_features = sorted(importance.importances.items(), key=lambda kv: -kv[1])
        target_results = []
        for rank, (feature, score) in enumerate(ranked_features, start=1):
            kb_variable = FEATURE_TO_KB_VARIABLE.get(feature, feature)
            kb_hyp = _get_kb_hypothesis(con, kb_variable, target_metric)
            ml_dir = importance.directions.get(feature, "mixed")
            agreement, validation_status = _reconcile_with_kb(ml_dir, kb_hyp)

            cur.execute(
                """INSERT INTO environmental_influence_quantification
                   (run_id, target_metric, variable_name, selected_model_name,
                    ml_importance_score, ml_importance_rank, ml_importance_method, ml_direction,
                    pearson_r, spearman_rho, best_lag_days, best_lag_correlation,
                    seasonal_breakdown_json, kb_hypothesis_direction, kb_hypothesis_confidence,
                    agreement_with_kb, validation_status_assigned, evidence_label, causation_caveat)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, target_metric, feature, best.model_name,
                    score, rank, importance.method, ml_dir,
                    correlations.get(feature, {}).get("pearson_r"),
                    correlations.get(feature, {}).get("spearman_rho"),
                    lags.get(feature, {}).get("best_lag_days"),
                    lags.get(feature, {}).get("best_lag_correlation"),
                    seasonal.get(feature),
                    kb_hyp["influence_direction"] if kb_hyp else None,
                    kb_hyp["confidence_level"] if kb_hyp else None,
                    agreement, validation_status,
                    "observed_ml_influence", CAUSATION_CAVEAT,
                ),
            )
            target_results.append({
                "feature": feature, "kb_variable": kb_variable, "rank": rank,
                "ml_importance": round(score, 5), "ml_direction": ml_dir,
                "agreement_with_kb": agreement, "validation_status": validation_status,
            })
        con.commit()

        return {
            "best_model": best.model_name,
            "best_model_r2": round(best.r2_mean, 4),
            "n_samples": len(X),
            "explainability_method": importance.method,
            "top_features": target_results[:5],
        }
    finally:
        con.close()


def complete_run(db_path: str, run_id: int) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("UPDATE module4_run_metadata SET run_completed_at = datetime('now') WHERE id = ?", (run_id,))
        con.commit()
    finally:
        con.close()


def run_influence_quantification(db_path: str, cv_folds: int = 5, benchmark_sample_size: int = 15000,
                                   verbose: bool = True) -> dict:
    """Convenience wrapper: runs all three targets back-to-back under one
    run_id. Equivalent to calling start_run() -> run_target() x3 ->
    complete_run() manually (which is what very large datasets or
    time-constrained execution environments may prefer to do instead, to
    checkpoint progress between targets)."""
    run_id = start_run(db_path)
    all_results = {}
    for target_metric in ["gcv", "moisture", "ash"]:
        all_results[target_metric] = run_target(
            db_path, run_id, target_metric, cv_folds=cv_folds,
            benchmark_sample_size=benchmark_sample_size, verbose=verbose,
        )
    complete_run(db_path, run_id)
    return {"run_id": run_id, "results": all_results}


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    result = run_influence_quantification(db_file)
    print("\n=== Module 4 run complete ===")
    print(json.dumps(result, indent=2, default=str))
