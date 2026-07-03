"""
Module 5 -- Prediction Engine
Model deployment.

Takes the Module 4 GroupKFold-validated best model per target (read from
model_benchmark_results for a given source run_id) and:
  1. Re-derives the exact same model (same hyperparameters, same
     random_state) via model_benchmarking._build_model_registry -- the
     model FAMILY and hyperparameters are never re-decided here.
  2. Computes HONEST out-of-fold residuals using the same GroupKFold(mine_code,
     n_splits=5) split Module 4 validated with (via cross_val_predict),
     so prediction intervals reflect genuine held-out uncertainty, not
     training-set residuals (which would understate uncertainty).
  3. Refits the model on the FULL target-specific dataset (same as Module 4
     did before SHAP/importance extraction) -- this is the artifact that
     gets serialized and deployed.
  4. Serializes the fitted model to disk (joblib) and persists metadata
     (feature column order, interval offsets, CV metrics, training
     provenance) to prediction_models.

This is deployment, not re-validation -- if you want to change which model
family is used, that is a Module 4 decision (re-run influence
quantification with different settings), not something this file decides.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from ..influence_quantification.data_assembly import assemble_dataset
from ..influence_quantification.model_benchmarking import _build_model_registry

DEFAULT_INTERVAL_LOWER_QUANTILE = 0.10
DEFAULT_INTERVAL_UPPER_QUANTILE = 0.90
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


def _get_selected_model_info(con: sqlite3.Connection, source_run_id: int, target_metric: str) -> dict:
    cur = con.cursor()
    cur.execute(
        """
        SELECT model_name, model_implementation, r2_mean, rmse_mean, mae_mean, cv_folds, training_sample_size
        FROM model_benchmark_results
        WHERE run_id = ? AND target_metric = ? AND is_selected_best = 1
        """,
        (source_run_id, target_metric),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No selected-best model found for run_id={source_run_id}, target_metric={target_metric}")
    return {
        "model_name": row[0], "model_implementation": row[1],
        "r2_mean": row[2], "rmse_mean": row[3], "mae_mean": row[4],
        "cv_folds": row[5], "training_sample_size": row[6],
    }


def _compute_oof_residual_offsets(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str,
                                    cv_folds: int = 5, random_state: int = 42,
                                    lower_q: float = DEFAULT_INTERVAL_LOWER_QUANTILE,
                                    upper_q: float = DEFAULT_INTERVAL_UPPER_QUANTILE) -> tuple[float, float]:
    """
    Returns (residual_lower_offset, residual_upper_offset) -- additive
    offsets to apply to a point estimate to form the prediction interval,
    derived from the [lower_q, upper_q] quantiles of (actual - predicted)
    on HELD-OUT folds of the same GroupKFold split Module 4 validated with.
    Using OOF predictions (not training-set residuals) is what makes this
    interval honest -- a model's error on data it was trained on is always
    optimistic.
    """
    registry = _build_model_registry(random_state=random_state)
    estimator, _, _ = registry[model_name]

    n_groups = groups.nunique()
    effective_folds = min(cv_folds, n_groups)
    gkf = GroupKFold(n_splits=effective_folds)

    oof_predictions = cross_val_predict(estimator, X, y, groups=groups, cv=gkf, n_jobs=1)
    residuals = y.values - oof_predictions

    lower_offset = float(np.quantile(residuals, lower_q))
    upper_offset = float(np.quantile(residuals, upper_q))
    return lower_offset, upper_offset


def deploy_model_for_target(db_path: str, source_run_id: int, target_metric: str,
                              models_dir: str = DEFAULT_MODELS_DIR, random_state: int = 42,
                              verbose: bool = True) -> int:
    """
    Deploys (refits + serializes + persists metadata for) the production
    model for one target_metric, sourced from a specific Module 4 run_id.
    Returns the new prediction_models.id. Deactivates any previously
    active model for this target_metric (is_active=0) without deleting it
    -- full deployment history is preserved.
    """
    os.makedirs(models_dir, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        model_info = _get_selected_model_info(con, source_run_id, target_metric)
        if verbose:
            print(f"[{target_metric}] Deploying {model_info['model_name']} "
                  f"({model_info['model_implementation']}) from run_id={source_run_id}")

        ds = assemble_dataset(db_path, target_metric)
        X = ds.df[ds.feature_columns]
        y = ds.df["target"]
        groups = ds.df["mine_code"]

        lower_offset, upper_offset = _compute_oof_residual_offsets(
            X, y, groups, model_info["model_name"], cv_folds=model_info["cv_folds"], random_state=random_state,
        )
        if verbose:
            print(f"[{target_metric}] OOF residual interval offsets: [{lower_offset:.3f}, {upper_offset:.3f}]")

        registry = _build_model_registry(random_state=random_state)
        estimator, impl_label, _ = registry[model_info["model_name"]]
        fitted = estimator.fit(X, y)

        artifact_filename = f"{target_metric}_{model_info['model_name']}_run{source_run_id}.joblib"
        artifact_path = os.path.join(models_dir, artifact_filename)
        joblib.dump(fitted, artifact_path)
        if verbose:
            print(f"[{target_metric}] Model artifact saved: {artifact_path}")

        cur = con.cursor()
        cur.execute(
            "UPDATE prediction_models SET is_active = 0 WHERE target_metric = ? AND is_active = 1",
            (target_metric,),
        )

        cur.execute(
            """
            INSERT INTO prediction_models
                (target_metric, source_run_id, model_name, model_implementation, model_artifact_path,
                 feature_columns_json, cv_r2_mean, cv_rmse_mean, cv_mae_mean,
                 interval_lower_quantile, interval_upper_quantile,
                 residual_lower_offset, residual_upper_offset,
                 training_sample_size, random_state, is_active, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                target_metric, source_run_id, model_info["model_name"], impl_label, artifact_path,
                json.dumps(ds.feature_columns), model_info["r2_mean"], model_info["rmse_mean"], model_info["mae_mean"],
                DEFAULT_INTERVAL_LOWER_QUANTILE, DEFAULT_INTERVAL_UPPER_QUANTILE,
                lower_offset, upper_offset,
                len(X), random_state,
                f"Deployed from Module 4 run_id={source_run_id} (GroupKFold-validated). "
                f"Interval offsets derived from honest out-of-fold residuals on the same GroupKFold(mine_code) split.",
            ),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def deploy_all_targets(db_path: str, source_run_id: int, models_dir: str = DEFAULT_MODELS_DIR,
                         random_state: int = 42, verbose: bool = True) -> dict:
    results = {}
    for target_metric in ["gcv", "moisture", "ash"]:
        model_id = deploy_model_for_target(db_path, source_run_id, target_metric, models_dir, random_state, verbose)
        results[target_metric] = model_id
    return results


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    run_id = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    target = sys.argv[3] if len(sys.argv) > 3 else None
    if target:
        model_id = deploy_model_for_target(db_file, run_id, target)
        print(f"Deployed prediction_models.id={model_id} for target={target}")
    else:
        results = deploy_all_targets(db_file, run_id)
        print("Deployed:", results)
