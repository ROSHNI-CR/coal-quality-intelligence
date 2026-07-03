"""
Module 4 -- Environmental Influence Quantification Engine
Model benchmarking layer.

Trains and cross-validates four model families per target metric (Linear
Regression, Random Forest, XGBoost, LightGBM), as required, and selects
the best model per target by mean CV R^2 (ties broken by lower RMSE).

METHODOLOGY (finalized per post-audit requirements):
  - Cross-validation uses GroupKFold(n_splits=5), grouped by mine_code.
    Every fold contains completely unseen mines -- no mine_code appears in
    both the train and validation portion of any fold. This replaces an
    earlier shuffled KFold pass (preserved in the database under an older
    run_id for comparison) that risked mine-level leakage: a model could
    partly learn a mine's own baseline GCV/moisture/ash level from other
    rows of that same mine in the training fold, inflating CV R^2 beyond
    what it would achieve on a genuinely unseen mine. GroupKFold gives a
    more realistic estimate of generalization to unseen mines, which is
    the deployment-relevant question for this platform (used to assess
    new/newly-mapped mines, not just resample existing ones).
  - Linear Regression is the only model wrapped in a
    Pipeline(StandardScaler, LinearRegression). Tree-based models
    (Random Forest, XGBoost/substitute, LightGBM/substitute) are
    scale-invariant by construction and are deliberately left unscaled.

ENVIRONMENT HONESTY NOTE: This sandbox does not have network access to
install the xgboost or lightgbm packages (confirmed -- same constraint
documented for FastAPI in PROJECT_STATUS.md Section 7). This module:
  - ALWAYS attempts to import the real xgboost/lightgbm/shap libraries first.
  - If unavailable, falls back to a clearly-labeled scikit-learn-native
    substitute from the same algorithm family:
      xgboost  -> sklearn.ensemble.GradientBoostingRegressor
      lightgbm -> sklearn.ensemble.HistGradientBoostingRegressor (a
                  histogram-based gradient boosting implementation,
                  algorithmically the closest native sklearn equivalent
                  to LightGBM's core technique)
  - NEVER silently relabels a substitute as the real thing. Every benchmark
    result row records `model_implementation` (the actual Python class
    used), and module4_run_metadata records exactly which substitutions
    were made, if any.
  - If this code is later run in an environment with real xgboost/lightgbm/
    shap installed, it will automatically use them -- no code change needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelResult:
    model_name: str               # canonical name: 'linear_regression' | 'random_forest' | 'xgboost' | 'lightgbm'
    model_implementation: str     # actual class used, e.g. 'sklearn.ensemble.HistGradientBoostingRegressor'
    is_substitute: bool
    cv_folds: int
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    training_sample_size: int
    fitted_model: object           # the final model, refit on full data, for SHAP/importance extraction


def _build_model_registry(random_state: int = 42) -> dict:
    """Returns {canonical_name: (estimator, implementation_label, is_substitute)}.

    Hyperparameters are deliberately modest (fewer trees/shallower depth
    than a production-tuned model would use) to keep cross-validated
    benchmarking tractable on the full ~90K-row dataset within this
    environment's compute constraints. This is a benchmarking pass to
    select the best MODEL FAMILY, not final hyperparameter tuning --
    once a model is selected per target, a follow-up tuning pass (e.g.
    grid/random search) on that single model is a natural next step and
    is noted as such in the run report rather than done speculatively
    here for all four candidates."""
    registry = {
        "linear_regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "sklearn.pipeline.Pipeline(StandardScaler, LinearRegression)", False,
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=random_state),
            "sklearn.ensemble.RandomForestRegressor", False,
        ),
    }

    if HAS_XGBOOST:
        registry["xgboost"] = (
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.15, random_state=random_state, n_jobs=-1),
            "xgboost.XGBRegressor", False,
        )
    else:
        registry["xgboost"] = (
            GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.15, random_state=random_state),
            "sklearn.ensemble.GradientBoostingRegressor (XGBoost unavailable in this environment)", True,
        )

    if HAS_LIGHTGBM:
        registry["lightgbm"] = (
            lgb.LGBMRegressor(n_estimators=100, max_depth=7, learning_rate=0.15, random_state=random_state, n_jobs=-1, verbosity=-1),
            "lightgbm.LGBMRegressor", False,
        )
    else:
        registry["lightgbm"] = (
            HistGradientBoostingRegressor(max_iter=100, max_depth=7, learning_rate=0.15, random_state=random_state),
            "sklearn.ensemble.HistGradientBoostingRegressor (LightGBM unavailable in this environment)", True,
        )

    return registry


def benchmark_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_folds: int = 5, random_state: int = 42,
                      benchmark_sample_size: Optional[int] = 15000) -> list[ModelResult]:
    """
    Cross-validates every registered model to select the best model FAMILY,
    using GroupKFold(n_splits=cv_folds) grouped by `groups` (mine_code).
    Every fold contains completely unseen mines -- no mine_code appears in
    both train and validation within any fold. The selected model is then
    refit on the FULL dataset (done by the caller via refit_on_full_data,
    not here) for downstream SHAP/importance extraction.

    groups: a Series aligned index-for-index with X/y (typically mine_code)
    used purely for fold assignment -- never used as a model feature.

    benchmark_sample_size: if the full dataset exceeds this size, a random
    subsample of this size is used for the cross-validated comparison pass
    only (group membership is preserved through the subsample -- a sampled
    row keeps its real mine_code, so GroupKFold on the subsample is still
    valid; some mines may simply be absent from the subsample, which does
    not violate the no-mine-overlap guarantee). This is standard practice
    for model-family selection on large datasets where the slower
    candidates (Random Forest, gradient boosting) would otherwise make a
    full 4-model x 3-target x 5-fold benchmark intractable. The model
    selected from this subsample comparison is then refit on the FULL
    dataset (see refit_on_full_data) before any SHAP/importance/prediction
    work happens -- so the final fitted model always sees all available
    data; only the *comparison* step is subsampled. Set to None to disable
    subsampling and use the full dataset for CV too.
    """
    registry = _build_model_registry(random_state=random_state)

    if benchmark_sample_size is not None and len(X) > benchmark_sample_size:
        sample_idx = X.sample(n=benchmark_sample_size, random_state=random_state).index
        X_bench, y_bench, groups_bench = X.loc[sample_idx], y.loc[sample_idx], groups.loc[sample_idx]
    else:
        X_bench, y_bench, groups_bench = X, y, groups

    n_unique_groups = groups_bench.nunique()
    effective_folds = min(cv_folds, n_unique_groups)
    gkf = GroupKFold(n_splits=effective_folds)

    scoring = {
        "rmse": make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
    }

    results = []
    for name, (estimator, impl_label, is_sub) in registry.items():
        cv_results = cross_validate(
            estimator, X_bench, y_bench, groups=groups_bench, cv=gkf, scoring=scoring, n_jobs=1
        )

        rmse_scores = -cv_results["test_rmse"]
        mae_scores = -cv_results["test_mae"]
        r2_scores = cv_results["test_r2"]

        results.append(ModelResult(
            model_name=name,
            model_implementation=impl_label,
            is_substitute=is_sub,
            cv_folds=effective_folds,
            rmse_mean=float(rmse_scores.mean()), rmse_std=float(rmse_scores.std()),
            mae_mean=float(mae_scores.mean()), mae_std=float(mae_scores.std()),
            r2_mean=float(r2_scores.mean()), r2_std=float(r2_scores.std()),
            training_sample_size=len(X_bench),
            fitted_model=None,  # not fit yet -- see refit_on_full_data
        ))

    return results


def refit_on_full_data(model_name: str, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Refits the named model on the FULL dataset (not the benchmark
    subsample) -- this is the model whose SHAP/importance values are
    actually reported. Returns (fitted_model, implementation_label)."""
    registry = _build_model_registry(random_state=random_state)
    estimator, impl_label, _ = registry[model_name]
    fitted = estimator.fit(X, y)
    return fitted, impl_label


def select_best_model(results: list[ModelResult]) -> ModelResult:
    """Best = highest mean CV R^2; ties broken by lower mean RMSE."""
    return sorted(results, key=lambda r: (-r.r2_mean, r.rmse_mean))[0]


def get_library_availability() -> dict:
    return {
        "xgboost_available": HAS_XGBOOST,
        "lightgbm_available": HAS_LIGHTGBM,
        "xgboost_substitute_used": None if HAS_XGBOOST else "sklearn.ensemble.GradientBoostingRegressor",
        "lightgbm_substitute_used": None if HAS_LIGHTGBM else "sklearn.ensemble.HistGradientBoostingRegressor",
    }
