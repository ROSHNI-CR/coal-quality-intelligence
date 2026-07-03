"""
Module 4 -- Environmental Influence Quantification Engine
Explainability layer.

Produces feature importance for the SELECTED best model per target. SHAP
is the required primary method (per architecture). This sandbox does not
have network access to install the `shap` package (same constraint as
xgboost/lightgbm, documented in model_benchmarking.py).

Fallback: scikit-learn's permutation_importance, computed on a held-out
sample, used ONLY when the real `shap` library is unavailable. This is
clearly labeled in every output (`ml_importance_method` field) -- never
silently presented as SHAP. Permutation importance answers a related but
distinct question ("how much does shuffling this feature degrade model
performance") rather than SHAP's per-prediction attribution, and this
difference is preserved in the data rather than hidden.

If shap becomes available later, this module will automatically use
shap.TreeExplainer for tree-based models (random_forest, xgboost,
lightgbm) and shap.LinearExplainer for linear_regression -- no code
change needed elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class FeatureImportanceResult:
    method: str  # 'shap' | 'permutation_importance'
    importances: dict       # {feature_name: importance_score}
    directions: dict        # {feature_name: 'positive' | 'negative' | 'mixed'} -- best-effort direction signal


def _direction_from_correlation(X: pd.DataFrame, predictions: np.ndarray, feature: str) -> str:
    """Best-effort direction signal: correlate the feature's raw values
    against the model's predictions. Used identically regardless of
    whether SHAP or permutation importance produced the magnitude, since
    neither method directly outputs sign for non-linear models in a way
    that's simpler than this. For SHAP this is refined using the mean SHAP
    value's own sign where available."""
    corr = np.corrcoef(X[feature], predictions)[0, 1]
    if np.isnan(corr):
        return "mixed"
    if corr > 0.05:
        return "positive"
    if corr < -0.05:
        return "negative"
    return "mixed"


def compute_feature_importance(fitted_model, model_name: str, X: pd.DataFrame, y: pd.Series,
                                 random_state: int = 42, n_repeats: int = 5) -> FeatureImportanceResult:
    feature_names = list(X.columns)

    if HAS_SHAP:
        try:
            if model_name in ("random_forest", "xgboost", "lightgbm"):
                explainer = shap.TreeExplainer(fitted_model)
                shap_values = explainer.shap_values(X)
            else:
                explainer = shap.LinearExplainer(fitted_model, X)
                shap_values = explainer.shap_values(X)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            mean_signed_shap = np.array(shap_values).mean(axis=0)

            importances = {f: float(v) for f, v in zip(feature_names, mean_abs_shap)}
            directions = {
                f: ("positive" if s > 0.01 else "negative" if s < -0.01 else "mixed")
                for f, s in zip(feature_names, mean_signed_shap)
            }
            return FeatureImportanceResult(method="shap", importances=importances, directions=directions)
        except Exception:
            pass  # fall through to permutation importance if SHAP errors on this model type

    # --- fallback: permutation importance ---
    result = permutation_importance(
        fitted_model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1, scoring="r2"
    )
    importances = {f: float(v) for f, v in zip(feature_names, result.importances_mean)}

    predictions = fitted_model.predict(X)
    directions = {f: _direction_from_correlation(X, predictions, f) for f in feature_names}

    return FeatureImportanceResult(method="permutation_importance", importances=importances, directions=directions)


def get_shap_availability() -> dict:
    return {"shap_available": HAS_SHAP, "shap_substitute_used": None if HAS_SHAP else "sklearn.inspection.permutation_importance"}
