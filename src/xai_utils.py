"""
XAI — SHAP-based explanations for MLP predictions under drift.

Generates:
  1. Global feature importance (bar chart)
  2. SHAP beeswarm plot
  3. Per-batch feature importance shift (how drift changes what the model relies on)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from config import PLOTS_DIR


def compute_shap(model, X_background: pd.DataFrame,
                  X_explain: pd.DataFrame, max_background: int = 200) -> np.ndarray:
    """Use KernelExplainer (model-agnostic) for MLP."""
    bg = shap.sample(X_background, min(max_background, len(X_background)))
    explainer = shap.KernelExplainer(model.predict_proba, bg)
    shap_values = explainer.shap_values(X_explain, silent=True)
    # Newer SHAP returns (n_samples, n_features, n_classes); older returns a list per class
    if isinstance(shap_values, list):
        return shap_values[1]           # list → pick fraud class
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]     # 3D array → slice fraud class
    return shap_values


def plot_global_importance(shap_values: np.ndarray, feature_names: list[str],
                            label: str = "reference") -> Path:
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.head(15).plot(kind="bar", ax=ax, color="#e05c5c")
    ax.set_title(f"Global Feature Importance (SHAP) — {label}", fontsize=14)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = PLOTS_DIR / f"shap_importance_{label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance_shift(shap_ref: np.ndarray, shap_drift: np.ndarray,
                                   feature_names: list[str]) -> Path:
    ref_imp  = np.abs(shap_ref).mean(axis=0)
    drft_imp = np.abs(shap_drift).mean(axis=0)
    delta = drft_imp - ref_imp

    df = pd.DataFrame({"feature": feature_names, "delta": delta})
    df = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(15)

    colors = ["#e05c5c" if d > 0 else "#5c9ee0" for d in df["delta"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"], df["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Importance Shift Under Drift (post − pre)", fontsize=14)
    ax.set_xlabel("ΔSHAP importance")
    plt.tight_layout()

    path = PLOTS_DIR / "shap_importance_shift.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_xai_pipeline(model, ref_X: pd.DataFrame,
                      drift_X: pd.DataFrame) -> dict:
    """Run full XAI pipeline and save plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = list(ref_X.columns)

    print("  Computing SHAP for reference window...")
    shap_ref = compute_shap(model, ref_X, ref_X.sample(min(100, len(ref_X)), random_state=42),
                             max_background=100)

    print("  Computing SHAP for drift window...")
    shap_drift = compute_shap(model, ref_X, drift_X.sample(min(100, len(drift_X)), random_state=42),
                               max_background=100)

    p1 = plot_global_importance(shap_ref,   feature_names, label="reference")
    p2 = plot_global_importance(shap_drift, feature_names, label="drift")
    p3 = plot_feature_importance_shift(shap_ref, shap_drift, feature_names)

    print(f"  Saved: {p1.name}, {p2.name}, {p3.name}")
    return {"shap_ref": shap_ref, "shap_drift": shap_drift, "plots": [p1, p2, p3]}
