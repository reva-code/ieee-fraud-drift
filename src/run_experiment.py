"""
Main experiment runner. Execute from the project root:
    python src/run_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import pandas as pd
from config import RESULTS_DIR, REFERENCE_BATCHES, PLOTS_DIR
from preprocessing import prepare_data
from models import train_fast, train_mlp, evaluate_mlp, HATTWrapper, _predict
from drift_detection import run_all_detectors
from adwin_strategies import run_all_strategies
from mcdm import run_mcdm
from stats_tests import run_significance_tests
from xai_utils import run_xai_pipeline
from visualizations import (
    plot_drift_signals, plot_strategy_f1,
    plot_strategy_radar, plot_vikor_scores, plot_window_size_over_time,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Data Preprocessing")
    print("=" * 60)
    batches, ref_X, ref_y = prepare_data()

    # ── 2. Baseline LR — same model family as ADWIN strategies ───────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Baseline LR Training (consistent with ADWIN loop)")
    print("=" * 60)
    baseline_model = train_fast(ref_X, ref_y)
    metrics = evaluate_mlp(baseline_model, ref_X, ref_y)
    print(f"  Baseline LR F1 (reference window, class_weight=balanced): {metrics['f1']:.4f}")

    # ── 3. HATT streaming baseline ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — HATT Streaming Baseline")
    print("=" * 60)
    hatt = HATTWrapper()
    hatt.partial_fit(ref_X, ref_y)
    hatt_results = []
    hatt_batches = batches[REFERENCE_BATCHES: REFERENCE_BATCHES + 30]
    for batch in hatt_batches:
        res = hatt.predict_batch(batch["X"], batch["y"])
        hatt_results.append({"batch_id": batch["batch_id"], "f1": res["f1"]})
    hatt_df = pd.DataFrame(hatt_results)
    print(f"  HATT mean F1 (stream, 30 batches): {hatt_df['f1'].mean():.4f}")
    hatt_df.to_csv(RESULTS_DIR / "hatt_results.csv", index=False)

    # ── 4. Drift Detection ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Drift Detection (DDM, KL, KS)")
    print("=" * 60)
    drift_df = run_all_detectors(batches, ref_X, baseline_model, REFERENCE_BATCHES)
    drift_df.to_csv(RESULTS_DIR / "drift_results.csv", index=False)
    print(drift_df.groupby("method")["is_drift"].sum().rename("drift_events").to_string())
    plot_drift_signals(drift_df)

    # ── 5. ADWIN Strategy Comparison ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — ADWIN Window Strategy Comparison")
    print("=" * 60)
    per_batch_df, summary_df = run_all_strategies(batches, ref_X, ref_y)
    per_batch_df.to_csv(RESULTS_DIR / "strategy_per_batch.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "strategy_summary.csv", index=False)
    print("\nStrategy summary:")
    print(summary_df[["strategy", "mean_f1", "std_f1", "drift_events",
                       "max_window", "mean_retrain_s"]].to_string(index=False))
    plot_strategy_f1(per_batch_df)
    plot_window_size_over_time(per_batch_df)
    plot_strategy_radar(summary_df)

    # ── 6. Statistical Significance Tests ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Wilcoxon Signed-Rank Tests")
    print("=" * 60)
    sig_df = run_significance_tests(per_batch_df)
    sig_df.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    print(sig_df.to_string(index=False))

    # ── 7. MCDM — AHP + VIKOR ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — AHP + VIKOR MCDM Ranking")
    print("=" * 60)
    ahp_weights, vikor_df = run_mcdm(summary_df)
    vikor_df.to_csv(RESULTS_DIR / "vikor_results.csv", index=False)
    plot_vikor_scores(vikor_df)

    # ── 8. XAI (MLP+SHAP — separate from comparison models) ──────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — XAI (SHAP on MLP, separate from LR comparison)")
    print("=" * 60)
    mlp_model = train_mlp(ref_X, ref_y)
    print(f"  MLP F1 on reference: {evaluate_mlp(mlp_model, ref_X, ref_y)['f1']:.4f}")
    detected = drift_df[(drift_df["is_drift"]) & (drift_df["method"] == "KS_Statistic")]
    drift_X = (batches[int(detected.iloc[0]["batch_id"])]["X"]
               if len(detected) > 0 else batches[-1]["X"])
    run_xai_pipeline(mlp_model, ref_X, drift_X)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPLETE — Results saved to results/")
    print(f"  CSV files  : {RESULTS_DIR}")
    print(f"  Plot files : {PLOTS_DIR}")
    print("\nLaunch dashboard:  streamlit run src/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
