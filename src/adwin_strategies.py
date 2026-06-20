"""
Compares four retraining window strategies after drift is detected.

After ADWIN signals drift, the question is: how many recent samples should
we retrain on?
  W+1   — add exactly 1 new batch  (conservative, slow adaptation)
  W×2   — double the window        (aggressive, more historical context)
  W÷2   — halve the window         (focus tightly on recent data)
  W_fix — keep window size fixed   (baseline)
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score
from river import drift as river_drift
from models import train_fast, _predict
from config import ADWIN_STRATEGIES, REFERENCE_BATCHES, BATCH_SIZE, ADWIN_INIT_WINDOW

# Only evaluate this many post-reference batches per strategy (keeps runtime ~minutes)
MAX_EVAL_BATCHES = 30


def _retrain_on_window(all_batches: list[dict], end_idx: int, window_size: int):
    """Collect `window_size` most recent samples ending at `end_idx` and retrain."""
    rows_X, rows_y = [], []
    collected = 0
    for i in range(end_idx, -1, -1):
        rows_X.insert(0, all_batches[i]["X"])
        rows_y.insert(0, all_batches[i]["y"])
        collected += len(all_batches[i]["X"])
        if collected >= window_size:
            break
    X = pd.concat(rows_X, ignore_index=True).iloc[-window_size:]
    y = pd.concat(rows_y, ignore_index=True).iloc[-window_size:]
    return train_fast(X, y)


def run_strategy(strategy_name: str, window_fn, batches: list[dict],
                  ref_X: pd.DataFrame, ref_y: pd.Series) -> dict:
    """
    Simulate streaming evaluation for one ADWIN window strategy.
    All strategies use LogisticRegression as base learner for fair comparison.
    window_fn=None means no retraining (static baseline).
    """
    adwin = river_drift.ADWIN()
    current_window = ADWIN_INIT_WINDOW

    # Seed model on the most recent ADWIN_INIT_WINDOW rows of the reference window
    seed_X = ref_X.iloc[-ADWIN_INIT_WINDOW:]
    seed_y = ref_y.iloc[-ADWIN_INIT_WINDOW:]
    model = train_fast(seed_X, seed_y)

    results = []
    retrain_times = []
    drift_reactions = []

    eval_batches = batches[REFERENCE_BATCHES: REFERENCE_BATCHES + MAX_EVAL_BATCHES]
    for batch in eval_batches:
        X_cur = batch["X"]
        y_cur = batch["y"]
        bid = batch["batch_id"]

        t0 = time.perf_counter()
        y_pred = _predict(model, X_cur)
        inf_time = time.perf_counter() - t0

        drift_detected = False
        for err in (y_pred != y_cur.values).astype(int):
            adwin.update(float(err))
            if adwin.drift_detected:
                drift_detected = True

        if drift_detected and window_fn is not None:
            new_window = window_fn(current_window)
            t_retrain = time.perf_counter()
            model = _retrain_on_window(batches, bid - 1, new_window)
            retrain_time = time.perf_counter() - t_retrain
            retrain_times.append(retrain_time)
            drift_reactions.append({"batch_id": bid, "retrain_time": retrain_time,
                                     "new_window": new_window})
            current_window = new_window
            y_pred = _predict(model, X_cur)

        f1 = float(f1_score(y_cur, y_pred, zero_division=0))

        results.append({
            "batch_id": bid,
            "strategy": strategy_name,
            "f1": f1,
            "drift_detected": drift_detected,
            "window_size": current_window,
            "inference_time_s": inf_time,
        })

    df = pd.DataFrame(results)
    summary = {
        "strategy": strategy_name,
        "mean_f1": float(df["f1"].mean()),
        "std_f1": float(df["f1"].std()),
        "drift_events": int(df["drift_detected"].sum()),
        "mean_window": float(df["window_size"].mean()),
        "max_window": int(df["window_size"].max()),
        "mean_inference_s": float(df["inference_time_s"].mean()),
        "mean_retrain_s": float(np.mean(retrain_times)) if retrain_times else 0.0,
        "drift_reactions": drift_reactions,
    }
    return {"per_batch": df, "summary": summary}


def run_all_strategies(batches: list[dict],
                        ref_X: pd.DataFrame, ref_y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all ADWIN strategies and return (per_batch_df, summary_df)."""
    all_per_batch, all_summaries = [], []

    for name, fn in ADWIN_STRATEGIES.items():
        print(f"  Running strategy: {name}")
        result = run_strategy(name, fn, batches, ref_X, ref_y)
        all_per_batch.append(result["per_batch"])
        all_summaries.append(result["summary"])

    per_batch_df = pd.concat(all_per_batch, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    return per_batch_df, summary_df
