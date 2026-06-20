"""
Multi-Criteria Decision Making to rank ADWIN retraining strategies.

AHP  — derives criteria weights from expert pairwise comparisons.
VIKOR — ranks strategies by proximity to ideal solution across criteria.
"""

import numpy as np
import pandas as pd
from config import AHP_PAIRWISE, CRITERIA_NAMES, STRATEGY_NAMES


# ── AHP ───────────────────────────────────────────────────────────────────────

def ahp_weights(pairwise: list[list[float]] = AHP_PAIRWISE) -> np.ndarray:
    A = np.array(pairwise, dtype=float)
    n = A.shape[0]

    # Normalise each column then average rows
    col_sums = A.sum(axis=0)
    A_norm = A / col_sums
    weights = A_norm.mean(axis=1)

    # Consistency check (CR < 0.10 is acceptable)
    lambda_max = float((A @ weights / weights).mean())
    ci = (lambda_max - n) / (n - 1)
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0.0
    print(f"  AHP  λ_max={lambda_max:.4f}  CI={ci:.4f}  CR={cr:.4f} {'✓' if cr < 0.10 else '✗ inconsistent'}")

    return weights


# ── VIKOR ─────────────────────────────────────────────────────────────────────

def vikor_rank(decision_matrix: np.ndarray, weights: np.ndarray,
               benefit_mask: list[bool], v: float = 0.5) -> pd.DataFrame:
    """
    Parameters
    ----------
    decision_matrix : shape (n_strategies, n_criteria)
    weights         : AHP-derived criteria weights
    benefit_mask    : True = higher is better, False = lower is better
    v               : weight of group utility vs. individual regret (0.5 = balanced)
    """
    dm = decision_matrix.astype(float)
    n_strat, n_crit = dm.shape

    # Ideal best (f*) and worst (f-)
    f_best = np.where(benefit_mask, dm.max(axis=0), dm.min(axis=0))
    f_worst = np.where(benefit_mask, dm.min(axis=0), dm.max(axis=0))

    # Normalised weighted distances
    denom = f_best - f_worst
    denom[denom == 0] = 1e-10  # avoid division by zero

    # S = group utility (sum of weighted distances)
    # R = individual regret (max weighted distance)
    S = np.sum(weights * (f_best - dm) / denom, axis=1)
    R = np.max(weights * (f_best - dm) / denom, axis=1)

    S_best, S_worst = S.min(), S.max()
    R_best, R_worst = R.min(), R.max()

    Q = v * (S - S_best) / (S_worst - S_best + 1e-10) + \
        (1 - v) * (R - R_best) / (R_worst - R_best + 1e-10)

    result = pd.DataFrame({
        "strategy": STRATEGY_NAMES,
        "S": S,
        "R": R,
        "Q": Q,
    }).sort_values("Q").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def run_mcdm(summary_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build decision matrix from strategy summary stats and run AHP + VIKOR.

    Criteria (5):
      F1_Score          → benefit (higher better)
      Drift_Speed       → benefit (more drift events caught ≈ faster reaction)
      Memory_Efficiency → cost   (lower max_window better)
      Stability         → benefit (lower std_f1 → more stable → invert)
      Retrain_Time      → cost   (lower mean_retrain_s better)
    """
    weights = ahp_weights()

    dm = np.column_stack([
        summary_df["mean_f1"].values,            # F1         — benefit
        summary_df["drift_events"].values,        # Drift speed— benefit
        summary_df["max_window"].values,          # Memory     — cost
        1 - summary_df["std_f1"].values,          # Stability  — benefit
        summary_df["mean_retrain_s"].values,      # Retrain time — cost
    ])

    benefit_mask = [True, True, False, True, False]
    vikor_df = vikor_rank(dm, weights, benefit_mask)

    print("\n  VIKOR ranking:")
    print(vikor_df[["rank", "strategy", "Q"]].to_string(index=False))
    return weights, vikor_df
