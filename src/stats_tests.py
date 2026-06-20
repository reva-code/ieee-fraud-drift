"""
Statistical significance tests for ADWIN strategy comparison.

Wilcoxon signed-rank test is used (non-parametric, paired) because:
  - F1 scores per batch are not normally distributed
  - Each batch is evaluated by all strategies → paired samples
  - We compare every adaptive strategy against No_Retrain baseline
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare


def run_significance_tests(per_batch_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    strategies = per_batch_df["strategy"].unique().tolist()
    # Align on common batch_ids so comparisons are truly paired
    pivot = (per_batch_df.pivot(index="batch_id", columns="strategy", values="f1")
                         .dropna())

    records = []

    # Wilcoxon between every pair of strategies
    for s1, s2 in itertools.combinations(strategies, 2):
        x, y = pivot[s1].values, pivot[s2].values
        if np.allclose(x, y):
            stat, p_val = np.nan, 1.0
        else:
            stat, p_val = wilcoxon(x, y, alternative="two-sided")
        records.append({
            "strategy_A": s1,
            "strategy_B": s2,
            "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else np.nan,
            "p_value": round(float(p_val), 4),
            "significant": p_val < alpha,
            "better": s1 if pivot[s1].mean() > pivot[s2].mean() else s2,
        })

    # Friedman test across all strategies simultaneously
    group_arrays = [pivot[s].values for s in strategies]
    f_stat, f_p = friedmanchisquare(*group_arrays)
    print(f"  Friedman test across all strategies: χ²={f_stat:.4f}  p={f_p:.4f} "
          f"({'significant' if f_p < alpha else 'not significant'})")

    return pd.DataFrame(records)
