import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from dataclasses import dataclass, field
from config import KL_BINS
from models import _predict


@dataclass
class DriftResult:
    batch_id: int
    method: str
    score: float
    is_drift: bool
    feature_scores: dict = field(default_factory=dict)


# ── DDM (Drift Detection Method) ─────────────────────────────────────────────

class DDM:
    """Gama et al. 2004 — monitors classifier error rate over a stream."""

    def __init__(self, warning_level=2.0, drift_level=3.0, min_samples=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 1.0        # error rate
        self.s = 0.0        # std estimate
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, error: int) -> str:
        """Feed one prediction error (1=wrong, 0=correct). Returns 'drift'/'warning'/'normal'."""
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)

        if self.n < self.min_samples:
            return "normal"

        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            self.reset()
            return "drift"
        if self.p + self.s > self.p_min + self.warning_level * self.s_min:
            return "warning"
        return "normal"

    def evaluate_batch(self, errors: np.ndarray, batch_id: int) -> DriftResult:
        status_counts = {"drift": 0, "warning": 0, "normal": 0}
        last_status = "normal"
        for e in errors:
            last_status = self.update(int(e))
            status_counts[last_status] += 1

        is_drift = status_counts["drift"] > 0
        score = self.p + self.s  # higher = more likely drift
        return DriftResult(batch_id=batch_id, method="DDM", score=float(score), is_drift=is_drift)


# ── KL Divergence ─────────────────────────────────────────────────────────────

def _feature_kl(ref: np.ndarray, cur: np.ndarray) -> float:
    # Clip to 1st-99th percentile to prevent outliers collapsing all mass into one bin
    lo, hi = np.percentile(ref, [1, 99])
    if hi <= lo:
        return 0.0
    ref_c = np.clip(ref, lo, hi)
    cur_c = np.clip(cur, lo, hi)
    # Freedman-Diaconis rule picks bin width based on IQR — adapts to each feature's scale
    bin_edges = np.histogram_bin_edges(ref_c, bins="fd")
    if len(bin_edges) < 3:
        bin_edges = np.linspace(lo, hi, KL_BINS + 1)
    bin_edges = np.clip(bin_edges, lo, hi)
    p, _ = np.histogram(ref_c, bins=bin_edges)
    q, _ = np.histogram(cur_c, bins=bin_edges)
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def kl_divergence_batch(ref_X: pd.DataFrame, cur_X: pd.DataFrame,
                         batch_id: int, threshold: float = 0.5) -> DriftResult:
    feature_scores = {}
    for col in ref_X.columns:
        feature_scores[col] = _feature_kl(ref_X[col].values, cur_X[col].values)

    mean_kl = float(np.mean(list(feature_scores.values())))
    return DriftResult(
        batch_id=batch_id,
        method="KL_Divergence",
        score=mean_kl,
        is_drift=mean_kl > threshold,
        feature_scores=feature_scores,
    )


# ── KS Statistic ──────────────────────────────────────────────────────────────

def ks_statistic_batch(ref_X: pd.DataFrame, cur_X: pd.DataFrame,
                        batch_id: int, p_threshold: float = 0.05) -> DriftResult:
    feature_scores = {}
    drift_flags = []
    n_features = len(ref_X.columns)
    bonferroni = p_threshold / n_features  # Bonferroni correction for multiple comparisons
    for col in ref_X.columns:
        stat, p_val = stats.ks_2samp(ref_X[col].values, cur_X[col].values)
        feature_scores[col] = float(stat)
        drift_flags.append(p_val < bonferroni)

    mean_ks = float(np.mean(list(feature_scores.values())))
    is_drift = sum(drift_flags) > len(drift_flags) * 0.5  # majority of features drifting
    return DriftResult(
        batch_id=batch_id,
        method="KS_Statistic",
        score=mean_ks,
        is_drift=is_drift,
        feature_scores=feature_scores,
    )


# ── Run all detectors ─────────────────────────────────────────────────────────

def run_all_detectors(batches: list[dict], ref_X: pd.DataFrame,
                       model, reference_batches: int) -> pd.DataFrame:
    """Run DDM, KL, KS on every post-reference batch. Returns results DataFrame."""
    ddm = DDM()
    records = []

    for batch in batches[reference_batches:]:
        X_cur = batch["X"]
        y_cur = batch["y"]
        bid = batch["batch_id"]

        # DDM needs prediction errors
        y_pred = _predict(model, X_cur)
        errors = (y_pred != y_cur.values).astype(int)
        ddm_result = ddm.evaluate_batch(errors, bid)

        kl_result  = kl_divergence_batch(ref_X, X_cur, bid)
        ks_result  = ks_statistic_batch(ref_X, X_cur, bid)

        for r in [ddm_result, kl_result, ks_result]:
            records.append({
                "batch_id": r.batch_id,
                "method": r.method,
                "score": r.score,
                "is_drift": r.is_drift,
            })

    return pd.DataFrame(records)
