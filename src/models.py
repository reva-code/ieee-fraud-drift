import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from river import tree, drift, metrics as river_metrics
from config import MLP_HIDDEN_LAYERS, MLP_MAX_ITER, MLP_RANDOM_STATE


# ── MLP (scikit-learn) ────────────────────────────────────────────────────────

def train_mlp(X: pd.DataFrame, y: pd.Series):
    """Train MLP once on reference window with scaling + oversampling."""
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=MLP_RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    clf = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER,
        random_state=MLP_RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    clf.fit(X_scaled, y_res)
    clf._scaler = scaler
    return clf


def train_fast(X: pd.DataFrame, y: pd.Series):
    """
    HistGradientBoostingClassifier for ADWIN retraining loop.
    Chosen over LR because:
      - Handles class imbalance natively via class_weight
      - 5-10x better F1 than LR on tabular fraud data
      - No scaling needed — tree-based model
      - Fast: ~1-3s on 10k rows
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=MLP_RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf


def _predict(clf, X: pd.DataFrame) -> np.ndarray:
    """Predict — handles both scaler-attached models (MLP) and tree models."""
    scaler = getattr(clf, "_scaler", None)
    X_in = scaler.transform(X) if scaler is not None else X
    return clf.predict(X_in)


def evaluate_mlp(clf: MLPClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = _predict(clf, X)
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y, y_pred, zero_division=0),
    }


# ── HATT (Hoeffding Adaptive Tree) ───────────────────────────────────────────

class HATTWrapper:
    """
    Wraps river's HoeffdingAdaptiveTreeClassifier for batch-mode evaluation.
    HATT natively handles concept drift via ADWIN-based branch replacement.
    """

    def __init__(self):
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            drift_detector=drift.ADWIN(),
            leaf_prediction="nba",
        )
        self.metric = river_metrics.F1()

    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        for xi, yi in zip(X.to_dict(orient="records"), y):
            self.model.learn_one(xi, int(yi))

    def predict_batch(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds, trues = [], []
        # Convert once to avoid repeated .to_dict overhead on large frames
        records = X.to_dict(orient="records")
        for xi, yi in zip(records, y):
            p = self.model.predict_one(xi)
            preds.append(p if p is not None else 0)
            trues.append(int(yi))
            self.model.learn_one(xi, int(yi))

        f1 = float(f1_score(trues, preds, zero_division=0))
        return {"f1": f1, "preds": np.array(preds), "trues": np.array(trues)}
