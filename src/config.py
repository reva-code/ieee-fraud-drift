from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Temporal batching
BATCH_SIZE = 5000
REFERENCE_BATCHES = 10          # first 10 batches = reference window (~50k rows)
DRIFT_DETECTION_THRESHOLD = 0.05

# ADWIN initial retraining window (rows). Smaller than reference window so
# W_times_2 doublings stay tractable (max = 2 * 10000 = 20000 rows).
ADWIN_INIT_WINDOW = 10000

# ADWIN strategies — None means no retraining (static baseline)
ADWIN_STRATEGIES = {
    "No_Retrain": None,
    "W_plus_1":   lambda w: w + ADWIN_INIT_WINDOW,          # grow by one "chunk"
    "W_times_2":  lambda w: min(w * 2, 40000),              # double, cap at 40k
    "W_div_2":    lambda w: max(w // 2, 1000),              # halve, floor at 1k
    "W_fixed":    lambda w: w,                               # keep fixed
}

# MLP architecture (used only for XAI/SHAP)
MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_MAX_ITER = 200
MLP_RANDOM_STATE = 42

# KL divergence fallback bins (used only when FD rule fails)
KL_BINS = 50

# AHP pairwise matrix — 5 criteria: F1, Drift_Speed, Memory, Stability, Retrain_Time
# Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme preference
AHP_PAIRWISE = [
    [1,   3,   5,   3,   7],   # F1               (most important)
    [1/3, 1,   3,   2,   5],   # Drift_Speed
    [1/5, 1/3, 1,   1/2, 3],   # Memory
    [1/3, 1/2, 2,   1,   3],   # Stability
    [1/7, 1/5, 1/3, 1/3, 1],   # Retrain_Time     (least important)
]

CRITERIA_NAMES = ["F1_Score", "Drift_Speed", "Memory_Efficiency", "Stability", "Retrain_Time"]
STRATEGY_NAMES = list(ADWIN_STRATEGIES.keys())

TOP_FEATURES = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "C1", "C2", "C6", "C11",
    "D1", "D4", "D10", "D15", "V95", "V96", "V97",
    "V126", "V127", "V128", "V130", "V131",
]
