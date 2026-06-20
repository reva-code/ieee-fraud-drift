# MCDM-Guided Adaptive Learning Under Distribution Drift
### Fraud Detection in Non-Stationary Data Streams
Dataset: IEEE-CIS Fraud Detection (590k transactions)

---

## Overview

Static fraud detection models degrade silently as fraud patterns evolve over time — a phenomenon known as **concept drift** and **distribution drift**. This project proposes a framework that:

1. **Detects drift** using three complementary methods: DDM, KL Divergence, and KS Statistic
2. **Compares adaptive retraining strategies** triggered by ADWIN drift signals
3. **Ranks strategies** using Multi-Criteria Decision Making (AHP + VIKOR)
4. **Explains model behaviour** under drift using SHAP (XAI)
5. **Visualises everything** in an interactive Streamlit dashboard

The core novelty: applying **AHP + VIKOR** (MCDM) to select the optimal ADWIN window retraining strategy — the first framework to treat drift response as a multi-criteria decision problem.

---

## Key Results

| Method | Drift Events Detected |
|---|---|
| KS Statistic (Bonferroni-corrected) | 106 / 109 batches |
| DDM | 2 |
| KL Divergence | 1 |

| Strategy | Mean F1 | Drift Events | VIKOR Rank |
|---|---|---|---|
| W_times_2 | 0.290 | 15 | 1st |
| W_fixed | 0.333 | 8 | 2nd |
| W_plus_1 | 0.268 | 10 | 3rd |
| No_Retrain | 0.193 | 11 | 4th |
| W_div_2 | 0.180 | 11 | 5th |

- **Friedman test**: chi-squared = 72.39, p < 0.0001
- **Wilcoxon**: 9/10 pairwise comparisons statistically significant (alpha = 0.05)
- **AHP consistency ratio**: CR = 0.028 (well under 0.10 threshold)

---

## Architecture

```
ieee-fraud-drift/
├── src/
│   ├── config.py              # All tunable parameters
│   ├── preprocessing.py       # Load, sort by TransactionDT, batch into stream
│   ├── drift_detection.py     # DDM, KL Divergence, KS Statistic
│   ├── models.py              # HistGradientBoosting (ADWIN) + MLP (XAI)
│   ├── adwin_strategies.py    # 5 window strategies + ADWIN loop
│   ├── mcdm.py                # AHP weight derivation + VIKOR ranking
│   ├── stats_tests.py         # Wilcoxon signed-rank + Friedman tests
│   ├── xai_utils.py           # SHAP feature importance (pre/post drift)
│   ├── visualizations.py      # Publication-quality static plots
│   ├── run_experiment.py      # Single entry point — runs all 8 steps
│   └── dashboard.py           # Streamlit dashboard (5 tabs)
├── results/
│   ├── plots/                 # 8 PNG figures
│   ├── drift_results.csv
│   ├── strategy_summary.csv
│   ├── vikor_results.csv
│   └── significance_tests.csv
├── data/                      # Kaggle CSVs go here (not in repo, ~700MB)
├── requirements.txt
└── SETUP.md
```

---

## Methodology

### 1. Data Stream Simulation
IEEE-CIS transactions are sorted by `TransactionDT` and split into batches of 5,000 rows, simulating a real-time fraud detection stream. The first 10 batches (50k rows) form the reference window.

### 2. Drift Detection

| Method | What it measures |
|---|---|
| **DDM** | Classifier error rate increase (Gama et al. 2004) |
| **KL Divergence** | Distribution shift per feature (Freedman-Diaconis adaptive bins) |
| **KS Statistic** | Per-feature two-sample test (Bonferroni-corrected, majority vote) |

### 3. ADWIN Window Strategies
After ADWIN detects drift, the model retrains on a window of recent data. Five strategies are compared:

| Strategy | Window Update Rule |
|---|---|
| `No_Retrain` | Never retrain — static baseline |
| `W_plus_1` | Grow by one chunk (W + 10,000) |
| `W_times_2` | Double the window, capped at 40k rows |
| `W_div_2` | Halve the window, floored at 1k rows |
| `W_fixed` | Keep window size fixed |

Base learner: **HistGradientBoostingClassifier** with `class_weight="balanced"` to handle fraud class imbalance (~3.5% positive rate).

### 4. AHP + VIKOR (MCDM)
Five criteria are used to rank strategies:

| Criterion | Type | Rationale |
|---|---|---|
| F1 Score | Benefit | Primary performance metric |
| Drift Reaction Speed | Benefit | Faster detection = less performance loss |
| Memory Efficiency | Cost | Lower max window = lower memory footprint |
| Stability (1 - std F1) | Benefit | Consistent performance preferred |
| Retraining Time | Cost | Operational overhead |

AHP derives criteria weights from expert pairwise comparisons (CR = 0.028). VIKOR ranks alternatives by proximity to the ideal solution across all criteria simultaneously.

### 5. XAI — SHAP
SHAP KernelExplainer on MLP reveals which features change importance under drift. `TransactionAmt`, `C1`, and `D1` show the largest shifts between reference and drift windows.

### 6. Statistical Validation
- **Friedman test**: non-parametric omnibus test across all 5 strategies
- **Wilcoxon signed-rank test**: pairwise comparisons on paired F1 scores per batch

---

## Setup

### 1. Get the dataset
```bash
# Requires a Kaggle account and API token at ~/.kaggle/kaggle.json
python3 -m kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip ieee-fraud-detection.zip && cd ..
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full experiment
```bash
python3 src/run_experiment.py
```

Runs all 8 steps in sequence (~5-10 minutes). All results saved to `results/`.

### 4. Launch the interactive dashboard
```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501`

---

## Dashboard

| Tab | Content |
|---|---|
| Drift Detection | Interactive score plots for DDM / KL / KS with drift event markers |
| ADWIN Strategies | F1 over time per strategy + window size evolution |
| MCDM Ranking | VIKOR S/R/Q bar charts + radar chart across all criteria |
| XAI | SHAP importance pre/post drift + feature importance shift |
| Summary | Best strategy, drift event counts, all static plots |

---

## Technologies

`scikit-learn` · `river` · `shap` · `streamlit` · `plotly` · `scipy` · `pandas` · `matplotlib` · `seaborn`

---

[github.com/reva-code](https://github.com/reva-code)
