# Setup & Run Guide

## 1. Get the data

```bash
# Option A — Kaggle CLI (recommended)
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip ieee-fraud-detection.zip && cd ..

# Option B — Manual download
# Go to https://www.kaggle.com/c/ieee-fraud-detection/data
# Download train_transaction.csv and train_identity.csv into data/
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the experiment

```bash
python src/run_experiment.py
```

This runs all 7 steps (preprocessing → drift detection → ADWIN strategies → MCDM → XAI) and saves results to `results/`.

## 4. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Opens at http://localhost:8501

---

## Project structure

```
ieee-fraud-drift/
├── data/                        ← Kaggle CSVs go here
├── results/
│   ├── plots/                   ← All generated figures (PNG)
│   ├── drift_results.csv
│   ├── strategy_per_batch.csv
│   ├── strategy_summary.csv
│   ├── vikor_results.csv
│   └── hatt_results.csv
├── src/
│   ├── config.py                ← All tunable parameters
│   ├── preprocessing.py         ← Load, sort, batch
│   ├── drift_detection.py       ← DDM, KL Divergence, KS Statistic
│   ├── models.py                ← MLP + HATT
│   ├── adwin_strategies.py      ← W+1, W×2, W÷2, W_fixed comparison
│   ├── mcdm.py                  ← AHP weights + VIKOR ranking
│   ├── xai_utils.py             ← SHAP explanations
│   ├── visualizations.py        ← Publication-quality static plots
│   ├── dashboard.py             ← Streamlit dashboard
│   └── run_experiment.py        ← Main entry point
└── requirements.txt
```

## Key parameters (src/config.py)

| Parameter | Default | What it controls |
|---|---|---|
| `BATCH_SIZE` | 5000 | Rows per time batch |
| `REFERENCE_BATCHES` | 10 | Batches used as reference window |
| `KL_BINS` | 50 | Histogram bins for KL divergence |
| `AHP_PAIRWISE` | See config | Expert pairwise comparison matrix |
| `MLP_HIDDEN_LAYERS` | (128,64,32) | MLP architecture |
