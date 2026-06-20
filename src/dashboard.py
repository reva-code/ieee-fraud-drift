"""
Streamlit dashboard — run with:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, PLOTS_DIR

st.set_page_config(page_title="Fraud Drift Monitor", layout="wide", page_icon="📡")

# ── Load results ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    drift_df   = pd.read_csv(RESULTS_DIR / "drift_results.csv")
    per_batch  = pd.read_csv(RESULTS_DIR / "strategy_per_batch.csv")
    summary_df = pd.read_csv(RESULTS_DIR / "strategy_summary.csv")
    vikor_df   = pd.read_csv(RESULTS_DIR / "vikor_results.csv")
    return drift_df, per_batch, summary_df, vikor_df


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📡 Continuous Learning under Distribution Drift")
st.markdown("**Dataset:** IEEE-CIS Fraud Detection  |  **Methods:** DDM · KL Divergence · KS Statistic · HATT · MLP + SHAP  |  **MCDM:** AHP + VIKOR")
st.divider()

try:
    drift_df, per_batch, summary_df, vikor_df = load_results()
except FileNotFoundError:
    st.error("Results not found. Run `python src/run_experiment.py` first.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Drift Detection", "⚡ ADWIN Strategies", "🏆 MCDM Ranking", "🧠 XAI", "📊 Summary"]
)


# ── Tab 1: Drift Detection ────────────────────────────────────────────────────

with tab1:
    st.subheader("Drift Detection Signals")
    method = st.selectbox("Select method", drift_df["method"].unique())
    sub = drift_df[drift_df["method"] == method]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["batch_id"], y=sub["score"],
                              mode="lines", name="Score", line=dict(width=2)))
    drift_pts = sub[sub["is_drift"]]
    fig.add_trace(go.Scatter(x=drift_pts["batch_id"], y=drift_pts["score"],
                              mode="markers", name="Drift Detected",
                              marker=dict(color="red", size=10, symbol="x")))
    fig.update_layout(xaxis_title="Batch ID", yaxis_title="Score",
                       height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Batches", len(sub))
    col2.metric("Drift Events", int(sub["is_drift"].sum()))
    col3.metric("Drift Rate", f"{sub['is_drift'].mean()*100:.1f}%")


# ── Tab 2: ADWIN Strategies ───────────────────────────────────────────────────

with tab2:
    st.subheader("F1 Score per ADWIN Window Strategy")

    fig = px.line(per_batch, x="batch_id", y="f1", color="strategy",
                   labels={"batch_id": "Batch ID", "f1": "F1 Score"},
                   template="plotly_white", height=420)
    drift_batches = per_batch[per_batch["drift_detected"]]["batch_id"].unique()
    for b in drift_batches[:30]:
        fig.add_vline(x=b, line_dash="dot", line_color="red", opacity=0.2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Retraining Window Size")
    fig2 = px.line(per_batch, x="batch_id", y="window_size", color="strategy",
                    labels={"batch_id": "Batch ID", "window_size": "Window Size"},
                    template="plotly_white", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Strategy Summary Table")
    st.dataframe(summary_df[["strategy", "mean_f1", "std_f1", "drift_events",
                               "max_window", "mean_inference_s"]].round(4),
                  use_container_width=True)


# ── Tab 3: MCDM ──────────────────────────────────────────────────────────────

with tab3:
    st.subheader("AHP + VIKOR Multi-Criteria Ranking")

    col_v, col_r = st.columns([2, 1])

    with col_v:
        fig = go.Figure()
        for col, name, color in [("S", "Group Utility (S)", "#5c9ee0"),
                                   ("R", "Individual Regret (R)", "#e05c5c"),
                                   ("Q", "VIKOR Score Q", "#5ce08a")]:
            fig.add_trace(go.Bar(x=vikor_df["strategy"], y=vikor_df[col],
                                  name=name, marker_color=color))
        fig.update_layout(barmode="group", template="plotly_white",
                           yaxis_title="Score", height=400,
                           title="VIKOR Decision Scores (lower Q = better)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### Final Ranking")
        for _, row in vikor_df.sort_values("Q").iterrows():
            medal = ["🥇", "🥈", "🥉", "4️⃣"][int(row["rank"]) - 1]
            st.markdown(f"{medal} **{row['strategy']}**  `Q={row['Q']:.4f}`")

    # Radar chart (static image)
    radar_path = PLOTS_DIR / "strategy_radar.png"
    if radar_path.exists():
        st.image(str(radar_path), caption="Normalised strategy performance across all criteria")


# ── Tab 4: XAI ────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("SHAP Feature Importance")
    col_a, col_b = st.columns(2)

    for col, label, filename in [
        (col_a, "Reference Window", "shap_importance_reference.png"),
        (col_b, "Drift Window",     "shap_importance_drift.png"),
    ]:
        path = PLOTS_DIR / filename
        if path.exists():
            col.image(str(path), caption=label)
        else:
            col.info(f"{filename} not found — run experiment first.")

    shift_path = PLOTS_DIR / "shap_importance_shift.png"
    if shift_path.exists():
        st.subheader("Feature Importance Shift Under Drift")
        st.image(str(shift_path),
                 caption="Features that gained (red) or lost (blue) importance after drift")


# ── Tab 5: Summary ────────────────────────────────────────────────────────────

with tab5:
    st.subheader("Experiment Summary")

    winner = vikor_df.sort_values("Q").iloc[0]["strategy"]
    best_f1_row = summary_df.loc[summary_df["mean_f1"].idxmax()]

    st.success(f"**Best MCDM-ranked strategy:** `{winner}` (lowest VIKOR Q-score)")
    st.info(f"**Highest avg F1:** `{best_f1_row['strategy']}` — F1 = {best_f1_row['mean_f1']:.4f}")

    drift_summary = (
        drift_df.groupby("method")["is_drift"].sum()
        .reset_index(name="drift_events_detected")
    )
    st.subheader("Drift Events by Method")
    st.bar_chart(drift_summary.set_index("method"))

    st.subheader("All Plots")
    for img in sorted(PLOTS_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem.replace("_", " ").title())
