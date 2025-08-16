#!/usr/bin/env python3
"""
Streamlit dashboard to explore results produced by the pipeline.

Repo layout assumed:
repo-root/
├─ app/streamlit_app.py         <-- this file
├─ src/utils_viz.py
├─ results/summary.csv
└─ results/pred_sub*_*.parquet

Run locally:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# ---------- Make src/ importable when app/ is the CWD ----------
APP_DIR = Path(__file__).resolve().parent            # .../repo-root/app
REPO_ROOT = APP_DIR.parent                           # .../repo-root
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------- Imports (now that sys.path includes src/) ----------
import pandas as pd
import streamlit as st
import plotly.express as px
from utils_viz import confmat_fig  # from src/utils_viz.py

# ---------- Config ----------
st.set_page_config(page_title="EEG-ImageNet Results", layout="wide")

# Allow overriding paths via env vars (useful on cloud)
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", REPO_ROOT / "results")).resolve()
PLOTS_DIR   = Path(os.getenv("PLOTS_DIR",   REPO_ROOT / "plots")).resolve()

# ---------- Guards / helpful messages ----------
# Optional guard to prevent running via `python` instead of `streamlit run`
try:
    from streamlit.runtime.scriptrun_context import get_script_run_ctx
    if get_script_run_ctx() is None:
        import sys as _sys
        print("This app must be launched with: streamlit run app/streamlit_app.py")
        _sys.exit(0)
except Exception:
    pass

if not RESULTS_DIR.exists():
    st.error(f"Results directory not found: {RESULTS_DIR}\n"
             "Make sure you've committed results/summary.csv and pred_sub*.parquet.")
    st.stop()

SUMMARY_CSV = RESULTS_DIR / "summary.csv"
if not SUMMARY_CSV.exists():
    st.error(f"Missing summary file: {SUMMARY_CSV}\n"
             "Run training + evaluation locally and commit the results.")
    st.stop()

# ---------- Cached data loaders ----------
@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure expected dtypes
    if "subject" in df.columns:
        df["subject"] = pd.to_numeric(df["subject"], errors="coerce").astype("Int64")
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def list_pred_files(results_dir: Path) -> dict[tuple[int, str], Path]:
    """Map (subject, model) -> parquet path"""
    mapping = {}
    for pf in results_dir.glob("pred_sub*_*.parquet"):
        stem = pf.stem  # e.g., pred_sub12_rf
        parts = stem.split("_")
        # expected: ["pred", "sub12", "rf"]
        if len(parts) >= 3 and parts[0] == "pred" and parts[1].startswith("sub"):
            try:
                sub = int(parts[1].replace("sub", ""))
                model = "_".join(parts[2:])  # supports model names w/ underscores
                mapping[(sub, model)] = pf
            except ValueError:
                continue
    return mapping

@st.cache_data(show_spinner=False)
def load_preds(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

# ---------- UI ----------
st.title("🧠 EEG-ImageNet — Lightweight Models Dashboard")

summary = load_summary(SUMMARY_CSV)
pred_map = list_pred_files(RESULTS_DIR)

# KPIs
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Subjects", int(summary["subject"].nunique()))
with k2:
    st.metric("Models", int(summary["model"].nunique()))
with k3:
    st.metric("Experiments", len(summary))

# Accuracy by model
st.subheader("Accuracy by Model")
fig_box = px.box(summary, x="model", y="accuracy", points="all")
st.plotly_chart(fig_box, use_container_width=True)

# Per-subject accuracy (grouped)
st.subheader("Per-Subject Accuracy by Model")
pivot = summary.pivot_table(index="subject", columns="model", values="accuracy")
fig_bar = px.bar(
    pivot.reset_index().melt(id_vars="subject", var_name="model", value_name="accuracy"),
    x="subject", y="accuracy", color="model", barmode="group"
)
st.plotly_chart(fig_bar, use_container_width=True)

# Drilldown
st.subheader("Dive Deeper")
colA, colB = st.columns(2)
with colA:
    subject = st.selectbox("Subject", sorted([int(s) for s in summary["subject"].dropna().unique()]))
with colB:
    # Only show models that appear for the selected subject
    models_for_sub = sorted(summary.loc[summary["subject"] == subject, "model"].unique())
    model = st.selectbox("Model", models_for_sub)

key = (int(subject), str(model))
pf = pred_map.get(key)

if pf and pf.exists():
    dfp = load_preds(pf)
    st.markdown(f"**Subject {subject} — Model {model}**")

    # Confusion matrix
    if {"y", "yhat"}.issubset(dfp.columns):
        fig_cm = confmat_fig(dfp["y"].astype(str).tolist(),
                             dfp["yhat"].astype(str).tolist(),
                             title=f"Confusion Matrix — Subject {subject} · {model}")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning(f"`y`/`yhat` columns not found in {pf.name}")

    # Misclassifications table
    if {"y", "yhat"}.issubset(dfp.columns):
        dfp = dfp.copy()
        dfp["correct"] = (dfp["y"] == dfp["yhat"])
        mis = dfp.loc[~dfp["correct"], ["y", "yhat"]].value_counts().reset_index(name="count")
        st.write("**Misclassifications**")
        if mis.empty:
            st.info("No misclassifications for this selection.")
        else:
            st.dataframe(mis.rename(columns={"y": "True", "yhat": "Predicted"}),
                         use_container_width=True)

    # Top-k expander (if present)
    if "top3" in dfp.columns or "top5" in dfp.columns:
        with st.expander("Top-k Predictions (first 50 rows)"):
            st.dataframe(dfp.head(50), use_container_width=True)
else:
    st.info("No prediction file found for this selection. "
            "Check that files are named like `pred_sub{ID}_{model}.parquet` in results/.")
