#!/usr/bin/env python3
"""
Streamlit dashboard to explore results produced by the pipeline.

Repo layout:
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

# ---------- Imports ----------
import pandas as pd
import streamlit as st
import plotly.express as px
from utils_viz import confmat_fig  # from src/utils_viz.py

# ---------- Config ----------
st.set_page_config(page_title="EEG-ImageNet Results", layout="wide")

# Allow overriding paths via env vars (useful on cloud)
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", REPO_ROOT / "results")).resolve()
PLOTS_DIR   = Path(os.getenv("PLOTS_DIR",   REPO_ROOT / "plots")).resolve()
ASSESTS_DIR = Path(os.getenv("ASSETS_DIR",   REPO_ROOT / "assets")).resolve()

# ---------- Guards / helpful messages ----------
# Guard to handle error of running via `python` instead of `streamlit run`
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
                model = "_".join(parts[2:])  # parse model names w/ underscores
                mapping[(sub, model)] = pf
            except ValueError:
                continue
    return mapping

@st.cache_data(show_spinner=False)
def load_preds(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

# ---------- UI ----------
st.title("🧠 EEG-ImageNet — Visualizing ML Model Classification Performance on Brainwaves")

summary = load_summary(SUMMARY_CSV)
pred_map = list_pred_files(RESULTS_DIR)

with st.expander("ℹ️ About this Dashboard", expanded=True):
    top_L, top_R = st.columns(2)
    with top_L:
        st.image(f'{ASSESTS_DIR}/dataset_overview_diagram.png')
    with top_R:
        st.markdown(
        """
        ### Overview  
        This dashboard visualizes the **accuracy** and **performance** of four machine learning models in classifying human brainwaves.  

        ### Models  
        The following models were trained and evaluated:  
        - **Logistic Regression (LR)**  
        - **Multi-Layer Perceptron (MLP)**  
        - **Random Forest (RF)**  
        - **Support Vector Machine (SVM)**  

        ### Dataset  
        The models were trained on EEG recordings from **two subjects** in the [EEG-ImageNet dataset](https://arxiv.org/abs/2406.07151v1).  
        Each subject was shown images while their **brainwaves were recorded**.  

        ### Results  
        The evaluation metrics for each model are visualized below.  
        """
    )
st.divider()


data_exp_tab, results_tab = st.tabs(['Data Exploration','Results'])
## Data Explorer: 
with data_exp_tab:
    # Show snippets of brainwaves for specific images: 
    
    ## Fish
    st.title('Goldfish Image/EEG Reading Pair - Subect 0')
    gf_snippet_L, gf_snippet_R = st.columns([0.20,0.8])   
    with gf_snippet_L: 
        st.subheader('Goldfish - n01443537_00020822')
        st.image(f'{ASSESTS_DIR}/ILSVRC2012_val_n01443537_00020822.JPEG')
        st.caption('Image shown to subject 0 to produce the EEG reading on the right')
    with gf_snippet_R:
        with open(f"{PLOTS_DIR}/goldfish_snippet.html", "r") as f:
            st.components.v1.html(f.read(), height=520, scrolling=True)

    ## DOG: 
    st.title('Dog Image/EEG Reading Pair - Subject 0')
    dog_snippet_L, dog_snippet_R = st.columns([0.20,0.8])   
    with dog_snippet_L:
        st.subheader('Dog - n02099712')
        st.image(f'{ASSESTS_DIR}/ILSVRC2012_val_n02099712_00035360.JPEG')
        st.caption('Image shown to subject 0 to produce the EEG reading on the right')
    with dog_snippet_R:
        with open(f"{PLOTS_DIR}/dog_snippet.html", "r") as f:
            st.components.v1.html(f.read(), height=520, scrolling=True)


    with open(f"{PLOTS_DIR}/trial_timeseries.html", "r") as f:
        st.components.v1.html(f.read(), height=520, scrolling=True)
        st.caption('Multi-channel raw signal - Line plot of the EEG signals over time, one line per electrode/channel.')
    with open(f"{PLOTS_DIR}/channel_corr.html", "r") as f:
        st.components.v1.html(f.read(), height=520, scrolling=True)
        st.caption('Channel × channel correlation heatmap - A grid showing how similar each channel’s signal is to every other channel.')
    with open(f"{PLOTS_DIR}/spectrogram_ch1.html", "r") as f:
        st.components.v1.html(f.read(), height=520, scrolling=True)
        st.caption('Time-frequency spectrogram - A heatmap showing how the frequency content changes over time for one channel. Brighter areas mean stronger activity at that frequency and time.')
    with open(f"{PLOTS_DIR}/trial_heatmap.html", "r") as f:
        st.components.v1.html(f.read(), height=520, scrolling=True)
        st.caption('EEG trial heatmap (channels × time) - A 2D heatmap with channels on the y-axis and time on the x-axis. Color = signal strength')


## Results Tab: 
with results_tab:
    st.title('Model Performance Results')
    st.subheader('Model Statistics')
    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Subjects", int(summary["subject"].nunique()))
    with k2:
        st.metric("Models", int(summary["model"].nunique()))
    with k3:
        st.metric("Experiments", len(summary))


    # Accuracy by model - Box Plot
    st.subheader("Accuracy by Model")
    fig_box = px.box(summary, x="model", y="accuracy", points="all")
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption( "Distribution of per-subject classification accuracy by model. The box shows the median (line) and interquartile range; whiskers show the spread, and each dot is an individual observation (e.g., subject/run)")

    # Per-subject accuracy (grouped) - Bar Chart
    st.subheader("Per-Subject Accuracy by Model")
    pivot = summary.pivot_table(index="subject", columns="model", values="accuracy")
    fig_bar = px.bar(
        pivot.reset_index().melt(id_vars="subject", var_name="model", value_name="accuracy"),
        x="subject", y="accuracy", color="model", barmode="group"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Per-subject classification accuracy by model. Each group of bars is a subject; RF leads for both subjects while SVM trails.")

    st.subheader('Most Frequently Confused Classes')
    with open(f"{PLOTS_DIR}/top_confusions_results.html", "r") as f:
        st.components.v1.html(f.read(), height=520, scrolling=True)

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
        st.dataframe(dfp)
        st.markdown(f"**Subject {subject} — Model {model}**")

        # Confusion matrix
        if {"y", "yhat"}.issubset(dfp.columns):
            fig_cm = confmat_fig(dfp["y"].astype(str).tolist(),
                                dfp["yhat"].astype(str).tolist(),
                                title=f"Confusion Matrix — Subject {subject} · {model}")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning(f"`y`/`yhat` columns not found in {pf.name}")
        st.caption(
            f"Confusion matrix for Subject {subject} · {model}. Rows are the true classes (y) and columns are the model’s predictions (ŷ). "
            "Each cell shows how many (or what percent) of trials fell into that true/predicted pair—strong values on the diagonal mean correct classifications, "
            "while darker off-diagonals highlight specific class confusions to investigate."
            )


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
