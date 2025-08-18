#!/usr/bin/env python3
"""
04_evaluate.py
Aggregate results, compute metrics, and emit Plotly HTML figures.
Creates:
  - plots/acc_by_model.html
  - plots/acc_by_subject.html
  - plots/confmat_subX_modelY.html (per prediction file)
  - plots/topk_by_model.html (if top3/top5 available)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from utils_viz import box_acc_by_model, bars_subject_by_model, confmat_fig, fig_save_html


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--in", dest="results_dir", required=True, help="Directory with results (summary.csv + pred_*.parquet)")
    # ap.add_argument("--plots", dest="plots_dir", required=True, help="Directory to write HTML plots")
    # args = ap.parse_args()
    results_dir = Path('/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data/results/')#Path(args.in_dir)
    plots_dir = Path('/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data/plots/')#Path(args.out_dir)

    #results_dir, plots_dir = Path(args.results_dir), Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    if summary.empty:
        raise SystemExit("summary.csv is empty")

    # 1) Accuracy by model (box)
    fig = box_acc_by_model(summary)
    fig_save_html(fig, str(plots_dir / "acc_by_model.html"))

    # 2) Per-subject bars
    fig = bars_subject_by_model(summary)
    fig_save_html(fig, str(plots_dir / "acc_by_subject.html"))

    # 3) Confusion matrices
    pred_files = list(results_dir.glob("pred_sub*_*.parquet"))
    for pf in pred_files:
        dfp = pd.read_parquet(pf)
        model = pf.stem.split("_")[-1]
        sub = pf.stem.split("_")[1].replace("sub", "")
        fig = confmat_fig(dfp["y"].astype(str).tolist(), dfp["yhat"].astype(str).tolist(),
                          title=f"Confusion Matrix — Subject {sub} · {model}")
        fig_save_html(fig, str(plots_dir / f"confmat_sub{sub}_{model}.html"))

    # 4) Top-k overview (if present)
    if {"top3", "top5"}.issubset(set(summary.columns)):
        fig = summary.melt(id_vars=["subject", "model"], value_vars=["top3", "top5"],
                           var_name="metric", value_name="value")
        import plotly.express as px
        fig = px.box(fig, x="model", y="value", color="metric", points="all",
                     title="Top-k Accuracy by Model")
        fig_save_html(fig, str(plots_dir / "topk_by_model.html"))

    print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    main()
