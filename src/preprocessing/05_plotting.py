#!/usr/bin/env python3
"""
05_plotting.py
Optional: utilities to generate additional static artifacts beyond 04_evaluate.py.
Usage example:
  python src/05_plotting.py --results results --out plots_extra
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
from utils_viz import fig_save_html


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--results", required=True)
    # ap.add_argument("--out", required=True)
    # args = ap.parse_args()

    # res = Path(args.results)
    # out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = Path('/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data/results/')#Path(args.in_dir)
    out = Path('/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data/plots/')#Path(args.out_dir)

    summary = pd.read_csv(res / "summary.csv")
    # Learning curve-ish view: accuracy vs. subject index per model (not true learning curve but nice trend line)
    # Sort subjects numerically for a clean line
    summary["subject"] = summary["subject"].astype(int)
    summary = summary.sort_values(["model", "subject"])

    fig = px.line(summary, x="subject", y="accuracy", color="model", markers=True,
                  title="Accuracy Trend Across Subjects (by Model)")
    fig_save_html(fig, str(out / "acc_trend.html"))

    print(f"Wrote extra plots to {out}")


if __name__ == "__main__":
    main()
