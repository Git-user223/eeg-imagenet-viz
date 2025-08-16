#!/usr/bin/env python3
"""
utils_viz.py
Plotly helpers: confusion matrices, model/subject charts, etc.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio


def fig_save_html(fig, path: str) -> None:
    pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True)


def confmat_fig(y_true: List[str], y_pred: List[str], title: str = ""):
    labels = sorted(pd.unique(pd.Series(list(y_true) + list(y_pred)).astype(str)))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    annot = np.round(cm, 2).astype(str)
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels, annotation_text=annot, colorscale="Viridis", showscale=True
    )
    fig.update_layout(title=title or "Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig


def box_acc_by_model(summary_df: pd.DataFrame):
    fig = px.box(summary_df, x="model", y="accuracy", points="all", title="Accuracy by Model (All Subjects)")
    fig.update_traces(jitter=0.3, marker_opacity=0.6)
    return fig


def bars_subject_by_model(summary_df: pd.DataFrame):
    pivot = summary_df.pivot_table(index="subject", columns="model", values="accuracy")
    fig = px.bar(
        pivot.reset_index().melt(id_vars="subject", var_name="model", value_name="accuracy"),
        x="subject", y="accuracy", color="model", barmode="group",
        title="Per-Subject Accuracy by Model"
    )
    return fig


def lollipop_counts(class_counts: pd.DataFrame, title: str = "Class Frequency (Counts)"):
    # class_counts: columns = ['label', 'count', 'split']
    fig = px.scatter(
        class_counts, x="count", y="label", color="split", title=title, size="count",
        render_mode="svg"
    )
    return fig
