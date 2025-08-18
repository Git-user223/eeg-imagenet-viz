#!/usr/bin/env python3
"""
03_train.py
Train lightweight classifiers per subject (recommended) and save predictions & summary.
Models: lr, svm, rf, mlp
Outputs:
  - results/pred_sub<subject>_<model>.parquet (y, yhat, top3/top5)
  - results/summary.csv (subject, model, accuracy, top3, top5, n_test)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def make_models(selected: List[str]):
    base = {
        "lr":  Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=300, multi_class="multinomial"))]),
        "svm": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="linear", probability=True))]),
        "rf":  RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42),
        "mlp": Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(256,), max_iter=80, random_state=42))]),
    }
    if selected:
        base = {k: v for k, v in base.items() if k in selected}
    return base


def per_subject_split(df: pd.DataFrame, train_n: int = 30, test_n: int = 20):
    """
    For each class within a subject, take first 'train_n' as train and next 'test_n' as test.
    If not enough samples, fall back to 60/40 split.
    """
    tr_idx, te_idx = [], []
    for cls, g in df.groupby("label", sort=False):
        g = g.sort_index()  # preserve temporal order if present
        if len(g) >= (train_n + test_n):
            tr_idx.extend(g.index[:train_n])
            te_idx.extend(g.index[train_n:train_n + test_n])
        else:
            cut = max(1, int(0.6 * len(g)))
            tr_idx.extend(g.index[:cut])
            te_idx.extend(g.index[cut:])
    return df.loc[tr_idx], df.loc[te_idx]


def topk_from_proba(classes: List[str], proba: np.ndarray, k: int) -> List[str]:
    idx = np.argsort(-proba)[:k]
    return [classes[i] for i in idx]


def run_subject(subject_df: pd.DataFrame, models: Dict[str, object], save_dir: Path) -> List[dict]:
    # Build X/Y
    y = subject_df["label"].astype(str)
    X = subject_df.drop(columns=["subject", "label"])

    # Split
    tr_df, te_df = per_subject_split(subject_df)
    Xtr, ytr = tr_df.drop(columns=["subject", "label"]), tr_df["label"].astype(str)
    Xte, yte = te_df.drop(columns=["subject", "label"]), te_df["label"].astype(str)
    classes_sorted = sorted(ytr.unique().tolist())

    metrics_rows: List[dict] = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        acc = accuracy_score(yte, yhat)

        # Top-k metrics
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)  # (N, C)
        else:
            # Shouldn't happen for our choices, but guard.
            proba = None

        top3 = top5 = None
        top3_hits = top5_hits = 0
        top3_list, top5_list = [], []

        if proba is not None:
            class_order = model.classes_.tolist()
            for i in range(len(yte)):
                p = proba[i]
                t3 = topk_from_proba(class_order, p, 3)
                t5 = topk_from_proba(class_order, p, 5)
                top3_list.append(t3)
                top5_list.append(t5)
                if str(yte.iloc[i]) in t3:
                    top3_hits += 1
                if str(yte.iloc[i]) in t5:
                    top5_hits += 1
            top3 = top3_hits / len(yte) if len(yte) else None
            top5 = top5_hits / len(yte) if len(yte) else None

        # Save predictions
        pred_df = pd.DataFrame({
            "y": yte.values,
            "yhat": yhat,
        })
        if top3_list:
            pred_df["top3"] = [",".join(t) for t in top3_list]
        if top5_list:
            pred_df["top5"] = [",".join(t) for t in top5_list]

        pred_path = save_dir / f"pred_sub{int(subject_df['subject'].mode().iat[0])}_{name}.parquet"
        pred_df.to_parquet(pred_path, index=False)

        # Save model for later analysis (optional)
        joblib.dump(model, save_dir / f"model_sub{int(subject_df['subject'].mode().iat[0])}_{name}.joblib")

        metrics_rows.append({
            "subject": int(subject_df["subject"].mode().iat[0]),
            "model": name,
            "accuracy": acc,
            "top3": top3,
            "top5": top5,
            "n_test": len(yte),
        })
    return metrics_rows


def main():
    """Use arg parse or update the home directory manually:"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Processed feature tables (.parquet)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Results directory")
    ap.add_argument("--models", nargs="+", default=["lr", "svm", "rf", "mlp"])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    model_list = args.models
    #home_dir = '/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data'
    # in_dir = Path(f'{home_dir}/processed')
    # out_dir = Path(f'{home_dir}/data/results/')
    # model_list = ["lr", "svm", "rf", "mlp"]
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob("*.parquet")))
    if not files:
        raise SystemExit(f"No feature files found in {in_dir}")

    models = make_models(model_list)

    all_metrics: List[dict] = []
    for f in tqdm(files, desc="Subjects"):
        df = pd.read_parquet(f).reset_index(drop=True)
        if "subject" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{f.name} missing 'subject' or 'label' columns")
        # train per subject in file
        sub_id = int(df["subject"].mode().iat[0])
        print(f"Training subject {sub_id} from {f.name}")
        metrics = run_subject(df, models, out_dir)
        all_metrics.extend(metrics)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved summary to {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
