#!/usr/bin/env python3
"""
02_featureize.py
Convert preprocessed EEG (object arrays) into tabular features:
 - Per-channel bandpower for delta/theta/alpha/beta/gamma via Welch PSD
 - Differential entropy per band
Outputs one parquet per source file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.signal import welch
from utils_io import ensure_dir, load_df

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}


def diff_entropy(prob: np.ndarray) -> float:
    p = prob / (np.sum(prob) + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def row_to_features(eeg: np.ndarray, sfreq: float = 1000.0) -> Dict[str, float]:
    # eeg: (channels, samples)
    feats: Dict[str, float] = {}
    C, S = eeg.shape
    for ch in range(C):
        f, pxx = welch(eeg[ch], fs=sfreq, nperseg=min(256, S))
        for band, (lo, hi) in BANDS.items():
            mask = (f >= lo) & (f <= hi)
            bp = float(pxx[mask].sum())
            de = diff_entropy(pxx[mask])
            feats[f"ch{ch}_{band}_bp"] = bp
            feats[f"ch{ch}_{band}_de"] = de
    # Global summaries across channels (help small models)
    for band in BANDS:
        ch_cols = [f"ch{ch}_{band}_bp" for ch in range(C)]
        vals = np.array([feats[c] for c in ch_cols])
        feats[f"all_{band}_bp_mean"] = float(vals.mean())
        feats[f"all_{band}_bp_std"]  = float(vals.std(ddof=0))
    return feats


def main():
    """Use arg parse or update the home directory manually:"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Interim .pkl files directory")
    ap.add_argument("--out", dest="out_dir", required=True, help="Processed feature tables directory")
    ap.add_argument("--sfreq", type=float, default=1000.0)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    #home_dir = '/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data'
    #in_dir = Path(f'{home_dir}/interim')
    #out_dir = Path(f'{home_dir}/data/processed/')
    sfreq = 1000.0
    ensure_dir(out_dir)

    files = sorted(list(in_dir.glob("*.pkl")) + list(in_dir.glob("*.pickle")) + list(in_dir.glob("*.parquet")))
    if not files:
        raise SystemExit(f"No interim files found in {in_dir}")

    for f in files:
        print(f"Featurizing {f.name} ...")
        df = load_df(f)

        rows = []
        for _, r in df.iterrows():
            feats = row_to_features(r.eeg, sfreq=sfreq)
            feats.update({"subject": int(r.subject), "label": str(r.label)})
            rows.append(feats)

        out = pd.DataFrame(rows)
        out_path = out_dir / f"{f.stem}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"Saved {out_path}")

    print("Featureization complete.")


if __name__ == "__main__":
    main()
