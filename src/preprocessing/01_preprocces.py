#!/usr/bin/env python3
"""
01_preprocess.py
Load raw EEG bundles (.pth/.npz), apply light preprocessing, and save per-bundle
DataFrame with columns: ['subject','label','eeg'] where 'eeg' is a numpy array (channels x samples).
Saved as .pkl to support object dtype.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Optional deps (handled gracefully)
try:
    import torch
except Exception:
    torch = None

try:
    from mne.filter import filter_data
except Exception:
    filter_data = None

from utils_io import ensure_dir, save_df


def _load_pth(path: Path):
    if torch is None:
        raise RuntimeError("torch is required to read .pth. Please install PyTorch.")
    blob = torch.load(path, map_location="cpu",pickle_module=__import__("pickle"))
    # Expecting {'dataset': [ {'eeg_data': Tensor(C,S), 'label': int/str, 'subject': int}, ... ]}
    if isinstance(blob, dict) and "dataset" in blob:
        data = []
        for item in blob["dataset"]:
            eeg = item.get("eeg_data")
            if hasattr(eeg, "numpy"):
                eeg = eeg.numpy()
            label = item.get("label")
            subject = int(item.get("subject", -1))
            data.append((subject, str(label), np.asarray(eeg)))
        return data
    raise ValueError(f"Unrecognized .pth structure for {path}")


def _load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    # Allow formats: X (N,C,S), y (N,), subject (N,) or single subject id
    X = z["X"]
    y = z["y"]
    subj = z.get("subject", None)
    rows = []
    for i in range(len(X)):
        subject = int(subj[i]) if subj is not None and np.ndim(subj) else int(subj) if subj is not None else -1
        rows.append((subject, str(y[i]), X[i]))
    return rows


# def preprocess_eeg(eeg: np.ndarray, sfreq: float = 1000.0) -> np.ndarray:
#     """
#     eeg: (channels, samples)
#     Steps: average reference + bandpass 0.5-80 Hz (if mne available).
#     """
#     x = np.asarray(eeg, dtype=np.float64)
#     #x = eeg.astype(np.float32, copy=True)
#     x = np.ascontiguousarray(x)
#     # average reference
#     x = x - x.mean(axis=0, keepdims=True)
#     if filter_data is not None:
#         x = filter_data(x, sfreq=sfreq, l_freq=0.5, h_freq=80.0, verbose=False)
#     return x

def preprocess_eeg(eeg: np.ndarray, sfreq: float = 1000.0) -> np.ndarray:
    """
    eeg: (channels, samples)
    Steps: average reference + band-pass 0.5–80 Hz.
    - Uses MNE IIR on short segments to avoid FIR > signal length.
    - Falls back to SciPy if MNE isn't available.
    """
    x = np.asarray(eeg, dtype=np.float64, order="C")
    # average reference
    x -= x.mean(axis=0, keepdims=True)

    n_samp = x.shape[1]
    l_freq, h_freq = 0.5, 80.0

    # Prefer IIR for short signals (e.g., < 2000 samples)
    use_iir = True

    try:
        from mne.filter import filter_data as mne_filter_data
        if use_iir:
            # 4th-order zero-phase Butterworth (bidirectional)
            iir_params = dict(order=4, ftype="butter")
            x = mne_filter_data(
                x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                method="iir", iir_params=iir_params, phase="zero",
                pad="reflect_limited", verbose=False
            )
        else:
            # FIR path (kept for completeness; not used for short segments)
            x = mne_filter_data(
                x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                method="fir", phase="zero", fir_window="hamming",
                fir_design="firwin", filter_length="auto",
                pad="reflect_limited", verbose=False
            )
    except Exception:
        # SciPy fallback: zero-phase IIR
        from scipy.signal import iirfilter, filtfilt
        b, a = iirfilter(
            N=4, Wn=[l_freq/(sfreq/2), h_freq/(sfreq/2)],
            btype="band", ftype="butter"
        )
        for ch in range(x.shape[0]):
            # padlen=0 avoids edge issues on very short signals
            x[ch] = filtfilt(b, a, x[ch], padlen=0)

    return x



def main():
    """Use arg parse or update the home directory manually:"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Directory with .pth/.npz")
    ap.add_argument("--out", dest="out_dir", required=True, help="Directory for interim .pkl")
    #ap.add_argument("--sfreq", type=float, default=1000.0, help="Sampling rate (Hz)")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    #home_dir = '/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data'
    #in_dir = Path(f'{home_dir}/raw')
    #out_dir = Path(f'{home_dir}/data/interim/')
    
    sfreq = 1000.0
    ensure_dir(out_dir)

    files = list(in_dir.glob("*.pth")) + list(in_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .pth or .npz files found in {in_dir}")

    for src in files:
        print(f"Processing {src.name} ...")
        if src.suffix == ".pth":
            rows = _load_pth(src)
        else:
            rows = _load_npz(src)

        df_rows = []
        for subject, label, eeg in rows:
            eeg_p = preprocess_eeg(eeg, sfreq=sfreq)
            df_rows.append({"subject": subject, "label": label, "eeg": eeg_p})

        df = pd.DataFrame(df_rows)
        # Save as pickle to preserve ndarray objects
        save_path = out_dir / f"{src.stem}.pkl"
        save_df(df, save_path)
        print(f"Saved {save_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
