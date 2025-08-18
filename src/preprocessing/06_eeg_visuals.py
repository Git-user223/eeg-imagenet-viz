
#!/usr/bin/env python3
"""
06_eeg_visuals.py
Load the preprocessed EEG (.pkl from 01_preprocces.py) and optional features (.parquet from 02_featureize.py),
then write a set of dashboard-friendly Plotly HTML plots.

Examples:
  python 06_eeg_visuals.py --pkl data/preprocessed_subject1.pkl --out plots --fs 250 --trial-index 0
  python 06_eeg_visuals.py --pkl data/preprocessed_subject1.pkl --features features/subject1.parquet --out plots --fs 250
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# SciPy for PSD & spectrogram (optional but recommended)
try:
    from scipy.signal import welch, spectrogram
except Exception:
    welch = None
    spectrogram = None


# --------------------------
# Utilities
# --------------------------

def fig_save_html(fig, path: str) -> None:
    """Save Plotly figure to a standalone HTML file (Plotly JS via CDN)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def _ensure_2d(eeg: np.ndarray) -> np.ndarray:
    """Ensure eeg is 2D (channels x samples)."""
    eeg = np.asarray(eeg)
    if eeg.ndim == 1:
        return eeg[np.newaxis, :]
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x samples), got shape {eeg.shape}")
    return eeg


def _downsample(y: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return y
    return y[..., ::factor]


def _default_channel_names(n_ch: int) -> List[str]:
    return [f"Ch{ix+1}" for ix in range(n_ch)]


# --------------------------
# Plots
# --------------------------

def plot_trial_timeseries(
    eeg: np.ndarray,
    fs: float,
    channels: Optional[List[int]] = None,
    channel_names: Optional[List[str]] = None,
    downsample: int = 1,
    title: str = "EEG Trial: Raw Time Series"
) -> go.Figure:
    """Multi-channel time series plot (one trace per channel)."""
    eeg = _ensure_2d(eeg)
    C, T = eeg.shape
    if channels is None:
        channels = list(range(C))
    if channel_names is None:
        channel_names = _default_channel_names(C)

    y = _downsample(eeg[channels, :], downsample)
    x = np.arange(y.shape[1]) / fs * downsample

    fig = go.Figure()
    for i, ch in enumerate(channels):
        fig.add_trace(go.Scatter(x=x, y=y[i, :], mode="lines", name=channel_names[ch]))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        legend_title="Channels",
        template="plotly_white",
        height=500
    )
    return fig


def plot_channel_correlation(eeg: np.ndarray, channel_names: Optional[List[str]] = None,
                             title: str = "Channel Correlation") -> go.Figure:
    eeg = _ensure_2d(eeg)
    C, _ = eeg.shape
    if channel_names is None:
        channel_names = _default_channel_names(C)
    corr = np.corrcoef(eeg)
    fig = px.imshow(corr, x=channel_names, y=channel_names, origin="lower", aspect="auto",
                    labels=dict(color="corr"), title=title)
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plot_psd_welch(eeg: np.ndarray, fs: float,
                   nperseg: int = 256, noverlap: int = 128,
                   title: str = "Power Spectral Density (Welch)") -> go.Figure:
    if welch is None:
        raise ImportError("scipy is required for PSD. Please install scipy.")
    eeg = _ensure_2d(eeg)
    C, _ = eeg.shape
    fig = go.Figure()
    psd_all = []
    f = None

    for ch in range(C):
        f, Pxx = welch(eeg[ch, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
        psd_all.append(Pxx)
        fig.add_trace(go.Scatter(x=f, y=10*np.log10(Pxx + 1e-12), mode="lines", name=f"Ch{ch+1}"))

    psd_all = np.vstack(psd_all)
    psd_mean = psd_all.mean(axis=0)
    fig.add_trace(go.Scatter(x=f, y=10*np.log10(psd_mean + 1e-12), mode="lines",
                             name="Mean", line=dict(dash="dash")))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        template="plotly_white",
        height=500
    )
    return fig


def plot_spectrogram(eeg: np.ndarray, fs: float, channel: int = 0,
                     nperseg: int = 256, noverlap: int = 128,
                     title: Optional[str] = None) -> go.Figure:
    if spectrogram is None:
        raise ImportError("scipy is required for spectrogram. Please install scipy.")
    eeg = _ensure_2d(eeg)
    f, t, Sxx = spectrogram(eeg[channel, :], fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    if title is None:
        title = f"Spectrogram (Ch{channel+1})"
    fig = go.Figure(data=go.Heatmap(z=10*np.log10(Sxx + 1e-12), x=t, y=f, coloraxis="coloraxis"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        coloraxis_colorbar=dict(title="Power (dB)"),
        template="plotly_white",
        height=500
    )
    return fig


def plot_trial_heatmap(eeg: np.ndarray, channel_names: Optional[List[str]] = None,
                       title: str = "EEG Trial Heatmap (channels x time)") -> go.Figure:
    eeg = _ensure_2d(eeg)
    C, T = eeg.shape
    if channel_names is None:
        channel_names = _default_channel_names(C)
    fig = px.imshow(eeg, aspect="auto", origin="lower",
                    labels=dict(x="Time (samples)", y="Channel", color="µV"),
                    x=np.arange(T), y=channel_names, title=title)
    fig.update_layout(template="plotly_white", height=500)
    return fig


def compute_erp_gfp(df: pd.DataFrame, fs: float, label: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a DataFrame with 'eeg' arrays (C x T), compute ERP (mean across trials per time)
    and GFP (std across channels of ERP). If label is provided and present, filter to it.
    Returns (erp_mean (T,), gfp (T,)).
    """
    if label is not None and "label" in df.columns:
        df = df[df["label"] == label]
    stacks = []
    for eeg in df["eeg"]:
        eeg2 = _ensure_2d(np.asarray(eeg))
        stacks.append(eeg2)
    cube = np.stack(stacks, axis=0)  # (N, C, T)
    erp = cube.mean(axis=0)          # (C, T)
    erp_mean = erp.mean(axis=0)      # (T,)
    gfp = erp.std(axis=0)            # (T,)
    return erp_mean, gfp


def plot_erp_gfp(erp: np.ndarray, gfp: np.ndarray, fs: float,
                 title: str = "ERP & Global Field Power") -> go.Figure:
    T = erp.shape[0]
    x = np.arange(T) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=erp, name="ERP (mean across channels)", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=gfp, name="GFP (stdev across channels)", mode="lines", yaxis="y2"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        yaxis2=dict(title="GFP (µV)", overlaying="y", side="right"),
        template="plotly_white",
        height=500
    )
    return fig


def plot_bandpower_bars(features_parquet: str,
                        groupby_cols: Optional[List[str]] = None,
                        bands: Optional[List[str]] = None,
                        agg: str = "mean",
                        title: str = "Bandpower by Label / Channel") -> go.Figure:
    """
    Expects a parquet created by 02_featureize.py with columns like:
      ['subject','label','channel','delta','theta','alpha','beta','gamma', 'de_delta', ...]
    groupby_cols controls how bars are grouped (e.g., ['label'] or ['label','channel'])
    bands controls which bands to plot. Default: all present bands from the set.
    """
    df = pd.read_parquet(features_parquet)
    default_bands = [c for c in ["delta","theta","alpha","beta","gamma"] if c in df.columns]
    if bands is None:
        bands = default_bands
    if groupby_cols is None or len(groupby_cols) == 0:
        groupby_cols = ["label"] if "label" in df.columns else ["subject"]

    grouped = df.groupby(groupby_cols)[bands].agg(agg).reset_index()
    long = grouped.melt(id_vars=groupby_cols, value_vars=bands, var_name="band", value_name="power")

    if len(groupby_cols) > 1:
        long["x"] = long[groupby_cols].astype(str).agg(" / ".join, axis=1)
        xcol = "x"
        xlab = "/".join(groupby_cols)
    else:
        xcol = groupby_cols[0]
        xlab = xcol

    fig = px.bar(long, x=xcol, y="power", color="band",
                 barmode="group" if len(bands) <= 3 else "relative",
                 title=title)
    fig.update_layout(template="plotly_white", height=500, xaxis_title=xlab)
    return fig


# --------------------------
# I/O
# --------------------------

def load_preprocessed_df(pkl_path: str) -> pd.DataFrame:
    """Load .pkl created by 01_preprocces.py (columns: subject,label,eeg)."""
    return pd.read_pickle(pkl_path)


def write_all_html(
    pkl_path: str,
    out_dir: str,
    fs: float,
    trial_index: int = 0,
    spectro_channel: int = 0,
    features_parquet: Optional[str] = None,
    downsample: int = 2
) -> Dict[str, str]:
    """
    Produce a set of HTML plots for quick inspection. Returns a dict of {name: path}.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_preprocessed_df(pkl_path)
    if len(df) == 0:
        raise ValueError("Empty dataframe. Check your --pkl input.")
    eeg = _ensure_2d(np.asarray(df.iloc[trial_index]["eeg"]))

    paths = {}

    # 1) Time series
    fig_ts = plot_trial_timeseries(eeg, fs, downsample=downsample, title="EEG Trial Time Series")
    path_ts = str(out / "trial_timeseries.html"); fig_save_html(fig_ts, path_ts); paths["timeseries"] = path_ts

    # 2) Correlation
    fig_corr = plot_channel_correlation(eeg, title="Channel Correlation")
    path_corr = str(out / "channel_corr.html"); fig_save_html(fig_corr, path_corr); paths["corr"] = path_corr

    # 3) PSD
    if welch is not None:
        fig_psd = plot_psd_welch(eeg, fs, title="PSD (Welch)")
        path_psd = str(out / "psd_welch.html"); fig_save_html(fig_psd, path_psd); paths["psd"] = path_psd

    # 4) Spectrogram
    if spectrogram is not None:
        fig_sp = plot_spectrogram(eeg, fs, channel=spectro_channel)
        path_sp = str(out / f"spectrogram_ch{spectro_channel+1}.html"); fig_save_html(fig_sp, path_sp); paths["spectrogram"] = path_sp

    # 5) Heatmap
    fig_hm = plot_trial_heatmap(eeg)
    path_hm = str(out / "trial_heatmap.html"); fig_save_html(fig_hm, path_hm); paths["heatmap"] = path_hm

    # 6) ERP/GFP (if multiple trials available)
    if len(df) >= 3:
        # If labels exist, keep the first label for a consistent ERP set
        label = None
        if "label" in df.columns:
            label = df.iloc[0]["label"]
        erp, gfp = compute_erp_gfp(df, fs=fs, label=label)
        fig_eg = plot_erp_gfp(erp, gfp, fs)
        path_eg = str(out / "erp_gfp.html"); fig_save_html(fig_eg, path_eg); paths["erp_gfp"] = path_eg

    # 7) Bandpowers (if provided)
    if features_parquet is not None and Path(features_parquet).exists():
        fig_bp = plot_bandpower_bars(features_parquet, groupby_cols=["label"] if "label" in df.columns else ["subject"])
        path_bp = str(out / "bandpowers.html"); fig_save_html(fig_bp, path_bp); paths["bandpowers"] = path_bp

    return paths


# --------------------------
# CLI
# --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Write Plotly EEG visuals as HTML files.")
    p.add_argument("--pkl", required=True, help="Path to preprocessed .pkl (from 01_preprocces.py).")
    p.add_argument("--features", default=None, help="Optional path to features .parquet (from 02_featureize.py).")
    p.add_argument("--out", default="plots", help="Output directory for HTML files.")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz).")
    p.add_argument("--trial-index", type=int, default=0, help="Which trial/row to visualize for single-trial plots.")
    p.add_argument("--spectro-channel", type=int, default=0, help="Channel index for spectrogram.")
    p.add_argument("--downsample", type=int, default=2, help="Decimate time series by keeping every k-th sample.")
    return p.parse_args()


def main():
    """Use arg parse or update the home directory manually:"""
    # args = parse_args()
    # paths = write_all_html(
    #     pkl_path=args.pkl,
    #     features_parquet=args.features,
    #     out_dir=args.out,
    #     fs=args.fs,
    #     trial_index=args.trial_index,
    #     spectro_channel=args.spectro_channel,
    #     downsample=args.downsample
    # )
    home_dir = '/Users/charithwickrema/Desktop/Data Viz/Final/EEG-ImageNet-Dataset/data'
    paths = write_all_html(
        pkl_path=f'{home_dir}/interim/EEG-ImageNet_1.pkl',
        features_parquet=f'{home_dir}/processed/EEG-ImageNet_1.parquet',
        out_dir=f'{home_dir}/plots',
        fs=1000.0,
        trial_index=0,
        spectro_channel=0,
        downsample=2
    )
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k:12s} -> {v}")


if __name__ == "__main__":
    main()
