"""Paired finite HR agreement; descriptive repeated-window estimates."""

import numpy as np
import pandas as pd


def agreement(reference, estimate) -> dict:
    ref, est = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    if ref.ndim != 1 or est.shape != ref.shape:
        raise ValueError("Paired one-dimensional arrays of equal shape required")
    keep = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[keep], est[keep]
    if not len(ref):
        raise ValueError("No finite paired measurements")
    error = est - ref
    sd = float(np.std(error, ddof=1)) if len(error) > 1 else float("nan")
    corr = (
        float(np.corrcoef(ref, est)[0, 1])
        if len(ref) > 1 and np.std(ref) > 0 and np.std(est) > 0
        else float("nan")
    )
    bias = float(error.mean())
    return {
        "n_pairs": len(ref),
        "mae_bpm": float(np.abs(error).mean()),
        "rmse_bpm": float(np.sqrt(np.mean(error**2))),
        "bias_bpm": bias,
        "pearson_r": corr,
        "difference_sd_bpm": sd,
        "loa_low_bpm": bias - 1.96 * sd,
        "loa_high_bpm": bias + 1.96 * sd,
    }


def agreement_table(pairs: pd.DataFrame, group_columns=None):
    groups = (
        [((), pairs)]
        if not group_columns
        else pairs.groupby(group_columns, dropna=False)
    )
    rows = []
    for key, frame in groups:
        key = key if isinstance(key, tuple) else (key,)
        try:
            metrics = agreement(frame.ecg_hr, frame.ppg_hr)
        except ValueError:
            continue
        rows.append({**dict(zip(group_columns or [], key)), **metrics})
    return pd.DataFrame(rows)


def bland_altman_points(reference, estimate):
    ref, est = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    if ref.shape != est.shape or ref.ndim != 1:
        raise ValueError("Paired vectors required")
    mask = np.isfinite(ref) & np.isfinite(est)
    return (ref[mask] + est[mask]) / 2, est[mask] - ref[mask]
