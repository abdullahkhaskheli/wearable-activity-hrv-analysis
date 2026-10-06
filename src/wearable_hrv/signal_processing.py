"""Inspectable baseline peak detectors; not clinically validated algorithms."""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


def detect_peaks(
    filtered,
    fs: float,
    refractory_s: float = 0.3,
    prominence_mad: float = 3.0,
    ecg: bool = True,
):
    x = np.asarray(filtered, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 3
        or not np.isfinite(x).all()
        or fs <= 0
        or refractory_s <= 0
        or prominence_mad <= 0
    ):
        raise ValueError("Invalid peak detection input")
    centered = x - np.median(x)
    if np.max(np.abs(centered)) < np.finfo(float).eps:
        return np.array([], dtype=int)
    # ECG supports either polarity; PPG uses positive pulse maxima.
    y = np.abs(centered) if ecg else centered
    mad = np.median(np.abs(centered))
    prominence = max(
        prominence_mad * mad,
        0.05 * np.percentile(np.abs(centered), 99),
        np.finfo(float).eps,
    )
    peaks, _ = find_peaks(
        y, distance=max(1, int(round(fs * refractory_s))), prominence=prominence
    )
    return peaks


def rr_intervals(peaks, fs: float):
    peaks = np.asarray(peaks, dtype=float)
    if (
        peaks.ndim != 1
        or not np.isfinite(peaks).all()
        or np.any(np.diff(peaks) <= 0)
        or np.any(peaks < 0)
        or fs <= 0
    ):
        raise ValueError(
            "Peaks must be strictly increasing nonnegative positions; fs > 0"
        )
    return np.diff(peaks) / fs * 1000


def valid_rr_mask(
    rr, min_ms: float = 300, max_ms: float = 2000, local_deviation: float = 0.2
):
    rr = np.asarray(rr, dtype=float)
    if rr.ndim != 1 or not 0 < min_ms < max_ms or not 0 < local_deviation < 1:
        raise ValueError("Invalid RR input or limits")
    mask = np.isfinite(rr) & (rr >= min_ms) & (rr <= max_ms)
    if len(rr) >= 5 and mask.any():
        safe = np.where(mask, rr, np.median(rr[mask]))
        local = median_filter(safe, size=5, mode="nearest")
        mask &= np.abs(rr - local) <= local_deviation * local
    return mask


def heart_rate(rr, **limits) -> tuple[float, float]:
    rr = np.asarray(rr, dtype=float)
    valid = valid_rr_mask(rr, **limits)
    return (
        float(np.mean(60000 / rr[valid])) if valid.any() else float("nan"),
        float(valid.mean()) if len(valid) else 0.0,
    )


def peak_window_intervals(peaks, fs: float, start: float, end: float):
    """Only intervals with both endpoints inside the same half-open window."""
    p = np.asarray(peaks)
    inside = p[(p / fs >= start) & (p / fs < end)]
    return rr_intervals(inside, fs)


def ppg_quality(rr, acc, **limits) -> dict:
    """Diagnostics, not motion correction: retain movement and interval coverage."""
    from .preprocessing import movement_intensity

    hr, fraction = heart_rate(rr, **limits)
    return {
        "ppg_hr": hr,
        "ppg_valid_fraction": fraction,
        "movement": movement_intensity(acc),
    }
