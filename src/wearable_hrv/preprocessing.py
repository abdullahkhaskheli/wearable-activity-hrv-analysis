"""Zero-phase filtering, explicit boundaries and strict finite-data checks."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(signal, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Signal must be a finite one-dimensional array")
    if (
        not np.isfinite(fs)
        or not 0 < low < high < fs / 2
        or not isinstance(order, int)
        or order < 1
    ):
        raise ValueError("Require 0 < low < high < Nyquist and positive integer order")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if len(x) <= 3 * (2 * len(sos) + 1):
        raise ValueError("Signal too short for zero-phase filtering")
    return sosfiltfilt(sos, x)


def windows(duration_s: float, size_s: float, step_s: float):
    if (
        not all(np.isfinite(v) and v > 0 for v in (duration_s, size_s, step_s))
        or step_s > size_s
    ):
        raise ValueError("Invalid window parameters")
    for i in range(max(0, int(np.floor((duration_s - size_s) / step_s + 1e-9)) + 1)):
        start = i * step_s
        yield start, start + size_s


def slice_time(signal, fs: float, start: float, end: float):
    if fs <= 0 or start < 0 or end <= start or end * fs > len(signal) + 1e-6:
        raise ValueError("Time slice outside signal bounds")
    return signal[int(round(start * fs)) : int(round(end * fs))]


def activity_for_window(
    activity, fs: float, start: float, end: float, purity: float = 1.0
):
    labels = slice_time(activity, fs, start, end)
    ids, counts = np.unique(labels, return_counts=True)
    best = int(np.argmax(counts))
    label = int(ids[best])
    return label if label != 0 and counts[best] / len(labels) >= purity else None


def vector_magnitude(acc) -> np.ndarray:
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3 or not len(acc) or not np.isfinite(acc).all():
        raise ValueError("Accelerometer must be finite N x 3")
    return np.linalg.norm(acc, axis=1)


def movement_intensity(acc) -> float:
    """RMS of per-axis demeaned acceleration, in input sensor units."""
    vector_magnitude(acc)
    x = np.asarray(acc, dtype=float)
    return float(np.sqrt(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1))))
