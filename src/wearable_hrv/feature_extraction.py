"""Motion features: axis summaries plus dynamic magnitude spectral content."""

import numpy as np
from scipy.signal import periodogram
from .preprocessing import vector_magnitude


def extract_features(acc, fs: float) -> dict[str, float]:
    acc = np.asarray(acc, dtype=float)
    magnitude = vector_magnitude(acc)
    if len(acc) < 4 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Insufficient samples or invalid sampling frequency")
    features = {}
    for name, x in zip(("x", "y", "z", "magnitude"), (*acc.T, magnitude)):
        for metric, value in {
            "mean": np.mean(x),
            "std": np.std(x),
            "variance": np.var(x),
            "rms": np.sqrt(np.mean(x * x)),
            "range": np.ptp(x),
            "iqr": np.percentile(x, 75) - np.percentile(x, 25),
            "energy": np.mean(x * x),
        }.items():
            features[f"{name}_{metric}"] = float(value)
    # Spectrum of dynamic tri-axis magnitude; avoids a gravity-dominated DC peak.
    dynamic = np.linalg.norm(acc - acc.mean(axis=0), axis=1)
    frequencies, power = periodogram(dynamic, fs=fs, detrend="constant")
    selected = (frequencies >= 0.25) & (frequencies <= min(10.0, fs / 2))
    spectral = power[selected]
    features["dynamic_dominant_hz"] = (
        float(frequencies[selected][np.argmax(spectral)])
        if len(spectral) and spectral.max() > 1e-16
        else 0.0
    )
    features["dynamic_spectral_energy"] = (
        float(np.trapezoid(spectral, frequencies[selected]))
        if hasattr(np, "trapezoid")
        else float(np.trapz(spectral, frequencies[selected]))
    )
    return features
