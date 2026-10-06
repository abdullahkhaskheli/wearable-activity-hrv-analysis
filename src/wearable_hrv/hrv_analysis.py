"""Time-domain RR metrics with preserved adjacency through rejected beats.

Automated screening does not certify true sinus NN intervals. Outputs are
screened-RR estimates and must retain this qualification.
"""

import numpy as np
from .signal_processing import valid_rr_mask


def hrv_metrics(rr, valid=None) -> dict[str, float]:
    rr = np.asarray(rr, dtype=float)
    if rr.ndim != 1:
        raise ValueError("RR must be one-dimensional")
    valid = (
        np.isfinite(rr) & (rr > 0) if valid is None else np.asarray(valid, dtype=bool)
    )
    if valid.shape != rr.shape or np.any(valid & (~np.isfinite(rr) | (rr <= 0))):
        raise ValueError("Invalid interval mask")
    nn = rr[valid]
    # Do not bridge a missing/rejected interval for successive-difference metrics.
    differences = np.diff(rr)[valid[:-1] & valid[1:]]
    if len(nn) < 3 or len(differences) < 2:
        raise ValueError("Insufficient valid adjacent intervals for HRV")
    sdnn = np.std(nn, ddof=1)
    rmssd = np.sqrt(np.mean(differences**2))
    sd1 = np.std(differences, ddof=1) / np.sqrt(2)
    sd2 = np.sqrt(max(0.0, 2 * sdnn**2 - sd1**2))
    return {
        "mean_nn_ms": float(nn.mean()),
        "sdnn_ms": float(sdnn),
        "rmssd_ms": float(rmssd),
        "pnn50_percent": float(100 * np.mean(np.abs(differences) > 50)),
        "sd1_ms": float(sd1),
        "sd2_ms": float(sd2),
        "n_intervals": int(len(nn)),
        "n_adjacent_pairs": int(len(differences)),
    }


def screened_hrv(
    rr,
    min_intervals=30,
    min_valid_fraction=0.95,
    max_removed_fraction=0.05,
    max_half_mean_change=0.1,
    **limits,
):
    rr = np.asarray(rr, dtype=float)
    valid = valid_rr_mask(rr, **limits)
    fraction = float(valid.mean()) if len(valid) else 0.0
    status = {"valid_fraction": fraction, "accepted": False, "reason": ""}
    if (
        valid.sum() < min_intervals
        or fraction < min_valid_fraction
        or 1 - fraction > max_removed_fraction + 1e-12
    ):
        status["reason"] = "insufficient intervals or excessive rejected beats"
        return status
    mid = len(rr) // 2
    a, b = rr[:mid][valid[:mid]], rr[mid:][valid[mid:]]
    if (
        not len(a)
        or not len(b)
        or abs(a.mean() - b.mean()) / rr[valid].mean() > max_half_mean_change
    ):
        status["reason"] = (
            "half-segment RR means differ beyond configured stability screen"
        )
        return status
    try:
        metrics = hrv_metrics(rr, valid)
    except ValueError as exc:
        status["reason"] = str(exc)
        return status
    return {
        **status,
        **metrics,
        "accepted": True,
        "reason": "screen passed; sinus rhythm not verified",
    }
