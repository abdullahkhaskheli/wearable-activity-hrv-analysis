"""Participant-aware descriptive summaries; no window-level significance tests."""

import numpy as np
import pandas as pd


def participant_summary(frame, metrics):
    if frame.empty:
        return pd.DataFrame(columns=["subject_id", "activity", *metrics])
    return frame.groupby(["subject_id", "activity"])[metrics].mean().reset_index()


def activity_summary(participants, metrics):
    if participants.empty:
        return pd.DataFrame()
    result = (
        participants.groupby("activity")[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result.columns = ["_".join(filter(None, c)) for c in result.columns]
    return result


def movement_error_association(pairs):
    rows = []
    for subject, frame in pairs.groupby("subject_id"):
        x = frame.movement.to_numpy(float)
        y = np.abs(frame.ppg_hr.to_numpy(float) - frame.ecg_hr.to_numpy(float))
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        correlation = (
            float(np.corrcoef(x, y)[0, 1])
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0
            else float("nan")
        )
        rows.append(
            {
                "subject_id": subject,
                "n_pairs": len(x),
                "movement_error_pearson_r": correlation,
            }
        )
    return pd.DataFrame(rows)
