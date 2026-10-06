"""Scientific plots from caller-supplied computed data; no fallback fixtures."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .reliability import agreement, bland_altman_points
from .data_loader import ACTIVITIES


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def signal_comparison(
    raw, filtered, fs, peaks=None, title="Signal", unit="sensor units (uncalibrated)"
):
    if len(raw) == 0 or len(raw) != len(filtered):
        raise ValueError("Nonempty aligned signals required")
    t = np.arange(len(raw)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, raw, lw=0.7, label="Raw")
    axes[0].legend()
    axes[1].plot(t, filtered, lw=0.7, label="Filtered")
    if peaks is not None and len(peaks):
        axes[1].scatter(
            np.asarray(peaks) / fs,
            np.asarray(filtered)[peaks],
            s=12,
            label="Detected peaks",
        )
    axes[1].legend()
    axes[1].set_xlabel("Time from excerpt start (s)")
    for ax in axes:
        ax.set_ylabel(unit)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def activity_distribution(frame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([ACTIVITIES[int(v)] for v in frame.activity], frame.duration_s)
    ax.set_ylabel("Annotated duration (s)")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Activity coverage in processed subjects")
    return fig


def by_activity(frame, metric, ylabel, title):
    if frame.empty:
        raise ValueError("No observations to plot")
    labels = sorted(frame.activity.unique())
    values = [
        frame.loc[frame.activity == a, metric].dropna().to_numpy() for a in labels
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(values, tick_labels=[ACTIVITIES[int(a)] for a in labels])
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def confusion(frame, model):
    selected = frame[frame.model == model]
    matrix = (
        selected.groupby(["true_activity", "pred_activity"])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax, label="Held-out window count")
    ax.set_xticks(
        range(len(matrix.columns)),
        [ACTIVITIES[int(a)] for a in matrix.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(matrix)), [ACTIVITIES[int(a)] for a in matrix.index])
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("Actual activity")
    ax.set_title(model + " — GroupKFold")
    return fig


def model_comparison(summary):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary.model,
        summary.macro_f1_mean,
        yerr=summary.macro_f1_std.fillna(0),
        capsize=5,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (fold mean ± SD)")
    ax.set_title("Subject-independent evaluation")
    return fig


def agreement_plot(pairs):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pairs.ecg_hr, pairs.ppg_hr, s=5, alpha=0.25)
    lo = min(pairs.ecg_hr.min(), pairs.ppg_hr.min())
    hi = max(pairs.ecg_hr.max(), pairs.ppg_hr.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="black")
    ax.set_xlabel("ECG reference HR (bpm)")
    ax.set_ylabel("PPG HR (bpm)")
    ax.set_title("Paired heart-rate estimates")
    return fig


def bland_altman(pairs):
    mean, diff = bland_altman_points(pairs.ecg_hr, pairs.ppg_hr)
    metrics = agreement(pairs.ecg_hr, pairs.ppg_hr)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(mean, diff, s=5, alpha=0.25)
    for key in ("bias_bpm", "loa_low_bpm", "loa_high_bpm"):
        ax.axhline(metrics[key], ls="--", label=f"{key}: {metrics[key]:.2f}")
    ax.legend()
    ax.set_xlabel("Pair mean HR (bpm)")
    ax.set_ylabel("PPG − ECG (bpm)")
    ax.set_title("Descriptive Bland–Altman; overlapping repeated windows")
    return fig


def movement_error(pairs):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pairs.movement, np.abs(pairs.ppg_hr - pairs.ecg_hr), s=5, alpha=0.25)
    ax.set_xlabel("Demeaned wrist acceleration RMS (stored units)")
    ax.set_ylabel("Absolute HR error (bpm)")
    ax.set_title("Movement/error association — no causal inference")
    return fig
