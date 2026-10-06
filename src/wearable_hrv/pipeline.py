"""Sequential real-data pipeline with staged publication of successful outputs."""

import json
import logging
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from .data_loader import discover_subjects, load_subject, ACTIVITIES
from .validation import validate_dataset
from .preprocessing import bandpass, windows, slice_time, activity_for_window
from .signal_processing import (
    detect_peaks,
    peak_window_intervals,
    heart_rate,
    ppg_quality,
)
from .hrv_analysis import screened_hrv
from .feature_extraction import extract_features
from .activity_recognition import evaluate_models
from .reliability import agreement_table
from .statistics import (
    participant_summary,
    activity_summary,
    movement_error_association,
)
from .reporting import write_tables, provenance, analytical_report
from . import visualization as viz

LOGGER = logging.getLogger(__name__)


def run_pipeline(config, max_subjects=None):
    root = config.path("dataset")
    summary, errors = validate_dataset(root)
    if errors or summary.empty:
        raise ValueError("Dataset validation failed: " + "; ".join(errors))
    paths = discover_subjects(root)
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("--max-subjects must be >= 1")
        paths = paths[:max_subjects]
    v = config.values
    win = v["windows"]
    limits = v["rr"]
    output = config.path("output_root")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="wearable-run-", dir=output.parent))
    features = []
    pair_rows = []
    hrv_rows = []
    distribution = []
    peak_quality = []
    LOGGER.info("Processing %d actual subject files", len(paths))
    try:
        for folder in ("figures", "tables", "models", "reports", "processed"):
            (stage / folder).mkdir()
        for subject_index, path in enumerate(paths):
            s = load_subject(path)
            LOGGER.info("Processing %s", s.subject_id)
            ecg = bandpass(
                s.signals["ECG"], 700, *v["filters"]["ecg"], v["filters"]["order"]
            )
            ppg = bandpass(
                s.signals["BVP"], 64, *v["filters"]["ppg"], v["filters"]["order"]
            )
            detected = detect_peaks(
                ecg,
                700,
                v["peaks"]["refractory_s"],
                v["peaks"]["ecg_prominence_mad"],
                True,
            )
            pulses = detect_peaks(
                ppg,
                64,
                v["peaks"]["refractory_s"],
                v["peaks"]["ppg_prominence_mad"],
                False,
            )
            np.savez_compressed(
                stage / "processed" / (s.subject_id + ".npz"),
                activity=s.activity,
                ECG=s.signals["ECG"],
                ECG_filtered=ecg,
                BVP=s.signals["BVP"],
                BVP_filtered=ppg,
                wrist_ACC=s.signals["wrist_ACC"],
                chest_ACC=s.signals["chest_ACC"],
                RESP=s.signals["RESP"],
                ecg_peaks=detected,
                reference_rpeaks=s.reference_rpeaks,
                ppg_peaks=pulses,
            )
            # Coverage from annotated samples, independent of overlapping analytical windows.
            for activity, count in zip(*np.unique(s.activity, return_counts=True)):
                distribution.append(
                    {
                        "subject_id": s.subject_id,
                        "activity": int(activity),
                        "activity_name": ACTIVITIES[int(activity)],
                        "duration_s": count / 4,
                    }
                )
            for start, end in windows(
                s.duration_s, win["activity_s"], win["activity_step_s"]
            ):
                a = activity_for_window(s.activity, 4, start, end, win["label_purity"])
                if a is None:
                    continue
                acc = slice_time(s.signals["wrist_ACC"], 32, start, end)
                features.append(
                    {
                        "subject_id": s.subject_id,
                        "activity": a,
                        "start_s": start,
                        "end_s": end,
                        **{
                            "feature_" + k: val
                            for k, val in extract_features(acc, 32).items()
                        },
                    }
                )
            for start, end in windows(
                s.duration_s, win["reliability_s"], win["reliability_step_s"]
            ):
                a = activity_for_window(s.activity, 4, start, end, win["label_purity"])
                if a is None:
                    continue
                rr = peak_window_intervals(s.reference_rpeaks, 700, start, end)
                auto_rr = peak_window_intervals(detected, 700, start, end)
                ecg_hr, ecg_fraction = heart_rate(rr, **limits)
                auto_hr, _ = heart_rate(auto_rr, **limits)
                pulse_rr = peak_window_intervals(pulses, 64, start, end)
                quality = ppg_quality(
                    pulse_rr,
                    slice_time(s.signals["wrist_ACC"], 32, start, end),
                    **limits,
                )
                pair_rows.append(
                    {
                        "subject_id": s.subject_id,
                        "activity": a,
                        "start_s": start,
                        "end_s": end,
                        "ecg_hr": ecg_hr,
                        "auto_ecg_hr": auto_hr,
                        "ecg_valid_fraction": ecg_fraction,
                        "ecg_interval_count": len(rr),
                        "ppg_interval_count": len(pulse_rr),
                        **quality,
                    }
                )
                peak_quality.append(
                    {
                        "subject_id": s.subject_id,
                        "activity": a,
                        "start_s": start,
                        "auto_minus_reference_hr_bpm": auto_hr - ecg_hr,
                    }
                )
            for start, end in windows(s.duration_s, win["hrv_s"], win["hrv_step_s"]):
                a = activity_for_window(s.activity, 4, start, end, 1.0)
                if a is None:
                    continue
                rr = peak_window_intervals(s.reference_rpeaks, 700, start, end)
                result = screened_hrv(rr, **v["hrv"], **limits)
                hrv_rows.append(
                    {
                        "subject_id": s.subject_id,
                        "activity": a,
                        "start_s": start,
                        "end_s": end,
                        "duration_s": end - start,
                        **result,
                    }
                )
            if subject_index == 0:
                size = min(20.0, s.duration_s)
                for name, filtered, rate, peaks in (
                    ("ECG", ecg, 700, detected),
                    ("BVP", ppg, 64, pulses),
                ):
                    n = int(size * rate)
                    title = f"{s.subject_id} · {name} · {rate} Hz · first {size:g}s; activity IDs {sorted(set(s.activity[: int(size * 4)]))}"
                    viz.save(
                        viz.signal_comparison(
                            s.signals[name][:n],
                            filtered[:n],
                            rate,
                            peaks[peaks < n],
                            title,
                        ),
                        stage / "figures" / (name.lower() + "_raw_filtered.png"),
                    )
        f = pd.DataFrame(features)
        if f.empty:
            raise ValueError("No activity-pure feature windows")
        raw_pairs = pd.DataFrame(pair_rows)
        if raw_pairs.empty:
            raise ValueError("No activity-pure HR windows")
        # Retain all diagnostics; primary paired estimates require finite HR and >=80% valid intervals.
        keep = (
            (raw_pairs.ecg_valid_fraction >= 0.8)
            & (raw_pairs.ppg_valid_fraction >= 0.8)
            & (raw_pairs.ecg_interval_count >= 3)
            & (raw_pairs.ppg_interval_count >= 3)
            & np.isfinite(raw_pairs.ecg_hr)
            & np.isfinite(raw_pairs.ppg_hr)
        )
        pairs = raw_pairs.loc[keep].copy()
        pairs["absolute_hr_error_bpm"] = np.abs(pairs.ppg_hr - pairs.ecg_hr)
        hrv = pd.DataFrame(hrv_rows)
        if hrv.empty:
            hrv = pd.DataFrame(
                columns=[
                    "subject_id",
                    "activity",
                    "start_s",
                    "end_s",
                    "duration_s",
                    "accepted",
                    "reason",
                    "mean_nn_ms",
                    "sdnn_ms",
                    "rmssd_ms",
                ]
            )
        accepted = hrv.loc[hrv.accepted == True].copy()
        hrv_part = participant_summary(accepted, ["mean_nn_ms", "sdnn_ms", "rmssd_ms"])
        hr_finite = raw_pairs.loc[
            (raw_pairs.ecg_valid_fraction >= 0.8) & np.isfinite(raw_pairs.ecg_hr)
        ]
        hr_part = participant_summary(hr_finite, ["ecg_hr"])
        reliability = agreement_table(pairs)
        if not reliability.empty:
            reliability.insert(0, "scope", "overall")
        by_act = agreement_table(pairs, ["activity"])
        if not by_act.empty:
            by_act.insert(0, "scope", "activity")
        dist = pd.DataFrame(distribution)
        tables = {
            "dataset_summary": summary[
                summary.subject_id.isin([p.stem for p in paths])
            ],
            "activity_distribution": dist,
            "activity_features": f,
            "heart_rate_windows": raw_pairs,
            "heart_rate_participant_summary": hr_part,
            "heart_rate_by_activity": activity_summary(hr_part, ["ecg_hr"]),
            "hrv_segments": hrv,
            "hrv_participant_summary": hrv_part,
            "hrv_by_activity": activity_summary(
                hrv_part, ["mean_nn_ms", "sdnn_ms", "rmssd_ms"]
            ),
            "wearable_reliability": pd.concat([reliability, by_act], ignore_index=True),
            "wearable_reliability_by_subject": agreement_table(
                pairs, ["subject_id", "activity"]
            ),
            "movement_vs_error": pairs,
            "movement_error_by_subject": movement_error_association(pairs),
            "ecg_detector_diagnostics": pd.DataFrame(peak_quality),
        }
        model_status = "not run: fewer than two subjects"
        if f.subject_id.nunique() >= 2:
            LOGGER.info("Running subject-independent model evaluation")
            scores, model_summary, cm, splits, models = evaluate_models(
                f, v["seed"], v["models"]["folds"], v["models"]["trees"]
            )
            tables.update(
                activity_model_results=scores,
                activity_model_summary=model_summary,
                confusion_matrix=cm,
                subject_folds=splits,
            )
            joblib.dump(models, stage / "models" / "activity_models.joblib")
            viz.save(
                viz.model_comparison(model_summary),
                stage / "figures" / "model_comparison.png",
            )
            for model in cm.model.unique():
                viz.save(
                    viz.confusion(cm, model),
                    stage / "figures" / (model.replace(" ", "_") + "_confusion.png"),
                )
            model_status = "GroupKFold completed"
        viz.save(
            viz.activity_distribution(
                dist.groupby("activity", as_index=False).duration_s.sum()
            ),
            stage / "figures" / "activity_distribution.png",
        )
        if not hr_part.empty:
            viz.save(
                viz.by_activity(
                    hr_part,
                    "ecg_hr",
                    "ECG reference HR (bpm)",
                    "Participant mean HR by activity",
                ),
                stage / "figures" / "heart_rate_by_activity.png",
            )
        if not hrv_part.empty:
            for metric in ("mean_nn_ms", "sdnn_ms", "rmssd_ms"):
                viz.save(
                    viz.by_activity(
                        hrv_part,
                        metric,
                        metric + " (ms)",
                        "Participant mean screened-RR HRV",
                    ),
                    stage / "figures" / (metric + "_by_activity.png"),
                )
        if not pairs.empty:
            for name, fn in (
                ("agreement", viz.agreement_plot),
                ("bland_altman", viz.bland_altman),
                ("movement_error", viz.movement_error),
            ):
                viz.save(fn(pairs), stage / "figures" / (name + ".png"))
        write_tables(stage / "tables", tables)
        metadata = provenance(config, paths)
        metadata.update(
            run_time_utc=datetime.now(timezone.utc).isoformat(),
            model_status=model_status,
            paired_window_count=len(pairs),
            hrv_accepted_segments=len(accepted),
        )
        analytical_report(stage / "reports", tables, metadata, model_status)
        # Manifest is written last; dashboard accepts only committed runs.
        (stage / "reports" / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "real_data_run",
                    "subject_ids": [p.stem for p in paths],
                    "metadata": metadata,
                },
                indent=2,
            )
        )
        for folder in ("figures", "tables", "models", "reports"):
            target = output / folder
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(stage / folder, target)
        processed = config.path("processed_root")
        if processed.exists():
            shutil.rmtree(processed)
        shutil.copytree(stage / "processed", processed)
        LOGGER.info("Real-data run published; model status: %s", model_status)
        return metadata
    finally:
        shutil.rmtree(stage, ignore_errors=True)
