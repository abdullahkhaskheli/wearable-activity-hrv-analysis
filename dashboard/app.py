"""Streamlit presentation of committed pipeline outputs; no model training here."""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_hrv.config import load_config
from wearable_hrv.data_loader import ACTIVITIES

st.set_page_config(page_title="Wearable Activity & HRV Analysis", layout="wide")
config = load_config()
out = config.path("output_root")
project = config.project
st.title("Wearable Activity & HRV Analysis")
st.caption(
    "Physiological Signal Processing, Human Activity Recognition & Wearable Reliability"
)
page = st.sidebar.radio(
    "Research workspace",
    [
        "Overview",
        "Signal Explorer",
        "Activity Recognition",
        "Heart Rate & HRV",
        "Wearable Reliability",
        "Methodology",
        "Limitations",
    ],
)
st.sidebar.caption("Research/educational analysis only — not medical diagnosis.")
manifest_path = out / "reports/run_manifest.json"
manifest = None
if manifest_path.exists():
    try:
        candidate = json.loads(manifest_path.read_text())
        if candidate.get("status") == "real_data_run":
            manifest = candidate
    except (ValueError, OSError):
        st.error("Run manifest cannot be read. Re-run the pipeline.")


def table(name):
    path = out / "tables" / (name + ".csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def named(frame):
    frame = frame.copy()
    if "activity" in frame:
        frame["activity_name"] = frame.activity.map(ACTIVITIES)
    return frame


def unavailable():
    st.info(
        "Analysis results not generated yet. Add the PPG-DaLiA dataset and run the pipeline to generate real analytical results."
    )
    st.code("python scripts/run_pipeline.py")
    st.markdown("Dataset installation instructions: `data/README.md`")


if page in ("Methodology", "Limitations"):
    st.markdown((project / "docs" / (page.lower() + ".md")).read_text())
elif page == "Overview":
    st.write(
        "This platform analyses synchronized physiological and motion signals from the PPG-DaLiA dataset to investigate activity recognition, ECG-derived heart-rate variability, and wrist PPG reliability during daily activities."
    )
    if manifest is None:
        unavailable()
    else:
        data = table("dataset_summary")
        dist = named(table("activity_distribution"))
        columns = st.columns(4)
        columns[0].metric("Processed participants", data.subject_id.nunique())
        columns[1].metric(
            "Observed activities", dist.loc[dist.activity != 0, "activity"].nunique()
        )
        columns[2].metric("Recording hours", f"{data.duration_s.sum() / 3600:.2f}")
        modalities = set()
        for value in data.modalities:
            modalities.update(value.split(","))
        columns[3].metric("Recorded modalities", len(modalities))
        st.dataframe(data, hide_index=True)
        st.plotly_chart(
            px.bar(
                dist.groupby("activity_name", as_index=False).duration_s.sum(),
                x="activity_name",
                y="duration_s",
                labels={"duration_s": "Annotated duration (s)"},
            ),
            width="stretch",
        )
        st.caption(
            "Statistics refer to processed subjects, not the nominal full dataset."
        )
        st.json(manifest["metadata"])
elif manifest is None:
    unavailable()
elif page == "Signal Explorer":
    subject = st.selectbox("Participant", manifest["subject_ids"])
    path = config.path("processed_root") / (subject + ".npz")
    if not path.exists():
        st.error("Processed signal file missing. Re-run pipeline.")
    else:
        with np.load(path, allow_pickle=False) as archive:
            activity = archive["activity"]
            choices = sorted(np.unique(activity).tolist())
            selected = st.selectbox(
                "Activity", choices, format_func=lambda a: ACTIVITIES[a]
            )
            # Show one contiguous run of the chosen activity, not concatenated timestamps.
            mask = activity == selected
            changes = np.diff(np.r_[False, mask, False].astype(int))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            segment = st.selectbox(
                "Contiguous interval",
                list(range(len(starts))),
                format_func=lambda i: f"{starts[i] / 4:.1f}–{ends[i] / 4:.1f}s",
            )
            lower, upper = starts[segment] / 4, ends[segment] / 4
            signal = st.selectbox(
                "Signal", ["ECG", "BVP", "wrist_ACC", "chest_ACC", "RESP"]
            )
            rate = {
                "ECG": 700,
                "BVP": 64,
                "wrist_ACC": 32,
                "chest_ACC": 700,
                "RESP": 700,
            }[signal]
            start = st.number_input(
                "Start time (s)",
                min_value=lower,
                max_value=max(lower, upper - 0.1),
                value=lower,
            )
            length = st.number_input(
                "Excerpt length (s)",
                min_value=0.1,
                max_value=max(0.1, min(60.0, upper - start)),
                value=min(10.0, upper - start),
            )
            end = min(upper, start + length)
            a, b = int(round(start * rate)), int(round(end * rate))
            raw = archive[signal][a:b]
            # Bound rendering cost by subsampling; peak markers retain exact timestamps.
            stride = max(1, len(raw) // 10000)
            t = (a + np.arange(len(raw))[::stride]) / rate
            fig = go.Figure()
            if raw.ndim == 2:
                for j in range(3):
                    fig.add_scatter(
                        x=t, y=raw[::stride, j], name="XYZ"[j], mode="lines"
                    )
            else:
                fig.add_scatter(x=t, y=raw[::stride], name="Raw", mode="lines")
                filtered_key = signal + "_filtered"
                if filtered_key in archive and st.checkbox(
                    "Show filtered signal and detected peaks", value=True
                ):
                    filtered = archive[filtered_key]
                    fig.add_scatter(
                        x=t, y=filtered[a:b:stride], name="Filtered", mode="lines"
                    )
                    peaks = archive["ecg_peaks" if signal == "ECG" else "ppg_peaks"]
                    p = peaks[(peaks >= a) & (peaks < b)]
                    fig.add_scatter(
                        x=p / rate, y=filtered[p], name="Detected peaks", mode="markers"
                    )
                    if signal == "ECG":
                        p = archive["reference_rpeaks"]
                        p = p[(p >= a) & (p < b)]
                        fig.add_scatter(
                            x=p / rate,
                            y=filtered[p],
                            name="Dataset corrected R peaks",
                            mode="markers",
                        )
            fig.update_layout(
                xaxis_title="Recording time (s)",
                yaxis_title="Stored sensor units (uncalibrated)",
                title=f"{subject} · {ACTIVITIES[selected]} · {signal} · {rate} Hz",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Display may be subsampled for responsiveness; processing uses full-rate recordings."
            )
elif page == "Activity Recognition":
    st.subheader("Subject-independent evaluation")
    st.write(
        "GroupKFold keeps all windows from a participant in one fold. Scaling is fitted only on training subjects."
    )
    summary = table("activity_model_summary")
    if summary.empty:
        st.info(
            "Models were not evaluated in this run. At least two subjects with adequate class coverage are required."
        )
    else:
        st.dataframe(summary, hide_index=True)
        st.plotly_chart(
            px.bar(
                summary,
                x="model",
                y="macro_f1_mean",
                error_y="macro_f1_std",
                range_y=[0, 1],
            ),
            width="stretch",
        )
        st.dataframe(table("activity_model_results"), hide_index=True)
        st.dataframe(table("subject_folds"), hide_index=True)
        cm = table("confusion_matrix")
        model = st.selectbox("Model", cm.model.unique())
        matrix = (
            cm[cm.model == model]
            .groupby(["true_activity", "pred_activity"])["count"]
            .sum()
            .unstack(fill_value=0)
        )
        st.plotly_chart(
            px.imshow(
                matrix.to_numpy(),
                x=[ACTIVITIES[int(a)] for a in matrix.columns],
                y=[ACTIVITIES[int(a)] for a in matrix.index],
                labels={"x": "Predicted", "y": "Actual", "color": "Windows"},
            ),
            width="stretch",
        )
elif page == "Heart Rate & HRV":
    st.caption(
        "HRV uses corrected ECG R peaks, activity-pure 300-second segments and automated interval screening. True NN status is not certified."
    )
    hr = named(table("heart_rate_participant_summary"))
    if not hr.empty:
        st.plotly_chart(
            px.box(
                hr,
                x="activity_name",
                y="ecg_hr",
                points="all",
                hover_data=["subject_id"],
            ),
            width="stretch",
        )
        st.dataframe(table("heart_rate_by_activity"), hide_index=True)
    hrv = named(table("hrv_participant_summary"))
    if hrv.empty:
        st.info(
            "No HRV segments passed the configured duration, activity and quality requirements."
        )
    else:
        metric = st.selectbox(
            "Screened-RR HRV metric", ["mean_nn_ms", "sdnn_ms", "rmssd_ms"]
        )
        st.plotly_chart(
            px.box(
                hrv,
                x="activity_name",
                y=metric,
                points="all",
                hover_data=["subject_id"],
            ),
            width="stretch",
        )
    st.dataframe(table("hrv_segments"), hide_index=True)
elif page == "Wearable Reliability":
    st.dataframe(table("wearable_reliability"), hide_index=True)
    pairs = named(table("movement_vs_error"))
    if pairs.empty:
        st.info("No paired windows passed the quality checks.")
    else:
        selected = st.selectbox(
            "Activity", ["All", *sorted(pairs.activity_name.unique())]
        )
        if selected != "All":
            pairs = pairs[pairs.activity_name == selected]
        st.plotly_chart(
            px.scatter(
                pairs,
                x="ecg_hr",
                y="ppg_hr",
                color="activity_name",
                hover_data=["subject_id", "start_s"],
            ),
            width="stretch",
        )
        ba = pairs.assign(
            pair_mean=(pairs.ecg_hr + pairs.ppg_hr) / 2,
            difference=pairs.ppg_hr - pairs.ecg_hr,
        )
        fig = px.scatter(ba, x="pair_mean", y="difference", color="activity_name")
        bias = ba.difference.mean()
        sd = ba.difference.std(ddof=1)
        for value in (bias, bias - 1.96 * sd, bias + 1.96 * sd):
            if np.isfinite(value):
                fig.add_hline(y=value, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(
            px.box(pairs, x="activity_name", y="absolute_hr_error_bpm"), width="stretch"
        )
        st.plotly_chart(
            px.scatter(
                pairs, x="movement", y="absolute_hr_error_bpm", color="subject_id"
            ),
            width="stretch",
        )
        st.dataframe(table("movement_error_by_subject"), hide_index=True)
        st.caption(
            "Bland–Altman lines summarize repeated overlapping windows. They are descriptive; no independent-sample confidence intervals or causal claims are made."
        )
