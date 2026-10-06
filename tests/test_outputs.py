"""Test-only tables verify rendering/math contracts; never dataset findings."""

import numpy as np
import pandas as pd
import pytest
from wearable_hrv import visualization as viz
from wearable_hrv.statistics import (
    participant_summary,
    activity_summary,
    movement_error_association,
)
from wearable_hrv.reporting import write_tables, sha256
from wearable_hrv.reliability import agreement_table


def test_participant_weighting():
    rows = pd.DataFrame(
        {
            "subject_id": ["A", "A", "A", "B"],
            "activity": [1] * 4,
            "hr": [60, 60, 60, 100],
        }
    )
    people = participant_summary(rows, ["hr"])
    assert activity_summary(people, ["hr"]).hr_mean.iloc[0] == 80


def test_motion_within_subject():
    pairs = pd.DataFrame(
        {
            "subject_id": ["test"] * 3,
            "movement": [1.0, 2.0, 3.0],
            "ecg_hr": [60.0] * 3,
            "ppg_hr": [61.0, 62.0, 63.0],
            "activity": [1] * 3,
        }
    )
    assert movement_error_association(pairs).movement_error_pearson_r.iloc[
        0
    ] == pytest.approx(1.0)
    assert agreement_table(pairs, ["subject_id", "activity"]).mae_bpm.iloc[0] == 2.0


def test_all_scientific_plot_contracts(tmp_path):
    pairs = pd.DataFrame(
        {
            "movement": [1.0, 2.0, 3.0],
            "ecg_hr": [60.0, 70.0, 80.0],
            "ppg_hr": [61.0, 68.0, 83.0],
        }
    )
    distribution = pd.DataFrame({"activity": [1, 2], "duration_s": [20, 30]})
    participants = pd.DataFrame({"activity": [1, 1, 2, 2], "hr": [60, 65, 75, 80]})
    cm = pd.DataFrame(
        [
            {
                "model": "test",
                "true_activity": a,
                "pred_activity": b,
                "count": int(a == b),
            }
            for a in [1, 2]
            for b in [1, 2]
        ]
    )
    summary = pd.DataFrame(
        {"model": ["test"], "macro_f1_mean": [0.5], "macro_f1_std": [0.1]}
    )
    functions = [
        lambda: viz.activity_distribution(distribution),
        lambda: viz.by_activity(participants, "hr", "bpm", "Test fixture"),
        lambda: viz.confusion(cm, "test"),
        lambda: viz.model_comparison(summary),
        lambda: viz.agreement_plot(pairs),
        lambda: viz.bland_altman(pairs),
        lambda: viz.movement_error(pairs),
        lambda: viz.signal_comparison(
            np.arange(20), np.arange(20), 10, title="Test fixture"
        ),
    ]
    for i, fn in enumerate(functions):
        path = tmp_path / f"figure-{i}.png"
        viz.save(fn(), path)
        assert path.read_bytes().startswith(b"\x89PNG")


def test_table_roundtrip_and_hash(tmp_path):
    write_tables(tmp_path, {"test_fixture": pd.DataFrame({"a": [1, 2]})})
    path = tmp_path / "test_fixture.csv"
    assert pd.read_csv(path).a.tolist() == [1, 2]
    assert len(sha256(path)) == 64
