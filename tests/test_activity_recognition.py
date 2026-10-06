import numpy as np
import pandas as pd
import pytest
from wearable_hrv.activity_recognition import subject_folds, evaluate_models


def test_group_isolation():
    groups = np.repeat(["S1", "S2", "S3", "S4"], 10)
    tests = []
    for train, test in subject_folds(groups, 3):
        assert not set(groups[train]) & set(groups[test])
        tests.extend(test.tolist())
    assert sorted(tests) == list(range(len(groups)))


def test_requires_groups():
    with pytest.raises(ValueError):
        list(subject_folds(["S1"] * 4))


def test_reproducibility_fixture():
    # Explicit test-only feature matrix, not PPG-DaLiA observations or results.
    rows = [
        {
            "subject_id": f"T{s}",
            "activity": a,
            "feature_a": a + s * 0.01,
            "feature_b": a * 2,
        }
        for s in range(4)
        for a in (1, 2)
        for _ in range(4)
    ]
    frame = pd.DataFrame(rows)
    first = evaluate_models(frame, seed=7, folds=2, trees=5)
    second = evaluate_models(frame, seed=7, folds=2, trees=5)
    pd.testing.assert_frame_equal(first[0], second[0])
    for row in first[3].itertuples():
        assert not set(row.train_subjects.split(",")) & set(
            row.test_subjects.split(",")
        )
