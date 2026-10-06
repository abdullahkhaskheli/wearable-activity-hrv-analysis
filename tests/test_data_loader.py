import pickle
import numpy as np
import pytest
from wearable_hrv.data_loader import load_subject, discover_subjects, DatasetError
from wearable_hrv.validation import validate_dataset


def test_fixture_load(subject_file):
    path, _ = subject_file
    s = load_subject(path)
    assert s.subject_id == "S1" and s.signals["ECG"].shape == (14000,)
    assert s.duration_s == 20 and set(s.activity) == {1}
    assert discover_subjects(path.parent.parent) == [path]


def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subject(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        discover_subjects(tmp_path / "missing")


@pytest.mark.parametrize(
    "mutation", ["signal", "activity", "duration", "peaks", "subject", "targets", "nan"]
)
def test_malformed(subject_file, mutation):
    path, data = subject_file
    if mutation == "signal":
        del data["signal"]["wrist"]["BVP"]
    if mutation == "activity":
        data["activity"][0] = 9
    if mutation == "duration":
        data["signal"]["wrist"]["ACC"] = np.zeros((10, 3))
    if mutation == "peaks":
        data["rpeaks"] = np.array([1, 1])
    if mutation == "subject":
        data["subject"] = "S2"
    if mutation == "targets":
        data["label"] = np.ones(50)
    if mutation == "nan":
        data["signal"]["chest"]["ECG"][0] = np.nan
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(DatasetError):
        load_subject(path)
    table, errors = validate_dataset(path.parent.parent)
    assert table.empty and errors


def test_corrupt(tmp_path):
    path = tmp_path / "S1.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(DatasetError):
        load_subject(path)


def test_missing_subject_file(subject_file):
    path, _ = subject_file
    (path.parent.parent / "S2").mkdir()
    _, errors = validate_dataset(path.parent.parent)
    assert any("S2" in e for e in errors)
