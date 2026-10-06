"""Fixtures deliberately constructed for software tests; never research data."""

import pickle
import numpy as np
import pytest


@pytest.fixture
def subject_file(tmp_path):
    duration = 20
    folder = tmp_path / "S1"
    folder.mkdir()
    path = folder / "S1.pkl"
    data = {
        "subject": "S1",
        "activity": np.ones(duration * 4),
        "label": np.full(7, 60.0),
        "rpeaks": np.arange(350, duration * 700, 700),
        "signal": {
            "chest": {
                "ECG": np.zeros((duration * 700, 1)),
                "ACC": np.zeros((duration * 700, 3)),
                "RESP": np.zeros((duration * 700, 1)),
            },
            "wrist": {
                "BVP": np.zeros((duration * 64, 1)),
                "ACC": np.zeros((duration * 32, 3)),
            },
        },
    }
    path.write_bytes(pickle.dumps(data))
    return path, data
