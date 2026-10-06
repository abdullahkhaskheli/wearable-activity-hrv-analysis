"""Software orchestration check using an injected controlled test Subject.

This is NOT PPG-DaLiA integration or physiology validation. All artifacts remain
in pytest temporary directories and must never be reported as research output.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from wearable_hrv.config import Config, load_config
from wearable_hrv.data_loader import Subject
from wearable_hrv.pipeline import run_pipeline
import wearable_hrv.pipeline as pipeline


@pytest.fixture
def controlled_run(tmp_path, monkeypatch):
    config = load_config().values
    config["dataset"]["root"] = str(tmp_path / "input")
    config["output_root"] = str(tmp_path / "outputs")
    config["processed_root"] = str(tmp_path / "processed")
    config["models"]["trees"] = 5
    settings = Config(tmp_path, config)
    paths = []
    subjects = {}
    rows = []
    duration = 600
    for name in ["T1", "T2"]:
        path = tmp_path / (name + ".fixture")
        path.write_text("Controlled software fixture; not PPG-DaLiA")
        paths.append(path)
        ecg_time = np.arange(duration * 700) / 700
        ppg_time = np.arange(duration * 64) / 64
        acc_time = np.arange(duration * 32) / 32
        acc = np.column_stack(
            [np.sin(2 * np.pi * acc_time), np.zeros((len(acc_time), 2))]
        )
        acc[duration * 16 :] *= 2
        signal = {
            "ECG": np.sin(2 * np.pi * ecg_time),
            "BVP": np.sin(2 * np.pi * ppg_time),
            "wrist_ACC": acc,
            "chest_ACC": np.zeros((duration * 700, 3)),
            "RESP": np.zeros(duration * 700),
        }
        subject = Subject(
            name,
            signal,
            np.repeat([1, 2], duration * 2),
            np.arange(350, duration * 700, 700),
            np.full(297, 60.0),
            path,
        )
        subjects[path] = subject
        rows.append(
            {
                "subject_id": name,
                "duration_s": duration,
                "modalities": "ECG,BVP,wrist_ACC,chest_ACC,RESP",
            }
        )
    monkeypatch.setattr(
        pipeline, "validate_dataset", lambda root: (pd.DataFrame(rows), [])
    )
    monkeypatch.setattr(pipeline, "discover_subjects", lambda root: paths)
    monkeypatch.setattr(pipeline, "load_subject", lambda path: subjects[path])
    metadata = run_pipeline(settings)
    return settings, metadata


def test_pipeline_orchestration_contract(controlled_run):
    config, metadata = controlled_run
    assert metadata["model_status"] == "GroupKFold completed"
    assert metadata["hrv_accepted_segments"] == 4
    folder = config.path("output_root")
    splits = pd.read_csv(folder / "tables/subject_folds.csv")
    for row in splits.itertuples():
        assert not set(row.train_subjects.split(",")) & set(
            row.test_subjects.split(",")
        )
    assert (folder / "reports/analytical_report.md").is_file()
    assert (folder / "figures/bland_altman.png").is_file()
    assert (config.path("processed_root") / "T1.npz").is_file()


def test_dashboard_populated_views_contract(controlled_run, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import wearable_hrv.config as config_module

    config, _ = controlled_run
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    path = Path(__file__).resolve().parents[1] / "dashboard/app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    for page in [
        "Signal Explorer",
        "Activity Recognition",
        "Heart Rate & HRV",
        "Wearable Reliability",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
