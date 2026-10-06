from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


def test_dashboard_all_pages_without_data(tmp_path, monkeypatch):
    import wearable_hrv.config as config_module

    original = config_module.load_config()
    values = dict(original.values)
    values["output_root"] = str(tmp_path / "absent_outputs")
    isolated = config_module.Config(original.project, values)
    monkeypatch.setattr(config_module, "load_config", lambda: isolated)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    for page in (
        "Overview",
        "Signal Explorer",
        "Activity Recognition",
        "Heart Rate & HRV",
        "Wearable Reliability",
        "Methodology",
        "Limitations",
    ):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception
        if page not in ("Methodology", "Limitations"):
            assert any(
                "Analysis results not generated yet" in box.value for box in app.info
            )
