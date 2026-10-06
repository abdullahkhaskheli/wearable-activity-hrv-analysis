import subprocess
import sys
from pathlib import Path
import yaml
import pytest
from wearable_hrv.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["validate_dataset.py", "run_pipeline.py"])
def test_missing_data_clean_exit(tmp_path, script):
    config = load_config().values
    config["dataset"]["root"] = str(tmp_path / "absent")
    config["output_root"] = str(tmp_path / "outputs")
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config))
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--config", str(path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2 and "DATASET NOT INSTALLED" in run.stdout
    assert "Traceback" not in run.stderr
    assert not (tmp_path / "outputs").exists()
