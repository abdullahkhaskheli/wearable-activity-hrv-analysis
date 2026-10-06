import pytest
import yaml
from wearable_hrv.config import load_config


def test_project_relative():
    config = load_config()
    assert config.path("dataset") == config.project / "data/raw/PPG_DaLiA"
    assert config.values["rates"]["activity"] == 4


@pytest.mark.parametrize(
    "key,value", [("activity_step_s", 0), ("hrv_s", -1), ("label_purity", 1.1)]
)
def test_invalid_window(tmp_path, key, value):
    data = load_config().values
    data["windows"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_rates(tmp_path):
    data = load_config().values
    data["rates"]["chest"] = 600
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="documented"):
        load_config(path)
