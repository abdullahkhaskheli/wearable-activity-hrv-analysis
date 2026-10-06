"""Project-relative configuration with explicit parameter validation."""

from pathlib import Path
from dataclasses import dataclass
import math
import yaml


@dataclass(frozen=True)
class Config:
    project: Path
    values: dict

    def path(self, key: str) -> Path:
        value = self.values["dataset"]["root"] if key == "dataset" else self.values[key]
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project / path


def load_config(path: str | Path | None = None) -> Config:
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "config/settings.yaml"
    )
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a mapping")
    for key in (
        "dataset",
        "rates",
        "filters",
        "peaks",
        "rr",
        "windows",
        "hrv",
        "models",
        "seed",
        "output_root",
        "processed_root",
    ):
        if key not in values:
            raise ValueError(f"Missing configuration key: {key}")

    def positive(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value > 0

    for name, value in values["rates"].items():
        if not positive(value):
            raise ValueError(f"Invalid sampling rate: {name}")
    if values["rates"] != {"chest": 700, "bvp": 64, "wrist_acc": 32, "activity": 4}:
        raise ValueError(
            "Rates must match the documented synchronized PPG-DaLiA format"
        )
    for name, rate in (("ecg", 700), ("ppg", 64)):
        low, high = values["filters"][name]
        if not 0 < low < high < rate / 2:
            raise ValueError(f"Invalid {name} filter band")
    if (
        not isinstance(values["filters"]["order"], int)
        or values["filters"]["order"] < 1
    ):
        raise ValueError("Filter order must be a positive integer")
    for prefix in ("activity", "reliability", "hrv"):
        size, step = (
            values["windows"][prefix + suffix] for suffix in ("_s", "_step_s")
        )
        if not positive(size) or not positive(step) or step > size:
            raise ValueError(
                "Window lengths and steps must be positive; step <= length"
            )
    if values["windows"]["hrv_s"] < 300:
        raise ValueError("HRV requires at least 300-second windows in this protocol")
    if not 0 < values["windows"]["label_purity"] <= 1:
        raise ValueError("Label purity must lie in (0, 1]")
    if not 0 < values["rr"]["min_ms"] < values["rr"]["max_ms"]:
        raise ValueError("Invalid RR limits")
    if not 0 < values["rr"]["local_deviation"] < 1:
        raise ValueError("Invalid local RR deviation")
    if not positive(values["peaks"]["refractory_s"]):
        raise ValueError("Invalid refractory period")
    for key in ("ecg_prominence_mad", "ppg_prominence_mad"):
        if not positive(values["peaks"][key]):
            raise ValueError("Invalid peak prominence")
    for key in ("min_valid_fraction", "max_removed_fraction", "max_half_mean_change"):
        if not 0 <= values["hrv"][key] <= 1:
            raise ValueError("Invalid HRV quality threshold")
    for section, key, minimum in (
        ("models", "folds", 2),
        ("models", "trees", 1),
        ("hrv", "min_intervals", 3),
    ):
        if not isinstance(values[section][key], int) or values[section][key] < minimum:
            raise ValueError(f"Invalid {section}.{key}")
    if not isinstance(values["seed"], int) or values["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return Config(path.resolve().parent.parent, values)
