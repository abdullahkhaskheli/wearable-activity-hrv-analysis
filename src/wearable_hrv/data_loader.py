"""Load only synchronized official SX.pkl recordings.

Pickle is executable. This loader intentionally permits pickle only for locally
installed trusted official dataset files; never open arbitrary uploaded pickle.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import numpy as np

RATES = {"chest": 700, "bvp": 64, "wrist_acc": 32, "activity": 4}
ACTIVITIES = {
    0: "Transition",
    1: "Sitting",
    2: "Stairs",
    3: "Table soccer",
    4: "Cycling",
    5: "Driving",
    6: "Lunch",
    7: "Walking",
    8: "Working",
}


class DatasetError(ValueError):
    """Dataset schema or recording integrity error."""


@dataclass
class Subject:
    subject_id: str
    signals: dict[str, np.ndarray]
    activity: np.ndarray
    reference_rpeaks: np.ndarray
    reference_hr_labels: np.ndarray
    source: Path

    @property
    def duration_s(self) -> float:
        return len(self.signals["ECG"]) / RATES["chest"]


def discover_subjects(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError("PPG-DaLiA dataset not found.")
    files = sorted(
        (p for p in root.rglob("S*.pkl") if re.fullmatch(r"S[0-9]+", p.stem)),
        key=lambda p: int(p.stem[1:]),
    )
    if not files:
        raise DatasetError("No synchronized SX.pkl files found")
    ids = [p.stem for p in files]
    if len(set(ids)) != len(ids):
        raise DatasetError("Duplicate subject IDs: remove duplicate extracted copies")
    return files


def _array(value, name: str, columns: int = 1) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise DatasetError(f"{name} is not numeric")
    if columns == 1 and arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if (columns == 1 and arr.ndim != 1) or (
        columns > 1 and (arr.ndim != 2 or arr.shape[1] != columns)
    ):
        raise DatasetError(f"Invalid dimensions for {name}: {arr.shape}")
    if not len(arr) or not np.isfinite(arr).all():
        raise DatasetError(f"{name} is empty or contains non-finite values")
    return arr


def load_subject(path: str | Path) -> Subject:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        if not isinstance(data, dict):
            raise DatasetError("Subject payload must be a dictionary")
        for key in ("signal", "activity", "label", "rpeaks", "subject"):
            if key not in data:
                raise DatasetError(f"Missing required key: {key}")
        subject_id = str(data["subject"])
        if subject_id.isdigit():
            subject_id = "S" + subject_id
        if subject_id != path.stem:
            raise DatasetError("Subject identifier does not match filename")
        signals = {}
        for group, name, alias, cols in (
            ("chest", "ECG", "ECG", 1),
            ("chest", "ACC", "chest_ACC", 3),
            ("chest", "RESP", "RESP", 1),
            ("wrist", "BVP", "BVP", 1),
            ("wrist", "ACC", "wrist_ACC", 3),
        ):
            source = data["signal"][group]
            # The released PPG-DaLiA pickle spells these chest channels with
            # mixed case (Resp, Temp), while some prose/schema summaries use
            # uppercase abbreviations. Normalize to stable internal aliases.
            source_name = "Resp" if name == "RESP" and "Resp" in source else name
            if source_name not in source:
                raise DatasetError(
                    f"Missing required signal: signal.{group}.{source_name}"
                )
            signals[alias] = _array(source[source_name], alias, cols)
        for name in ("EDA", "TEMP"):
            if name in data["signal"]["wrist"]:
                signals[name] = _array(data["signal"]["wrist"][name], name)
        activity = _array(data["activity"], "activity")
        if (
            not np.equal(activity, np.floor(activity)).all()
            or not np.isin(activity, list(ACTIVITIES)).all()
        ):
            raise DatasetError("Activity IDs must be integers 0..8")
        peaks = _array(data["rpeaks"], "rpeaks")
        # Normalize reference peak annotations
        descending = int(np.count_nonzero(np.diff(peaks) < 0))
        cleaned = np.unique(peaks)
        duplicates = len(peaks) - len(cleaned)
        if descending or duplicates:
            print(
                f"{subject_id}: sorted {descending} descending peak steps; "
                f"removed {duplicates} exact duplicate annotations",
                flush=True,
            )
        peaks = cleaned
        if len(peaks) < 2:
            raise DatasetError("At least two distinct reference peaks required")

        if (
            not np.equal(peaks, np.floor(peaks)).all()
            or np.any(np.diff(peaks) <= 0)
            or peaks[0] < 0
            or peaks[-1] >= len(signals["ECG"])
        ):
            raise DatasetError(
                "Reference R peaks must be sorted unique valid ECG indices"
            )
        targets = _array(data["label"], "label")
        if np.any(targets <= 0):
            raise DatasetError("Reference HR labels must be positive")
        duration = len(signals["ECG"]) / 700
        for name, rate in (
            ("chest_ACC", 700),
            ("RESP", 700),
            ("BVP", 64),
            ("wrist_ACC", 32),
            ("EDA", 4),
            ("TEMP", 4),
        ):
            if name in signals and abs(len(signals[name]) / rate - duration) > 0.5:
                raise DatasetError(f"Duration mismatch: {name}")
        if abs(len(activity) / 4 - duration) > 0.5:
            raise DatasetError("Activity labels do not align with recording duration")
        expected = max(0, int(np.floor((duration - 8) / 2)) + 1)
        if abs(len(targets) - expected) > 1:
            raise DatasetError(
                "HR target count does not match documented 8s/2s windows"
            )
        return Subject(
            subject_id,
            signals,
            activity.astype(int),
            peaks.astype(int),
            targets.astype(float),
            path,
        )
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        KeyError,
        TypeError,
        AttributeError,
        ImportError,
        ValueError,
    ) as exc:
        raise DatasetError(f"{path.name}: {exc}") from exc
