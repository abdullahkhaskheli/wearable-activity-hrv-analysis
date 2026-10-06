"""Validation reports calculated from files, not nominal subject counts."""

from pathlib import Path
import pandas as pd
from .data_loader import discover_subjects, load_subject, DatasetError


def validate_dataset(root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, errors = [], []
    try:
        paths = discover_subjects(root)
    except (FileNotFoundError, DatasetError) as exc:
        return pd.DataFrame(), [str(exc)]
    present = {p.stem for p in paths}
    for folder in root.rglob("S*"):
        if folder.is_dir() and folder.name[1:].isdigit() and folder.name not in present:
            errors.append(f"{folder.name}: subject folder has no synchronized pickle")
    for path in paths:
        try:
            s = load_subject(path)
            rows.append(
                {
                    "subject_id": s.subject_id,
                    "source": str(path.relative_to(root)),
                    "duration_s": s.duration_s,
                    "ecg_samples": len(s.signals["ECG"]),
                    "bvp_samples": len(s.signals["BVP"]),
                    "wrist_acc_samples": len(s.signals["wrist_ACC"]),
                    "activity_samples": len(s.activity),
                    "reference_rpeaks": len(s.reference_rpeaks),
                    "hr_targets": len(s.reference_hr_labels),
                    "activities": ",".join(map(str, sorted(set(s.activity)))),
                    "modalities": ",".join(s.signals),
                }
            )
        except (DatasetError, FileNotFoundError) as exc:
            errors.append(str(exc))
    return pd.DataFrame(rows), errors


def print_validation(root: Path) -> bool:
    print("PPG-DaLiA Dataset Validation")
    print(f"Expected location: {root}")
    if not root.is_dir():
        print(
            "Status: DATASET NOT INSTALLED\nPPG-DaLiA dataset not found.\nDownload and extract the official dataset here. See data/README.md.\nNo physiological analysis has been executed."
        )
        return False
    table, errors = validate_dataset(root)
    print(f"Valid subjects: {len(table)}")
    if not table.empty:
        print(table.to_string(index=False))
    for error in errors:
        print(f"ERROR: {error}")
    passed = bool(len(table)) and not errors
    print("Status: " + ("PASS" if passed else "FAIL"))
    print(
        "Scope: discovered files only; completeness against the official archive is not inferred."
    )
    return passed
