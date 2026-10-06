"""Write tables and provenance into a run staging directory."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(config, paths):
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "streamlit",
        "PyYAML",
        "joblib",
    )
    return {
        "python": platform.python_version(),
        "packages": {p: version(p) for p in packages},
        "configuration": config.values,
        "dataset_source": "https://doi.org/10.24432/C53890",
        "input_files": [
            {"subject": p.stem, "name": p.name, "sha256": sha256(p)} for p in paths
        ],
    }


def write_tables(folder, tables):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / (name + ".csv"), index=False)


def analytical_report(folder, tables, metadata, status):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    lines = [
        "# Wearable Activity & HRV Analysis",
        "",
        "Real recordings processed; all findings are descriptive within this dataset.",
        "",
        "## Execution status",
        status,
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- `{name}.csv`: {len(frame)} rows" for name, frame in tables.items())
    for name in ("activity_model_summary", "wearable_reliability"):
        if name in tables and not tables[name].empty:
            lines.extend(
                [
                    "",
                    f"## {name}",
                    "",
                    "```text",
                    tables[name].to_string(index=False),
                    "```",
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Corrected dataset R peaks are the primary ECG reference. Automatic ECG detection is retained as a separate diagnostic.",
            "PPG peak detection is a baseline without active motion compensation; interval screening alone cannot establish pulse correctness.",
            "HRV uses stable, activity-pure 300-second segments and screened RR intervals; sinus rhythm/true NN status is not certified.",
            "Overlapping paired windows are repeated observations, not independent samples. Bland–Altman limits are descriptive, without independent-sample confidence intervals.",
            "See docs/methodology.md and docs/limitations.md. No clinical conclusions are supported.",
        ]
    )
    (folder / "analytical_report.md").write_text("\n".join(lines) + "\n")
