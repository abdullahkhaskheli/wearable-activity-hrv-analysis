# Wearable Activity & HRV Analysis

Physiological Signal Processing, Human Activity Recognition & Wearable Reliability

## Overview
A Python research-software pipeline for synchronized PPG-DaLiA recordings, connecting motion classification, ECG-derived HR/HRV and wrist PPG agreement with corrected chest ECG reference peaks. Designed as a Software Engineering portfolio project with traceable computations, modular data processing and deterministic tests. The pipeline has been run on the real PPG-DaLiA recordings; results are reported below and are descriptive within this dataset.

## Problem
Movement can interfere with optical pulse measurements. Evaluating wearable algorithms requires synchronized reference signals, explicit quality rules and participant-independent validation.

## Research Questions
1. Can wrist acceleration distinguish annotated daily activities across held-out participants?
2. How do reference ECG HR and screened-RR HRV vary across eligible activity segments?
3. How closely does baseline PPG HR agree with reference ECG HR, and how is error associated with movement?

## Why This Matters
Software architecture, signal processing and evaluation choices influence what can be concluded from wearable data. This project makes those choices inspectable rather than hiding computations behind a dashboard.

## Results
Real PPG-DaLiA recordings were processed with the full pipeline (15 subject recordings; the five-fold GroupKFold run completed). All figures below come from the generated tables in `outputs/tables/` and are descriptive within this dataset. Run provenance (parameters, seeds, file hashes, package versions) is in `outputs/reports/`.

### Activity recognition
Logistic Regression and Random Forest were evaluated on 23,379 activity windows using subject-independent five-fold GroupKFold. Scaling was fitted inside training folds only, and the fold subject IDs are saved in `subject_folds.csv` as leakage evidence. Values are the mean ± standard deviation across the five folds.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.607 ± 0.088 | 0.640 ± 0.096 | 0.667 ± 0.119 | 0.634 ± 0.107 |
| Random Forest | 0.638 ± 0.062 | 0.668 ± 0.074 | 0.654 ± 0.086 | 0.643 ± 0.079 |

The standard deviations across folds are large relative to the difference between the two models, so these results do not establish that one model is better than the other.

### Wearable reliability (baseline PPG HR vs corrected ECG reference HR)
Across 20,234 screened, overlapping paired windows: mean absolute error 5.32 bpm, RMSE 10.16 bpm, mean bias +2.01 bpm, Pearson r 0.913, with descriptive Bland-Altman limits of agreement from -17.51 to +21.53 bpm.

Error depends strongly on activity:

| Activity (PPG-DaLiA ID) | Paired windows | MAE (bpm) | RMSE (bpm) |
|---|---|---|---|
| Sitting (1) | 3,398 | 2.48 | 5.48 |
| Stairs (2) | 723 | 16.02 | 23.41 |
| Table soccer (3) | 401 | 11.41 | 16.08 |
| Cycling (4) | 1,698 | 4.12 | 8.56 |
| Driving (5) | 3,070 | 5.08 | 8.68 |
| Lunch break (6) | 5,480 | 5.59 | 9.55 |
| Walking (7) | 957 | 11.03 | 16.86 |
| Working (8) | 4,507 | 4.27 | 8.93 |

The largest errors occur during stairs, table soccer and walking, and the smallest during sitting. Movement-versus-error associations are computed per participant (`movement_error_by_subject.csv`); no causal claim is made.

### HRV
Mean NN, SDNN, RMSSD, pNN50, SD1 and SD2 were computed from screened reference RR intervals on stable, activity-pure 300-second segments. Segment-level results are in `hrv_segments.csv`, with summaries by participant (`hrv_participant_summary.csv`) and by activity (`hrv_by_activity.csv`). True NN status is not certified.

### Reading these results
Overlapping paired windows are repeated observations from the same participants, not independent samples, so the limits of agreement and error statistics are descriptive and carry no independent-sample confidence intervals. No clinical conclusions are supported.

## Architecture
`config/` contains settings. `src/wearable_hrv/` separates loading, validation, preprocessing, peaks, HRV, features, classification, agreement, descriptive statistics, plots, reporting and orchestration. `scripts/` exposes command-line workflows. `dashboard/` reads saved outputs only. `tests/` separates deterministic fixtures from optional real-data checks. `notebooks/` provides exploratory clients of reusable modules.

`pipeline.py` is an additional module to keep CLI wrappers small. Full-rate processed arrays are saved to `data/processed/`; CSV tables, figures, joblib models and run reports go into `outputs/`.

## Dataset
Uses official PPG-DaLiA synchronized SX.pkl files. The official documentation specifies modality rates, activity labels and corrected ECG peaks. **The dataset is not bundled or installed in this repository.** See [data/README.md](data/README.md) and [docs/dataset.md](docs/dataset.md). `activity` contains activity IDs; `label` contains reference HR targets. Table soccer is the documented activity.

## Signal Processing
Offline zero-phase ECG/PPG filtering, baseline peak detection, strict temporal slicing and RR screening. Corrected dataset R peaks drive the primary ECG reference. Automated ECG peak outputs are distinct diagnostics. PPG is a baseline detector with motion diagnostics, not validated artifact compensation. Parameters and definitions: [methodology](docs/methodology.md).

## Activity Recognition
Logistic Regression and Random Forest use subject-independent GroupKFold. Scaling stays inside training folds. Saved fold subject IDs provide leakage evidence. A single-subject run explicitly skips classification; two subjects enable only a development comparison.

## HRV Analysis
Mean NN, SDNN, RMSSD, pNN50, SD1 and SD2 from screened reference RR intervals, preserving original adjacency. Only complete activity-pure 300-second segments passing quality/stability screens are retained. True NN status is not certified. Frequency-domain HRV is intentionally excluded.

## Wearable Reliability
Paired ECG/PPG HR windows provide MAE, RMSE, bias, correlation and descriptive Bland-Altman lines, overall and by activity/participant. All window diagnostics remain available. Movement/error association is analyzed per participant; no causal claim or window-level significance test.

## Installation
Python 3.11 or newer. From the project directory on Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

For Command Prompt use `.venv\Scripts\activate.bat`. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation, use Command Prompt or call `.venv\Scripts\python.exe` directly. Streamlit can always be invoked as `python -m streamlit` without relying on PATH.

## Dataset Installation
Download the official `data.zip` listed at https://archive.ics.uci.edu/dataset/495/ppg+dalia (approximately 2.7 GB) and the format `readme.pdf`. Extract subject folders to `data/raw/PPG_DaLiA/`. Do not load untrusted pickle files. Full Windows instructions: [data/README.md](data/README.md).

## Running Dataset Validation
```bash
python scripts/validate_dataset.py
```
Missing/invalid data returns exit code 2 with clear instructions. PASS describes observed schema checks, not physiological algorithm validation or inferred archive completeness.

## Running Tests
```bash
python -m pytest -q
python -m pytest -q -m "not real_data"
```
The optional installed-dataset test skips explicitly when the dataset is absent. Deterministic controlled signals and temporary schema fixtures exist only in tests and never enter production analysis. Development test evidence is recorded in [docs/development_audit.md](docs/development_audit.md).

## Running Analysis
```bash
python scripts/run_pipeline.py --max-subjects 2
python scripts/run_pipeline.py
```
Use `--config path/to/settings.yaml` for an alternate project-relative configuration. `--max-subjects 2` selects numerically sorted discovered subject files; it is a development run, not a full-dataset result. No source modifications are necessary. Missing recordings produce no research outputs. Successful runs replace prior outputs; copy outputs first if you need to preserve a prior run.

## Running Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Before analysis, all result pages show an honest empty state. Methodology and limitations remain accessible. After a committed real-data run, the dashboard reads tables, saved signals and provenance. Signal Explorer selects participant, activity, contiguous interval, modality and elapsed time, with raw/filtered and peak views. Classification and physiological/reliability pages display generated results.

## Methodology
See [docs/methodology.md](docs/methodology.md). HR/HRV and classification use distinct window requirements. Every output retains subject IDs, temporal boundaries or participant-level aggregation. Pooled reliability is descriptive and supplemented by per-participant tables.

## Generated Outputs
A real-data run produces tables including dataset summary, annotated activity duration, feature windows, fold scores/summary, subject folds, confusion matrices, HR windows/summaries, HRV segments, wearable agreement and movement/error associations. Figures include raw/filtered signals, annotated duration, eligible HR/HRV distributions, model summaries/confusion matrices, agreement, Bland-Altman and movement/error scatter. Unsupported/empty analyses are recorded rather than fabricated. Provenance and run status live in `outputs/reports/`.

## Limitations
See [docs/limitations.md](docs/limitations.md). The pipeline has been executed on real PPG-DaLiA recordings, but the following limits apply:
- Results are descriptive within one dataset; they do not establish how the methods perform on other devices, populations or protocols.
- Overlapping paired windows are repeated observations, not independent samples.
- PPG peak detection is a baseline without active motion compensation, and interval screening alone cannot establish pulse correctness.
- Corrected reference peaks do not validate the separate automatic ECG detector.
- Sinus rhythm / true NN status is not certified for the screened RR intervals.
- Memory and disk requirements have not been profiled.

This is computational research, not clinical validation.

## Reproducibility
YAML parameters and fixed model seeds are recorded with SHA256 hashes of actual subject files, Python/package versions and UTC execution time. After installation, `python -m pip freeze > environment-lock.txt` records your exact environment. The development dependency snapshot is supplied as `development-environment.txt`; it is provenance, not a portable lock file for all OSes. Notebooks require Jupyter supplied separately and do not replace the pipeline.

## Dataset Citation
Reiss, A., Indlekofer, I., & Schmidt, P. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890

Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks. Sensors 19(14), 3079. https://doi.org/10.3390/s19143079

## License
Software: MIT, see LICENSE. Dataset: CC BY 4.0 according to UCI, separately licensed and not bundled. Research/educational analysis only, not medical diagnosis. My background is Software Engineering; this project does not claim clinical or biomedical expertise.