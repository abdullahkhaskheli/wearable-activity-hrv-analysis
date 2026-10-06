# Software development audit

## Status definitions

- IMPLEMENTED: source code exists.
- UNIT TESTED: deterministic software/math contracts passed.
- REAL-DATA VALIDATED: actual PPG-DaLiA recording processing passed.
- RESULT VERIFIED: a numerical finding is traceable to real-data execution.

This stage reaches IMPLEMENTED and UNIT TESTED only. PPG-DaLiA has not yet been integrated. No real-data analysis or result verification has taken place.

## Evidence by capability

| Capability | Implementation | Deterministic verification |
| --- | --- | --- |
| Project-relative validated configuration | src/wearable_hrv/config.py; config/settings.yaml | tests/test_config.py |
| Subject discovery, schema/duration/peak/target checks | data_loader.py; validation.py | tests/test_data_loader.py (explicit temporary schema fixtures) |
| Filtering, slicing, label purity, acceleration | preprocessing.py | tests/test_preprocessing.py; attenuation and boundary checks |
| ECG/PPG baseline detectors, RR and HR | signal_processing.py | tests/test_signal_processing.py; controlled signals only |
| HRV mathematics and screening | hrv_analysis.py | tests/test_hrv.py; independently derived answers and adjacency checks |
| Motion features | feature_extraction.py | tests/test_features.py; exact constant/spectral inputs |
| Subject-independent models | activity_recognition.py | tests/test_activity_recognition.py; fold isolation/reproducibility |
| Agreement and Bland–Altman mathematics | reliability.py | tests/test_reliability.py; exact MAE/RMSE/bias/limits |
| Participant summaries and movement association | statistics.py | tests/test_outputs.py |
| Figures, table serialization and hashing | visualization.py; reporting.py | tests/test_outputs.py |
| Pipeline publication and module wiring | pipeline.py; scripts/run_pipeline.py | tests/test_pipeline_contract.py; injected controlled Subject objects |
| Graceful missing data | scripts/validate_dataset.py; scripts/run_pipeline.py | tests/test_cli.py and actual missing-data CLI invocations |
| Dashboard navigation, populated/empty states | dashboard/app.py | tests/test_dashboard.py; tests/test_pipeline_contract.py; Streamlit AppTest |
| Real dataset installation instructions | data/README.md; docs/dataset.md | Grounded in official UCI page/readme; actual archive has not been inspected locally |
| Executable exploratory clients | notebooks/*.ipynb | JSON structure checked; execution with real data not verified |

Implementation names in the table are relative to src/wearable_hrv unless another directory is stated.

## Actual verification

Command: `python -m pytest -q`

See [pytest-output.txt](pytest-output.txt) for captured actual output. The single skip is the optional `real_data` test because recordings are unavailable.

`python -m compileall -q src scripts dashboard`: passed.

`python -m ruff check --select F .`: all checks passed after removing two unused imports. Source formatted with Ruff.

Streamlit server: launched headless, health endpoint returned `HTTP 200 ok`, server then terminated. AppTest exercised all seven missing-data pages and populated result views without exceptions. This verifies application behavior, not visual browser QA or research validity.

Actual validator and pipeline execution both returned exit code 2:

```text
PPG-DaLiA Dataset Validation
Status: DATASET NOT INSTALLED
PPG-DaLiA dataset not found.
No physiological analysis has been executed.
```

Pipeline additionally printed: `Dataset unavailable or invalid. Install PPG-DaLiA first. See data/README.md.` No research output was generated.

## Test scope and artifacts

Controlled test signals, schema files and feature tables are explicitly test-only. Orchestration/UI fixtures create temporary output inside pytest-managed directories; they are not PPG-DaLiA integration tests or physiological evidence. No fixture results are bundled in outputs/ or data/processed/.

Delivered outputs/ and processed data contain directory placeholders only. Generated deliverables are this source archive, documentation, dependency snapshot and actual test log. There are zero processed real subject files and zero real research result tables/figures.

## Pending verification

1. Install and inspect actual SX.pkl recordings, check documented shape/alignment tolerances against them.
2. Validate files with the supplied validator.
3. Run a two-subject development analysis and inspect raw/filtered ECG/PPG against corrected reference peaks.
4. Run all available subjects, inspect rejected-window coverage and HRV eligibility.
5. Review scientific results and detector limitations before making any portfolio performance claim.

Full real-data pipeline, real model performance, sensor agreement findings, true NN certification, clinical relevance and browser visual QA: **Not verified in the current environment.**

See data/README.md for exact installation and run commands.
