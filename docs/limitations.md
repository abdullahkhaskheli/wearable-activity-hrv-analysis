# Limitations

- Software tests are not physiological validation. Real PPG-DaLiA integration has not yet been executed in this development stage.
- The original study is a small, protocol-driven sample, with demographic and context limits. No population-wide generalization is established.
- Activities include table soccer, not outdoor soccer. Lunch covers multiple behaviors. S6 has shortened valid recording duration according to the source documentation.
- Baseline ECG and PPG peak detectors may fail under motion. There is no active adaptive motion compensation; plausible intervals do not imply correct beats. Review raw/filtered traces against corrected reference peaks before accepting algorithm performance.
- Reference ECG uses dataset-corrected peaks. Automated ECG detection has separate diagnostics; its accuracy must not be inferred from reference-based HR/HRV.
- Screening does not certify sinus rhythm or true NN status. Five-minute activity-pure, stable HRV segments may be scarce or absent. No missing HRV findings are fabricated.
- Global filtering introduces edge effects. Strict finite-data validation rejects recordings with missing values; it does not impute them silently.
- Sampling constants and duration/count tolerances are grounded in official documentation but still need confirmation on actual files.
- Signal calibration in synchronized pickle files has not been verified locally. Stored units are preserved; no invented millivolt/g conversion is used.
- Overlapping HR windows are correlated repeated measures. Pooled errors can overweight longer recordings. Participant summaries are provided; Bland–Altman lines have no independent-sample confidence intervals.
- Mean/stdev of folds are descriptive, not robust population uncertainty estimates. Small development runs and absent training classes may not support useful classification.
- Motion/error associations are descriptive and may reflect activity, HR, subject and sensor fit. They do not establish causation.
- The processed signal explorer duplicates recordings and can use substantial disk and RAM. Subjects are processed sequentially, but a full subject is held in memory.
- Staging prevents computation failures from mixing a new run with old outputs. Publication spans multiple filesystem directories and is not a cross-directory transaction; avoid simultaneous pipeline/dashboard writes, and re-run after an interrupted publication.
- Only trusted official pickle files should be loaded. Pickle is not a safe generic exchange format.
- Research/educational computation only; no diagnosis, clinical decision support or medical device validation.
