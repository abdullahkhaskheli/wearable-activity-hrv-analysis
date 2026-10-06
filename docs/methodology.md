# Methodology

## Data and synchronization
Use only synchronized SX.pkl files from PPG-DaLiA. Activity labels are 4 Hz; HR targets are separate 8s/2s reference labels. Subject identity remains on every window. Modality slicing uses elapsed seconds and each documented sampling rate. Transition windows are excluded from primary activity analyses.

## Signal preprocessing and peaks
ECG uses a fourth-order Butterworth 5–25 Hz bandpass; PPG uses 0.5–4 Hz. SciPy second-order sections with forward/backward filtering avoid phase shift offline. These are configurable baseline engineering choices, not the dataset authors' reference detector. Filter end effects and amplitude changes remain; raw/filtered plots support inspection. Full recordings are filtered before slicing windows.

ECG detection finds prominent absolute deflections with a 0.3-second refractory interval, supporting both polarities. This baseline may confuse T waves or artifacts with R waves. PPG detection finds positive filtered pulse maxima with a refractory interval. Prominence is scaled to the median absolute deviation with an amplitude floor. No active motion compensation is claimed.

Primary ECG reference HR and HRV use dataset-provided manually corrected R peaks. Independently detected ECG peaks and their HR differences are saved for diagnostic comparison. Reference HR labels are loaded and shape checked; primary paired HR is recomputed from peak intervals on exactly matching ECG/PPG time windows, rather than silently reusing labels under a different aggregation formula.

Intervals are screened to 300–2000 ms and within 20% of a local five-interval median when enough intervals exist. These broad engineering thresholds remove obvious extremes, not every artifact or ectopic beat. HR is the mean instantaneous rate (mean of 60000/RR), not 60000/mean(RR).

## Activity features and learning
Default motion windows: 8 seconds, step 4 seconds; label purity 100%. Wrist tri-axis acceleration features include axis/magnitude mean, population standard deviation, variance, RMS, range, IQR and mean squared energy. Orientation/gravity features preserve posture information; dynamic magnitude spectra describe movement. Dominant non-DC frequency and integrated spectral density are calculated between 0.25 Hz and min(10 Hz, Nyquist). This simple baseline includes some redundant features deliberately; no feature selection is fitted on held-out subjects.

Logistic Regression with training-fold StandardScaler and class-balanced Random Forest are evaluated using GroupKFold, up to five folds or the available number of subjects. Seeds are fixed, forest uses one worker. Training/test subject IDs are saved and asserted disjoint. Report all fold accuracy, macro precision, macro recall, macro F1, their mean/SD and aggregate out-of-fold confusion matrices. Macro metrics use the same observed class set across folds; classes missing from a fold receive zero undefined scores. Full-data fitted model artifacts are not evaluation evidence. No random-window split is used.

## HRV
Only complete 300-second windows of one non-transition activity are eligible (default non-overlap). Minimum 30 intervals, >=95% valid and <=5% removed, plus <=10% difference between first-/second-half RR means. This simple stability screen cannot guarantee stationarity. Rejected segments and reasons remain visible.

Mean NN = mean of screened RR. SDNN uses sample SD (ddof=1). RMSSD uses RMS of successive differences only when both original adjacent intervals survive screening. pNN50 uses absolute differences strictly greater than 50 ms divided by retained adjacent pairs. SD1 is sample SD of successive differences / sqrt(2), SD2 = sqrt(max(0, 2 SDNN² − SD1²)). These estimates use screened RR, without certification of sinus NN rhythm. SD1 is not always exactly RMSSD/sqrt(2) for finite samples with a nonzero mean difference. No LF, HF or LF/HF is implemented.

## PPG reliability and motion
Paired HR uses activity-pure 8-second windows at a 2-second step, both interval endpoints inside the window. Require at least three intervals and >=80% valid for each modality. Save all raw window diagnostics and report retained pairs separately. Do not hide rejected-window counts or interpret accepted windows as the full recording.

Mean bias = PPG − ECG; MAE and RMSE use paired finite values. Pearson correlation is undefined for constant sequences. Bland–Altman limits use bias ±1.96 sample SD of differences, as descriptive conventional lines for repeated overlapping observations; no independent-sample uncertainty or clinical agreement claim. Pooled activity results are duration/window-weighted and supplemented with participant/activity results.

Movement intensity is RMS of per-axis demeaned wrist acceleration in stored units. It is a movement proxy, not mechanical energy or calibrated metabolic intensity. Movement/error correlation is computed per subject; no p-values, causal conclusions or pseudoreplicated significance tests.

## Outputs and reproducibility
Pipeline stages outputs before publication; unsuccessful computation does not publish a new manifest. It processes one subject at a time, saves full-rate compressed signal excerpts for the explorer, tables, figures, models and a report. Provenance records input SHA256, configuration, package/Python versions and UTC run time. A run with missing data returns status 2 and writes no research output. The dashboard requires a committed real-data manifest and never trains models.

Configuration changes may alter scientific estimands; compare complete run provenance. Run directories replace previous results on success, so copy a run before changing settings if comparisons are required.
