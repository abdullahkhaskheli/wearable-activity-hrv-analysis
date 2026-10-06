# Dataset format and source

Official source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
Official format documentation: https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf
Dataset DOI: https://doi.org/10.24432/C53890

Verified against the official readme on 2026-10-01; real files have not yet been inspected.
The format specification, not observed results, documents 15 participants. The code discovers files dynamically and reports only actual processed subjects.

Each subject folder SX contains SX.pkl, SX_activity.csv, SX_quest.csv, SX_RespiBAN.h5 and SX_E4.zip. Only the synchronized SX.pkl is needed by this project. Do not manually combine the raw device files: their start times and clock drift need synchronization. The provided pickle already accounts for these corrections.

## Synchronized pickle dictionary

| Key | Meaning |
| --- | --- |
| subject | Participant ID |
| activity | Activity IDs at 4 Hz, not heart rate |
| label | Mean instantaneous ECG HR for 8-second windows shifted by 2 seconds |
| rpeaks | Manually inspected/corrected indices into the 700 Hz ECG |
| signal.chest.ECG | ECG, 700 Hz |
| signal.chest.ACC | Three-axis chest acceleration, 700 Hz |
| signal.chest.RESP | Respiration, 700 Hz |
| signal.wrist.BVP | PPG/BVP, 64 Hz |
| signal.wrist.ACC | Three-axis wrist acceleration, 32 Hz |
| signal.wrist.EDA | Optional electrodermal activity, 4 Hz |
| signal.wrist.TEMP | Optional temperature, 4 Hz |
| questionnaire | Participant information; not used as predictive features |

Chest EDA, EMG and temperature are dummy channels and are deliberately excluded.
Raw E4 acceleration is documented as 1/64g, EDA as microSiemens and temperature as degrees Celsius. This software preserves synchronized pickle values without assuming they retain every raw-device calibration. ECG, BVP, chest acceleration and respiration are displayed in stored units rather than inventing physical calibration.

Activity mapping: 0 transition, 1 sitting, 2 stairs, 3 table soccer (not field soccer), 4 cycling, 5 driving, 6 lunch, 7 walking, 8 desk working.
The readme reports truncated valid data for S6 due to hardware issues; equal participant duration is not assumed.
Sampling frequencies are format constants from the readme; they are not embedded in every pickle.

Validator requires numeric finite arrays, documented dimensions, valid activity IDs, matching participant identity, sorted in-range reference peaks, and modality durations within 0.5 seconds. HR label count is checked against 8s/2s windows with a one-window tolerance. These tolerances need checking against real files before any real-data validated claim.
Pickle can execute code: load only trusted official archive files. This is not a sandboxed generic file upload parser.

## Citations
Reiss, A., Indlekofer, I., & Schmidt, P. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890

Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks. Sensors, 19(14), 3079. https://doi.org/10.3390/s19143079
