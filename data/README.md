# Install PPG-DaLiA

1. Open the official UCI dataset page: https://archive.ics.uci.edu/dataset/495/ppg+dalia
2. Choose **Download (2.7 GB)**. The official file listing names the recordings archive **data.zip**, alongside **readme.pdf**. The browser download may wrap these files in an outer archive; extract the inner data.zip too if present. Download readme.pdf for the original format/protocol description. The public download does not require an account.
3. On Windows, right-click the ZIP and choose **Extract All...**. If Explorer cannot handle the archive, use 7-Zip. Do not leave the recordings inside a ZIP. Allow enough free space for the extracted recordings plus processed copies; no unverified extracted-size estimate is given here.
4. In your project, create **data/raw/PPG_DaLiA**. Place the extracted subject folders there. The official readme specifies one SX folder per subject with SX.pkl inside it. The outer archive wrapper folder is not specified by that readme; if your download includes one, move its subject folders into this destination.

```text
wearable-activity-hrv-analysis/
  config/settings.yaml
  data/
    README.md
    raw/
      PPG_DaLiA/
        S1/
          S1.pkl
          S1_activity.csv
          S1_quest.csv
          S1_RespiBAN.h5
          S1_E4.zip
        S2/
          S2.pkl
          ...
        ...
```

S1 and S2 above are path examples from the documented naming convention, not a claim about files currently installed. Keep all extracted subject folders. Only SX.pkl is used by this pipeline; retain the other files for provenance. The loader also discovers subject pickles recursively if an extra wrapper folder remains. Duplicate copies of one subject cause validation failure.

The configured root in config/settings.yaml is:

```yaml
dataset:
  root: data/raw/PPG_DaLiA
```

Paths resolve relative to the project, so no source-code change is needed. To use an existing external dataset, change this YAML path to an absolute directory. Load only the trusted official archive: pickle is executable.

## Verify installation

Open a terminal in **wearable-activity-hrv-analysis**, activate your virtual environment, then run:

```bash
python scripts/validate_dataset.py
```

Successful output has this format; angle brackets are placeholders, not recorded results:

```text
PPG-DaLiA Dataset Validation
Expected location: <resolved directory>
Valid subjects: <actual count>
<actual per-subject dimensions, durations, modality names and activity IDs>
Status: PASS
Scope: discovered files only; completeness against the official archive is not inferred.
```

PASS means discovered files meet implemented schema checks. It does not establish ECG detector validity or certify complete archive installation.

Then execute a small real-data run, followed by the full analysis:

```bash
python scripts/run_pipeline.py --max-subjects 2
python scripts/run_pipeline.py
python -m streamlit run dashboard/app.py
```

Two subjects allow a development GroupKFold run; its scores are not a full-dataset performance result. A one-subject run processes signals but explicitly skips model evaluation. Real integration and generated outputs remain unverified until these commands are run on actual recordings.

Raw recordings and generated outputs are excluded from git. Dataset is CC BY 4.0 according to UCI; retain attribution and review the source license when redistributing data.
