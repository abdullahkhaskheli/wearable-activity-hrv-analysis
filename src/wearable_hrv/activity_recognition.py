"""Subject-independent baselines; preprocessing fitted inside each fold."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def subject_folds(groups, n_splits=5):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError(
            "Subject-independent evaluation requires at least two subjects"
        )
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for train, test in splitter.split(np.zeros(len(groups)), groups=groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Subject leakage")
        yield train, test


def make_models(seed=42, trees=200):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, class_weight="balanced", random_state=seed, n_jobs=1
        ),
    }


def evaluate_models(features: pd.DataFrame, seed=42, folds=5, trees=200):
    columns = [c for c in features if c.startswith("feature_")]
    if not columns:
        raise ValueError("No feature columns")
    x = features[columns].to_numpy(dtype=float)
    y, groups = features.activity.to_numpy(), features.subject_id.to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Features contain non-finite values")
    labels = np.sort(np.unique(y))
    if len(labels) < 2:
        raise ValueError("Classification needs at least two activity classes")
    rows, cms, split_rows = [], [], []
    for fold, (train, test) in enumerate(subject_folds(groups, folds), 1):
        split_rows.append(
            {
                "fold": fold,
                "train_subjects": ",".join(sorted(set(groups[train]))),
                "test_subjects": ",".join(sorted(set(groups[test]))),
            }
        )
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"Fold {fold} training data contain only one class")
        for name, model in make_models(seed, trees).items():
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            p, r, f, _ = precision_recall_fscore_support(
                y[test], pred, labels=labels, average="macro", zero_division=0
            )
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "accuracy": accuracy_score(y[test], pred),
                    "macro_precision": p,
                    "macro_recall": r,
                    "macro_f1": f,
                    "n_test_windows": len(test),
                }
            )
            cm = confusion_matrix(y[test], pred, labels=labels)
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    cms.append(
                        {
                            "model": name,
                            "fold": fold,
                            "true_activity": int(a),
                            "pred_activity": int(b),
                            "count": int(cm[i, j]),
                        }
                    )
    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby("model")[
            ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, c)) for c in summary.columns]
    # Full-data fitted models are deployment artifacts; never used for reported CV scores.
    fitted = make_models(seed, trees)
    for model in fitted.values():
        model.fit(x, y)
    return (
        scores,
        summary,
        pd.DataFrame(cms),
        pd.DataFrame(split_rows),
        {"models": fitted, "feature_columns": columns, "labels": labels.tolist()},
    )
