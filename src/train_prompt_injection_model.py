"""Train a separate TF-IDF + linear SVM prompt-injection classifier.

Input CSV contract:
    text   - the complete prompt or untrusted instruction
    label  - 0 for benign, 1 for prompt injection
    source - dataset/provenance group used to prevent source leakage

The caller must explicitly assign disjoint sources to fitting, calibration,
threshold selection, and testing. This script never uses the CVE model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = ROOT / "artifacts" / "prompt_injection_model.joblib"
DEFAULT_METADATA_PATH = ROOT / "artifacts" / "prompt_injection_model_metadata.json"
DEFAULT_REPORT_DIR = ROOT / "reports" / "prompt_injection_model"
DEFAULT_DATASET_MANIFEST = (
    ROOT / "data" / "prompt_injection" / "dataset_manifest.json"
)
MODEL_VERSION = "prompt_injection_tfidf_svm_v1"
REQUIRED_COLUMNS = {"text", "label", "source"}


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate labels, provenance, text, and exact-row duplicates."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    clean = data[["text", "label", "source"]].copy()
    clean["text"] = clean["text"].fillna("").astype(str).str.strip()
    clean["source"] = clean["source"].fillna("").astype(str).str.strip()
    if clean["text"].eq("").any():
        raise ValueError("Every row must contain non-empty text.")
    if clean["source"].eq("").any():
        raise ValueError("Every row must identify its source.")
    if not set(clean["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary integers: 0 or 1.")
    clean["label"] = clean["label"].astype(int)

    normalized = clean["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    conflicting = clean.assign(_normalized=normalized).groupby("_normalized")[
        "label"
    ].nunique()
    if conflicting.gt(1).any():
        raise ValueError("Identical normalized prompts have conflicting labels.")
    clean = clean.loc[~normalized.duplicated()].reset_index(drop=True)
    return clean


def partition_by_source(
    data: pd.DataFrame,
    *,
    fit_sources: list[str],
    calibration_source: str,
    threshold_source: str,
    test_source: str,
) -> dict[str, pd.DataFrame]:
    """Construct four disjoint partitions using provenance groups."""
    assignments = {
        "fit": list(fit_sources),
        "calibration": [calibration_source],
        "threshold": [threshold_source],
        "test": [test_source],
    }
    flattened = [source for values in assignments.values() for source in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Fit, calibration, threshold, and test sources must be disjoint.")
    available = set(data["source"])
    missing = set(flattened) - available
    if missing:
        raise ValueError(f"Requested sources are absent: {sorted(missing)}")

    partitions = {
        name: data[data["source"].isin(sources)].copy()
        for name, sources in assignments.items()
    }
    for name, frame in partitions.items():
        if frame.empty or frame["label"].nunique() != 2:
            raise ValueError(f"The {name} partition must contain both labels.")
    return partitions


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def evaluation_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def train_prompt_injection_model(
    partitions: dict[str, pd.DataFrame],
    *,
    min_df: int = 2,
) -> tuple[CalibratedClassifierCV, dict]:
    """Fit, calibrate, select a threshold, and evaluate once on held-out data."""
    fit = partitions["fit"]
    calibration = partitions["calibration"]
    threshold_data = partitions["threshold"]
    test = partitions["test"]

    base_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.98,
                    max_features=50_000,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    dual=True,
                    max_iter=3_000,
                    random_state=42,
                ),
            ),
        ]
    )
    base_model.fit(fit["text"], fit["label"])
    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    model.fit(calibration["text"], calibration["label"])

    threshold_probabilities = model.predict_proba(threshold_data["text"])[:, 1]
    threshold, threshold_f1 = select_f1_threshold(
        threshold_data["label"].to_numpy(), threshold_probabilities
    )
    test_probabilities = model.predict_proba(test["text"])[:, 1]
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "calibrated_tfidf_linear_svm",
        "label_definition": "1 = prompt injection; 0 = benign prompt",
        "selected_threshold": threshold,
        "threshold_best_f1": threshold_f1,
        "partition_sources": {
            name: sorted(frame["source"].unique().tolist())
            for name, frame in partitions.items()
        },
        "partition_rows": {name: int(len(frame)) for name, frame in partitions.items()},
        "test_metrics": evaluation_metrics(
            test["label"].to_numpy(), test_probabilities, threshold
        ),
        "limitations": [
            "Performance depends on the coverage and labeling quality of the supplied data.",
            "Text classification cannot replace deterministic tool authorization.",
            "Novel, obfuscated, multilingual, and indirect injections may be missed.",
        ],
    }
    return model, metadata


def save_model(model, metadata: dict, artifact_path: Path, metadata_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_evaluation_reports(
    model,
    metadata: dict,
    test: pd.DataFrame,
    report_dir: Path,
) -> None:
    """Save reproducible external-test errors and linear term coefficients."""
    report_dir.mkdir(parents=True, exist_ok=True)
    probability = model.predict_proba(test["text"])[:, 1]
    threshold = float(metadata["selected_threshold"])
    predictions = probability >= threshold
    rows = test[["text", "label", "source"]].copy()
    rows["injection_probability"] = probability
    rows["prediction"] = predictions.astype(int)
    rows["error_type"] = np.where(
        (rows["label"] == 0) & (rows["prediction"] == 1),
        "false_positive",
        np.where(
            (rows["label"] == 1) & (rows["prediction"] == 0),
            "false_negative",
            "correct",
        ),
    )
    rows.to_csv(report_dir / "external_test_predictions.csv", index=False)
    rows[rows["error_type"].eq("false_positive")].sort_values(
        "injection_probability", ascending=False
    ).to_csv(report_dir / "false_positives.csv", index=False)
    rows[rows["error_type"].eq("false_negative")].sort_values(
        "injection_probability"
    ).to_csv(report_dir / "false_negatives.csv", index=False)

    calibrated = model.calibrated_classifiers_[0]
    pipeline = calibrated.estimator
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out(),
            "coefficient": classifier.coef_[0],
        }
    ).sort_values("coefficient", ascending=False).to_csv(
        report_dir / "term_coefficients.csv", index=False
    )
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fit-source", action="append", required=True)
    parser.add_argument("--calibration-source", required=True)
    parser.add_argument("--threshold-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    args = parser.parse_args()

    data = validate_dataset(pd.read_csv(args.data))
    partitions = partition_by_source(
        data,
        fit_sources=args.fit_source,
        calibration_source=args.calibration_source,
        threshold_source=args.threshold_source,
        test_source=args.test_source,
    )
    model, metadata = train_prompt_injection_model(partitions)
    if args.dataset_manifest.exists():
        metadata["dataset_manifest"] = json.loads(
            args.dataset_manifest.read_text(encoding="utf-8")
        )
    save_model(model, metadata, args.artifact, args.metadata)
    save_evaluation_reports(model, metadata, partitions["test"], args.report_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
