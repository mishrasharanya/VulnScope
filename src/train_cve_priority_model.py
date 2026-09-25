"""Train the CVE-only technical-priority model.

This model predicts whether a CVE has a CVSS base score of at least 9.0 from
its title and description.  It is a severity-prioritization model, not an
exploitation model.  CVSS values and vector components are used only to make
the historical label and are never supplied as predictors.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "processed" / "cves_clean.parquet"
ARTIFACT_PATH = ROOT / "artifacts" / "cve_priority_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "cve_priority_model_metadata.json"
REPORT_DIR = ROOT / "reports" / "cve_priority_model"
PREDICTION_PATH = (
    ROOT / "data" / "predictions" / "cve_priority_live_2026.parquet"
)

TARGET = "target_high_technical_priority"
MODEL_VERSION = "cve_priority_tfidf_svm_v2"
RANDOM_STATE = 42
CVSS_PRIORITY_CUTOFF = 9.0
CALIBRATION_CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")

LABEL_ONLY_FIELDS = [
    "cvss_score",
    "cvss_version",
    "severity",
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
]
QUARANTINED_FIELDS = [
    "has_cisa_adp",
    "ssvc_exploitation",
    "ssvc_automatable",
    "ssvc_technical_impact",
    "ssvc_timestamp",
    "ref_exploit",
]


def combine_text(frame):
    """Combine only the two documented predictor fields."""
    return (
        frame["title"].fillna("").astype(str)
        + " "
        + frame["description"].fillna("").astype(str)
    ).str.strip()


def select_f1_threshold(y_true, probabilities):
    """Select the maximum-F1 threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions)),
        "recall": float(recall_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_reports(
    base_model,
    threshold_y,
    threshold_probability,
    test_y,
    test_probability,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = base_model.named_steps["tfidf"]
    classifier = base_model.named_steps["classifier"]
    pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out(),
            "coefficient": classifier.coef_[0],
        }
    ).sort_values("coefficient", ascending=False).to_csv(
        REPORT_DIR / "term_coefficients.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    for label, y_true, probability in [
        ("2024 threshold set", threshold_y, threshold_probability),
        ("2025 temporal test", test_y, test_probability),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, probability)
        pr_auc = average_precision_score(y_true, probability)
        axis.plot(recall, precision, label=f"{label} (PR-AUC={pr_auc:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("CVE-only priority model precision-recall curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "precision_recall_curve.png", dpi=160)
    plt.close(figure)

    observed, predicted = calibration_curve(
        test_y, test_probability, n_bins=10, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.plot(predicted, observed, marker="o")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive rate")
    axis.set_title("2025 CVE-priority calibration")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "calibration_curve.png", dpi=160)
    plt.close(figure)


def main():
    columns = [
        "cve_id",
        "publication_year",
        "date_published",
        "title",
        "description",
        "cvss_score",
    ]
    data = pd.read_parquet(SOURCE_PATH, columns=columns)
    data = data[data["publication_year"].between(2020, 2026)].copy()
    data["model_text"] = combine_text(data)

    historical = data[
        data["publication_year"].between(2020, 2025)
        & data["cvss_score"].notna()
    ].copy()
    historical[TARGET] = (
        historical["cvss_score"] >= CVSS_PRIORITY_CUTOFF
    ).astype(int)

    fit = historical[historical["publication_year"] <= 2023]
    validation = historical[historical["publication_year"] == 2024]
    calibration = validation[
        validation["date_published"] < CALIBRATION_CUTOFF
    ]
    threshold_data = validation[
        validation["date_published"] >= CALIBRATION_CUTOFF
    ]
    test = historical[historical["publication_year"] == 2025]
    live = data[data["publication_year"] == 2026].copy()

    if min(map(len, [fit, calibration, threshold_data, test, live])) == 0:
        raise ValueError("At least one temporal partition is empty")

    base_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=3,
                    max_df=0.98,
                    max_features=100_000,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=0.1,
                    class_weight="balanced",
                    dual=True,
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    base_model.fit(fit["model_text"], fit[TARGET])

    model = CalibratedClassifierCV(
        estimator=base_model, method="sigmoid", cv="prefit"
    )
    model.fit(calibration["model_text"], calibration[TARGET])

    threshold_y = threshold_data[TARGET].to_numpy()
    threshold_probability = model.predict_proba(
        threshold_data["model_text"]
    )[:, 1]
    threshold, threshold_best_f1 = select_f1_threshold(
        threshold_y, threshold_probability
    )

    test_y = test[TARGET].to_numpy()
    test_probability = model.predict_proba(test["model_text"])[:, 1]
    threshold_metrics = metrics(
        threshold_y, threshold_probability, threshold
    )
    test_metrics = metrics(test_y, test_probability, threshold)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    save_reports(
        base_model,
        threshold_y,
        threshold_probability,
        test_y,
        test_probability,
    )

    live_probability = model.predict_proba(live["model_text"])[:, 1]
    predictions = live[["cve_id", "date_published"]].copy()
    predictions["model_version"] = MODEL_VERSION
    predictions["priority_probability"] = live_probability
    predictions["flagged"] = live_probability >= threshold
    predictions["decision_threshold"] = threshold
    predictions.to_parquet(PREDICTION_PATH, index=False, compression="zstd")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "calibrated_tfidf_linear_svm",
        "target": TARGET,
        "label_definition": "CVSS base score >= 9.0",
        "intended_interpretation": (
            "Probability of a high technical-priority CVSS profile; "
            "not probability of exploitation"
        ),
        "predictor_fields": ["title", "description"],
        "label_only_fields": LABEL_ONLY_FIELDS,
        "quarantined_fields": QUARANTINED_FIELDS,
        "data_limitation": (
            "The canonical table is a current snapshot, so title and "
            "description are not yet reconstructed point-in-time values."
        ),
        "fit_period": "2020-01-01 through 2023-12-31",
        "calibration_period": "2024-01-01 through 2024-06-30",
        "threshold_period": "2024-07-01 through 2024-12-31",
        "test_period": "2025-01-01 through 2025-12-31",
        "cvss_priority_cutoff": CVSS_PRIORITY_CUTOFF,
        "threshold_policy": "maximize F1 on the 2024 threshold set",
        "selected_threshold": threshold,
        "threshold_best_f1": threshold_best_f1,
        "threshold_metrics": threshold_metrics,
        "test_metrics": test_metrics,
        "live_rows": int(len(live)),
        "live_flagged": int(predictions["flagged"].sum()),
    }
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    pd.DataFrame(
        [
            {
                "field_group": "predictors",
                "fields": "title; description",
                "decision": "included",
                "reason": "CVE-only narrative available for prioritization",
            },
            {
                "field_group": "CVSS",
                "fields": "; ".join(LABEL_ONLY_FIELDS),
                "decision": "label_only",
                "reason": "Including target-defining fields would leak the label",
            },
            {
                "field_group": "post-publication_or_exploitation",
                "fields": "; ".join(QUARANTINED_FIELDS),
                "decision": "quarantined",
                "reason": "Could reveal later exploitation or enrichment evidence",
            },
        ]
    ).to_csv(REPORT_DIR / "feature_policy.csv", index=False)

    print("VulnScope CVE-only priority model complete")
    print("------------------------------------------")
    print(json.dumps(test_metrics, indent=2))
    print(f"2026 CVEs scored: {len(predictions):,}")
    print(f"2026 CVEs flagged: {int(predictions['flagged'].sum()):,}")


if __name__ == "__main__":
    main()
