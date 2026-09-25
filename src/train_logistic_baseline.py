"""Train and evaluate the VulnScope Logistic Regression baseline."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
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

from prepare_kev_labels import CANDIDATE_FEATURES, build_preprocessor


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports" / "logistic_baseline"
PREDICTIONS_DIR = ROOT / "data" / "predictions"

TRAIN_PATH = PROCESSED_DIR / "model_labeled_train_2020_2024.parquet"
TEST_PATH = PROCESSED_DIR / "model_labeled_test_2025.parquet"
LIVE_PATH = PROCESSED_DIR / "model_scoring_live_2026.parquet"

TARGET = "target_known_exploited"
MODEL_VERSION = "logistic_baseline_v1"
RANDOM_STATE = 42

for directory in [ARTIFACTS_DIR, REPORTS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def select_f1_threshold(y_true, probabilities):
    """Select the probability threshold with maximum validation F1."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.argmax(f1))

    return float(thresholds[best_index]), float(f1[best_index])


def classification_metrics(y_true, probabilities, threshold):
    """Return imbalance-aware classification and calibration metrics."""

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_pr_curve(validation_y, validation_probability, test_y, test_probability):
    """Save validation and test precision-recall curves."""

    figure, axis = plt.subplots(figsize=(8, 6))

    for name, y_true, probability in [
        ("2024 threshold validation", validation_y, validation_probability),
        ("2025 temporal test", test_y, test_probability),
    ]:
        precision, recall, _ = precision_recall_curve(
            y_true,
            probability,
        )
        score = average_precision_score(y_true, probability)
        axis.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={score:.4f})",
        )

    axis.axhline(
        np.mean(test_y),
        color="gray",
        linestyle="--",
        label=f"2025 prevalence={np.mean(test_y):.4f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Logistic Regression precision-recall curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        REPORTS_DIR / "precision_recall_curve.png",
        dpi=160,
    )
    plt.close(figure)


def save_calibration_plot(test_y, test_probability):
    """Save a reliability diagram for the untouched 2025 test set."""

    observed, predicted = calibration_curve(
        test_y,
        test_probability,
        n_bins=10,
        strategy="quantile",
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.plot(predicted, observed, marker="o")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive rate")
    axis.set_title("2025 Logistic Regression calibration")
    figure.tight_layout()
    figure.savefig(
        REPORTS_DIR / "calibration_curve.png",
        dpi=160,
    )
    plt.close(figure)


def save_confusion_matrix(test_y, test_probability, threshold):
    """Save the 2025 confusion matrix at the selected threshold."""

    matrix = confusion_matrix(
        test_y,
        test_probability >= threshold,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    axis.set_title("2025 temporal-test confusion matrix")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        dpi=160,
    )
    plt.close(figure)


def save_coefficients(base_pipeline):
    """Save coefficients from the uncalibrated interpretable base model."""

    preprocessor = base_pipeline.named_steps["preprocessor"]
    classifier = base_pipeline.named_steps["classifier"]

    coefficients = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "coefficient": classifier.coef_[0],
    })
    coefficients["odds_ratio"] = np.exp(
        coefficients["coefficient"].clip(-50, 50)
    )
    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )
    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )
    coefficients.to_csv(
        REPORTS_DIR / "coefficients.csv",
        index=False,
    )


def main():
    training_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)
    live_data = pd.read_parquet(LIVE_PATH)

    fit_data = training_data[
        training_data["publication_year"].between(2020, 2023)
    ].copy()
    validation_2024 = training_data[
        training_data["publication_year"] == 2024
    ].sort_values("date_published").copy()

    calibration_cutoff = pd.Timestamp(
        "2024-07-01",
        tz="UTC",
    )
    calibration_data = validation_2024[
        validation_2024["date_published"] < calibration_cutoff
    ].copy()
    threshold_data = validation_2024[
        validation_2024["date_published"] >= calibration_cutoff
    ].copy()

    for name, dataset in [
        ("fit", fit_data),
        ("calibration", calibration_data),
        ("threshold", threshold_data),
        ("test", test_data),
    ]:
        if dataset[TARGET].isna().any():
            raise ValueError(f"{name} target contains missing values")
        if dataset[TARGET].sum() == 0:
            raise ValueError(f"{name} contains no positive labels")

    base_pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    base_pipeline.fit(
        fit_data[CANDIDATE_FEATURES],
        fit_data[TARGET].astype(int),
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv="prefit",
    )
    calibrated_model.fit(
        calibration_data[CANDIDATE_FEATURES],
        calibration_data[TARGET].astype(int),
    )

    threshold_y = threshold_data[TARGET].astype(int).to_numpy()
    threshold_probability = calibrated_model.predict_proba(
        threshold_data[CANDIDATE_FEATURES]
    )[:, 1]
    threshold, validation_best_f1 = select_f1_threshold(
        threshold_y,
        threshold_probability,
    )

    test_y = test_data[TARGET].astype(int).to_numpy()
    test_probability = calibrated_model.predict_proba(
        test_data[CANDIDATE_FEATURES]
    )[:, 1]

    validation_metrics = classification_metrics(
        threshold_y,
        threshold_probability,
        threshold,
    )
    test_metrics = classification_metrics(
        test_y,
        test_probability,
        threshold,
    )

    joblib.dump(
        calibrated_model,
        ARTIFACTS_DIR / "logistic_regression_pipeline.joblib",
    )
    save_coefficients(base_pipeline)
    save_pr_curve(
        threshold_y,
        threshold_probability,
        test_y,
        test_probability,
    )
    save_calibration_plot(test_y, test_probability)
    save_confusion_matrix(test_y, test_probability, threshold)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "calibrated_logistic_regression",
        "target": TARGET,
        "label_definition": "CISA KEV membership by 2026-09-24 catalog cutoff",
        "feature_time_policy": "current_snapshot_proxy_not_point_in_time_verified",
        "fit_period": "2020-01-01 through 2023-12-31",
        "calibration_period": "2024-01-01 through 2024-06-30",
        "threshold_period": "2024-07-01 through 2024-12-31",
        "test_period": "2025-01-01 through 2025-12-31",
        "threshold_selection": "maximum F1 on threshold period",
        "selected_threshold": threshold,
        "validation_best_f1": validation_best_f1,
        "candidate_features": CANDIDATE_FEATURES,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    with (
        ARTIFACTS_DIR / "logistic_regression_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    live_probability = calibrated_model.predict_proba(
        live_data[CANDIDATE_FEATURES]
    )[:, 1]
    live_predictions = live_data[
        [
            "cve_id",
            "date_published",
            "is_kev_by_catalog_cutoff",
        ]
    ].copy()
    live_predictions["model_version"] = MODEL_VERSION
    live_predictions["exploitation_probability"] = live_probability
    live_predictions["flagged"] = live_probability >= threshold
    live_predictions["decision_threshold"] = threshold
    live_predictions.to_parquet(
        PREDICTIONS_DIR / "logistic_live_2026.parquet",
        index=False,
        compression="zstd",
    )

    print("VulnScope Logistic Regression baseline complete")
    print("-----------------------------------------------")
    print(f"Fit rows: {len(fit_data):,}")
    print(f"Calibration rows: {len(calibration_data):,}")
    print(f"Threshold-selection rows: {len(threshold_data):,}")
    print(f"2025 test rows: {len(test_data):,}")
    print(f"Selected threshold: {threshold:.6f}")
    print("\n2024 threshold-validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print("\n2025 temporal-test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(
        f"\n2026 flagged CVEs: "
        f"{int(live_predictions['flagged'].sum()):,}"
    )


if __name__ == "__main__":
    main()
