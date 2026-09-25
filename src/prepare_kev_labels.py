"""Prepare temporally labeled VulnScope datasets from CISA KEV.

This script deliberately stops before model fitting. It creates the target,
checks label maturity, documents feature leakage risk, defines the reusable
preprocessor, and measures class imbalance. The current cvelistV5 checkout is
a one-commit snapshot, so current-record features are marked as historical-
snapshot proxies rather than claimed to be publication-time safe.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

CVE_PATH = PROCESSED_DIR / "cves_clean.parquet"
KEV_PATH = DATA_DIR / "raw" / "cisa" / "known_exploited_vulnerabilities.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "cvss_score",
    "cwe_count",
    "vendor_count",
    "product_count",
    "affected_version_count",
]

CATEGORICAL_FEATURES = [
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
    "primary_cwe",
    "primary_vendor",
]

CANDIDATE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_kev(path):
    """Load and normalize the official CISA KEV JSON feed."""

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    kev = pd.DataFrame(payload["vulnerabilities"]).rename(
        columns={
            "cveID": "cve_id",
            "dateAdded": "kev_date_added",
        }
    )

    kev["kev_date_added"] = pd.to_datetime(
        kev["kev_date_added"],
        errors="raise",
        utc=True,
    )

    if kev["cve_id"].duplicated().any():
        duplicates = kev.loc[
            kev["cve_id"].duplicated(keep=False),
            "cve_id",
        ].tolist()
        raise ValueError(f"Duplicate KEV CVE IDs: {duplicates[:10]}")

    released_at = pd.to_datetime(
        payload["dateReleased"],
        errors="raise",
        utc=True,
    )

    metadata = {
        "catalog_title": payload["title"],
        "catalog_version": payload["catalogVersion"],
        "catalog_released_at": released_at,
        "catalog_count": payload["count"],
    }

    return kev, metadata


def add_kev_labels(cves, kev, observation_cutoff):
    """Create the fixed-cutoff KEV target and observation metadata.

    KEV launched after the start of the training period, so dateAdded cannot
    serve as a consistent exploitation date or fixed-horizon target for 2020.
    The primary retrospective target is therefore catalog membership at one
    explicit cutoff. The observation age is retained so right-censoring stays
    visible, and 2026 is excluded from supervised labels.
    """

    kev_columns = [
        "cve_id",
        "kev_date_added",
        "vendorProject",
        "product",
        "vulnerabilityName",
        "knownRansomwareCampaignUse",
    ]

    labeled = cves.merge(
        kev[kev_columns],
        how="left",
        on="cve_id",
        validate="one_to_one",
    )

    labeled["label_cutoff"] = observation_cutoff
    labeled["label_observation_days"] = (
        observation_cutoff
        - labeled["date_published"]
    ).dt.total_seconds().div(86400)

    labeled["is_kev_by_catalog_cutoff"] = (
        labeled["kev_date_added"].notna()
        & (labeled["kev_date_added"] <= observation_cutoff)
    ).astype("int8")

    labeled["use_for_supervised_label"] = (
        labeled["publication_year"] <= 2025
    )

    target = labeled[
        "is_kev_by_catalog_cutoff"
    ].astype("Int8")
    target.loc[
        ~labeled["use_for_supervised_label"]
    ] = pd.NA
    labeled["target_known_exploited"] = target

    # Diagnostic only. KEV dateAdded is catalog-entry time, not exploitation
    # time, and the catalog did not exist during 2020. Do not model this field.
    labeled["kev_added_within_180d"] = (
        labeled["kev_date_added"].notna()
        & (
            labeled["kev_date_added"]
            <= labeled["date_published"] + pd.Timedelta(days=180)
        )
    )

    return labeled


def build_preprocessor():
    """Return preprocessing to fit on training data inside a model pipeline."""

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="MISSING",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])


def feature_audit():
    """Document feature roles and point-in-time leakage decisions."""

    rows = []

    def add(features, status, reason):
        for feature in features:
            rows.append({
                "feature": feature,
                "status": status,
                "reason": reason,
            })

    add(
        ["cve_id", "date_published", "publication_year"],
        "split_or_identifier_only",
        "Used for joins and temporal splitting, not as predictive inputs.",
    )
    add(
        CANDIDATE_FEATURES,
        "candidate_snapshot_proxy",
        "Useful baseline candidate, but the one-commit CVE checkout cannot prove the current value existed at prediction time.",
    )
    add(
        [
            "tag_remote",
            "tag_code_execution",
            "tag_denial_of_service",
            "tag_authenticated",
            "tag_unauthenticated",
            "tag_sql_injection",
            "tag_xss",
            "tag_buffer_overflow",
            "tag_privilege_escalation",
            "tag_use_after_free",
            "tag_path_traversal",
            "tag_command_injection",
            "tag_information_disclosure",
            "tag_memory_corruption",
        ],
        "quarantine_until_snapshot_history",
        "Derived from the current description, which may have changed after publication.",
    )
    add(
        [
            "reference_count",
            "ref_vendor_advisory",
            "ref_third_party_advisory",
            "ref_patch",
            "ref_exploit",
            "ref_vdb",
            "ref_issue_tracking",
            "ref_mailing_list",
            "ref_government",
            "ref_release_notes",
            "ref_mitigation",
            "ref_technical_description",
        ],
        "quarantine_until_snapshot_history",
        "References can be added later; ref_exploit may directly reveal post-publication exploit evidence.",
    )
    add(
        [
            "date_updated",
            "days_published_to_updated",
            "cvss_provider_updated",
            "has_cisa_adp",
            "ssvc_exploitation",
            "ssvc_automatable",
            "ssvc_technical_impact",
            "ssvc_timestamp",
            "has_solution",
            "solution",
        ],
        "exclude_from_model",
        "Post-publication or outcome-adjacent information creates direct leakage risk.",
    )

    return pd.DataFrame(rows)


def save_split(dataset, filename):
    """Save identifiers, candidate features, label metadata, and target."""

    output_columns = [
        "cve_id",
        "date_published",
        "publication_year",
        *CANDIDATE_FEATURES,
        "kev_date_added",
        "label_cutoff",
        "label_observation_days",
        "use_for_supervised_label",
        "is_kev_by_catalog_cutoff",
        "target_known_exploited",
        "kev_added_within_180d",
    ]

    dataset[output_columns].to_parquet(
        PROCESSED_DIR / filename,
        index=False,
        compression="zstd",
    )


def main():
    cves = pd.read_parquet(CVE_PATH)
    kev, metadata = load_kev(KEV_PATH)

    labeled = add_kev_labels(
        cves,
        kev,
        metadata["catalog_released_at"],
    )

    train = labeled[
        labeled["publication_year"].between(2020, 2024)
    ].copy()
    test = labeled[
        labeled["publication_year"] == 2025
    ].copy()
    live = labeled[
        labeled["publication_year"] == 2026
    ].copy()

    if train["target_known_exploited"].isna().any():
        raise ValueError("Training targets contain missing values.")
    if test["target_known_exploited"].isna().any():
        raise ValueError("2025 test targets contain missing values.")
    if live["target_known_exploited"].notna().any():
        raise ValueError("2026 must remain unlabeled for live scoring.")

    save_split(
        train,
        "model_labeled_train_2020_2024.parquet",
    )
    save_split(
        test,
        "model_labeled_test_2025.parquet",
    )
    save_split(
        live,
        "model_scoring_live_2026.parquet",
    )

    audit = feature_audit()
    audit.to_csv(
        REPORTS_DIR / "feature_leakage_audit.csv",
        index=False,
    )

    supervised = labeled[
        labeled["use_for_supervised_label"]
        & labeled["publication_year"].between(2020, 2025)
    ]
    imbalance = (
        supervised.groupby("publication_year")
        .agg(
            total_cves=("cve_id", "size"),
            kev_positives=("target_known_exploited", "sum"),
            positive_rate=("target_known_exploited", "mean"),
            median_observation_days=("label_observation_days", "median"),
        )
        .reset_index()
    )
    imbalance["positive_rate_pct"] = (
        imbalance["positive_rate"] * 100
    )
    imbalance.to_csv(
        REPORTS_DIR / "class_imbalance_by_year.csv",
        index=False,
    )

    # Construction is validated here; fitting belongs inside each model
    # pipeline so the preprocessor learns only from that model's fit split.
    build_preprocessor()

    print("VulnScope KEV model preparation complete")
    print("----------------------------------------")
    print(
        f"KEV catalog: {metadata['catalog_version']} "
        f"({metadata['catalog_count']:,} entries)"
    )
    print(
        "Primary target: KEV membership by fixed catalog cutoff "
        f"{metadata['catalog_released_at']}"
    )
    print(
        f"Training: {len(train):,} CVEs, "
        f"{int(train['target_known_exploited'].sum()):,} positives"
    )
    print(
        f"Test: {len(test):,} CVEs, "
        f"{int(test['target_known_exploited'].sum()):,} positives"
    )
    print(
        f"Live: {len(live):,} CVEs, "
        "target intentionally unset"
    )
    print("\nClass imbalance by year:")
    print(
        imbalance[
            [
                "publication_year",
                "total_cves",
                "kev_positives",
                "positive_rate_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
