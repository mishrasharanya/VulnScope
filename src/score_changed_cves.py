"""Update live predictions only for CVEs changed by incremental ingestion."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "data" / "processed" / "cves_clean.parquet"
PREDICTION_PATH = ROOT / "data" / "predictions" / "cve_priority_live_2026.parquet"
MODEL_PATH = ROOT / "artifacts" / "cve_priority_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "cve_priority_model_metadata.json"
UPDATE_REPORT_PATH = ROOT / "reports" / "ingestion" / "latest_update.json"
SCORING_REPORT_PATH = ROOT / "reports" / "ingestion" / "latest_scoring.json"


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def merge_predictions(
    existing: pd.DataFrame,
    replacements: pd.DataFrame,
    changed_ids: list[str],
) -> pd.DataFrame:
    retained = existing[~existing["cve_id"].isin(changed_ids)].copy()
    for column in replacements.columns:
        if column not in retained.columns:
            retained[column] = pd.NA
    for column in retained.columns:
        if column not in replacements.columns:
            replacements[column] = pd.NA
    merged = pd.concat(
        [retained, replacements[retained.columns]], ignore_index=True
    )
    if merged["cve_id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs after incremental scoring.")
    return merged.sort_values("date_published", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not UPDATE_REPORT_PATH.exists():
        raise FileNotFoundError("Run incremental ingestion before scoring.")

    update = json.loads(UPDATE_REPORT_PATH.read_text(encoding="utf-8"))
    changed_ids = update.get("changed_cve_ids", [])
    if not changed_ids:
        print(json.dumps({"changed_cves": 0, "scored": 0}, indent=2))
        return

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=["cve_id", "publication_year", "date_published", "title", "description"],
    )
    changed = canonical[
        canonical["cve_id"].isin(changed_ids)
        & canonical["publication_year"].eq(2026)
    ].copy()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    replacements = pd.DataFrame(
        columns=[
            "cve_id", "date_published", "model_version",
            "priority_probability", "flagged", "decision_threshold",
            "scored_at", "source_revision",
        ]
    )
    if not changed.empty:
        model = joblib.load(MODEL_PATH)
        text = (
            changed["title"].fillna("")
            + " "
            + changed["description"].fillna("")
        ).str.strip()
        probability = model.predict_proba(text)[:, 1]
        threshold = float(metadata["selected_threshold"])
        replacements = changed[["cve_id", "date_published"]].copy()
        replacements["model_version"] = metadata["model_version"]
        replacements["priority_probability"] = probability
        replacements["flagged"] = probability >= threshold
        replacements["decision_threshold"] = threshold
        replacements["scored_at"] = now
        replacements["source_revision"] = update["to_revision"]

    report = {
        "source_revision": update["to_revision"],
        "changed_cves": len(changed_ids),
        "live_2026_scored": int(len(replacements)),
        "live_2026_flagged": int(replacements["flagged"].sum()) if len(replacements) else 0,
        "model_version": metadata["model_version"],
        "completed_at": now,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    existing = pd.read_parquet(PREDICTION_PATH)
    merged = merge_predictions(existing, replacements, changed_ids)
    atomic_parquet_write(merged, PREDICTION_PATH)
    atomic_json_write(SCORING_REPORT_PATH, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
