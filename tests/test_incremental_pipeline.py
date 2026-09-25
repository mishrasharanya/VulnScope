"""Unit tests for incremental CVE merging and live rescoring."""

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_changed_cves import merge_predictions  # noqa: E402
from update_cve_data import UpdateError, merge_changed_records  # noqa: E402


def test_changed_cve_replaces_existing_row():
    canonical = pd.DataFrame(
        [{"cve_id": "CVE-2026-1000", "description": "old", "score": 1}]
    )
    changed = pd.DataFrame(
        [{"cve_id": "CVE-2026-1000", "description": "new", "score": 2}]
    )
    result = merge_changed_records(
        canonical, changed, ["CVE-2026-1000"]
    )
    assert len(result) == 1
    assert result.iloc[0]["description"] == "new"


def test_rejected_or_deleted_cve_is_removed():
    canonical = pd.DataFrame(
        [
            {"cve_id": "CVE-2026-1000", "description": "remove"},
            {"cve_id": "CVE-2026-1001", "description": "keep"},
        ]
    )
    result = merge_changed_records(
        canonical, pd.DataFrame(), ["CVE-2026-1000"]
    )
    assert result["cve_id"].tolist() == ["CVE-2026-1001"]


def test_schema_drift_stops_ingestion():
    canonical = pd.DataFrame([{"cve_id": "CVE-2026-1000", "value": 1}])
    changed = pd.DataFrame([{"cve_id": "CVE-2026-1000", "other": 1}])
    with pytest.raises(UpdateError, match="Schema mismatch"):
        merge_changed_records(canonical, changed, ["CVE-2026-1000"])


def test_changed_prediction_replaces_old_score():
    existing = pd.DataFrame(
        [
            {
                "cve_id": "CVE-2026-1000",
                "date_published": pd.Timestamp("2026-01-01", tz="UTC"),
                "priority_probability": 0.2,
            }
        ]
    )
    replacement = pd.DataFrame(
        [
            {
                "cve_id": "CVE-2026-1000",
                "date_published": pd.Timestamp("2026-01-01", tz="UTC"),
                "priority_probability": 0.8,
                "scored_at": "now",
            }
        ]
    )
    result = merge_predictions(
        existing, replacement, ["CVE-2026-1000"]
    )
    assert len(result) == 1
    assert result.iloc[0]["priority_probability"] == 0.8
    assert result.iloc[0]["scored_at"] == "now"
