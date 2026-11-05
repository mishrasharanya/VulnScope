"""Tests for deterministic Phase 4B dataset preparation."""

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_prompt_injection_data import prepare_dataset  # noqa: E402


def write_fixture(path, prefix, rows):
    pd.DataFrame(
        [
            {"text": f"{prefix} benign {index}", "label": 0}
            for index in range(rows)
        ]
        + [
            {"text": f"{prefix} injection {index}", "label": 1}
            for index in range(rows)
        ]
    ).to_parquet(path, index=False)


def test_preparation_preserves_external_test_and_has_no_overlap(tmp_path):
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    external = tmp_path / "external.parquet"
    write_fixture(train, "train", 4)
    write_fixture(validation, "validation", 4)
    write_fixture(external, "external", 4)

    data, manifest = prepare_dataset(
        {
            "neuralchemy_train": train,
            "neuralchemy_validation": validation,
            "agentic_external_test": external,
        }
    )
    assert len(data) == 24
    assert manifest["rows_by_source"]["agentic5k_external_test"] == 8
    assert not any(manifest["cross_partition_overlaps"].values())
