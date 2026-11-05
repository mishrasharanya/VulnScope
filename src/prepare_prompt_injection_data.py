"""Download pinned public datasets and build the Phase 4B modeling table."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prompt_injection"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "prompt_injection_modeling.csv"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

NEURALCHEMY_REVISION = "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75"
AGENTIC_REVISION = "e658d94dcafc133e3a4483f9f9aa13507f3848c4"

SOURCES = {
    "neuralchemy_train": {
        "url": (
            "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/"
            f"resolve/{NEURALCHEMY_REVISION}/core/train-00000-of-00001.parquet"
        ),
        "license": "Apache-2.0",
        "purpose": "fit",
    },
    "neuralchemy_validation": {
        "url": (
            "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/"
            f"resolve/{NEURALCHEMY_REVISION}/core/validation-00000-of-00001.parquet"
        ),
        "license": "Apache-2.0",
        "purpose": "calibration_and_threshold",
    },
    "agentic_external_test": {
        "url": (
            "https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k/"
            f"resolve/{AGENTIC_REVISION}/data/test.parquet"
        ),
        "license": "CC-BY-4.0",
        "purpose": "external_test",
    },
}


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()


def download_sources() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, details in SOURCES.items():
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            urlretrieve(details["url"], path)
        paths[name] = path
    return paths


def prepare_dataset(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    train = pd.read_parquet(paths["neuralchemy_train"])
    validation = pd.read_parquet(paths["neuralchemy_validation"])
    external = pd.read_parquet(paths["agentic_external_test"])

    calibration, threshold = train_test_split(
        validation,
        test_size=0.5,
        random_state=42,
        stratify=validation["label"],
    )
    frames = []
    for frame, source, original_dataset, license_name in [
        (train, "neuralchemy_core_train", "neuralchemy/Prompt-injection-dataset", "Apache-2.0"),
        (calibration, "neuralchemy_core_calibration", "neuralchemy/Prompt-injection-dataset", "Apache-2.0"),
        (threshold, "neuralchemy_core_threshold", "neuralchemy/Prompt-injection-dataset", "Apache-2.0"),
        (external, "agentic5k_external_test", "3nesdeniz/agentic-prompt-injection-5k", "CC-BY-4.0"),
    ]:
        selected = frame[["text", "label"]].copy()
        selected["source"] = source
        selected["original_dataset"] = original_dataset
        selected["license"] = license_name
        frames.append(selected)

    combined = pd.concat(frames, ignore_index=True)
    combined["_normalized"] = normalized_text(combined["text"])
    conflicting = combined.groupby("_normalized")["label"].nunique()
    if conflicting.gt(1).any():
        raise ValueError("Cross-source prompts contain conflicting labels.")
    duplicate_rows = int(combined["_normalized"].duplicated().sum())
    combined = combined.drop_duplicates("_normalized").drop(columns="_normalized")

    source_sets = {
        name: set(normalized_text(group["text"]))
        for name, group in combined.groupby("source")
    }
    source_names = list(source_sets)
    overlaps = {}
    for index, left in enumerate(source_names):
        for right in source_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(source_sets[left] & source_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"Normalized prompts overlap across partitions: {overlaps}")

    manifest = {
        "dataset_contract": {"text": "string", "label": "0 benign / 1 injection", "source": "partition provenance"},
        "pinned_revisions": {
            "neuralchemy/Prompt-injection-dataset": NEURALCHEMY_REVISION,
            "3nesdeniz/agentic-prompt-injection-5k": AGENTIC_REVISION,
        },
        "licenses": {
            "neuralchemy/Prompt-injection-dataset": "Apache-2.0",
            "3nesdeniz/agentic-prompt-injection-5k": "CC-BY-4.0",
        },
        "rows": int(len(combined)),
        "rows_by_source": combined["source"].value_counts().to_dict(),
        "class_counts_by_source": {
            source: group["label"].value_counts().sort_index().to_dict()
            for source, group in combined.groupby("source")
        },
        "normalized_duplicates_removed": duplicate_rows,
        "cross_partition_overlaps": overlaps,
        "limitations": [
            "Calibration and threshold partitions come from one upstream validation split.",
            "The external test source is synthetic and may not represent production traffic.",
            "The resulting model detects textual patterns, not successful compromise.",
        ],
    }
    return combined, manifest


def main() -> None:
    paths = download_sources()
    data, manifest = prepare_dataset(paths)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
