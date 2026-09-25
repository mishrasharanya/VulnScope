"""Incrementally merge changed cvelistV5 records into canonical Parquet."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "raw" / "cvelistV5"
CANONICAL_PATH = ROOT / "data" / "processed" / "cves_clean.parquet"
STATE_PATH = ROOT / "data" / "state" / "ingestion_state.json"
REPORT_PATH = ROOT / "reports" / "ingestion" / "latest_update.json"
PARSER_PATH = ROOT / "src" / "profile_cve_data.py"


class UpdateError(RuntimeError):
    """Raised when an incremental update cannot be completed safely."""


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_revision() -> str:
    return run_git("rev-parse", "HEAD")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def changed_cve_paths(old_revision: str, new_revision: str) -> list[Path]:
    if old_revision == new_revision:
        return []
    output = run_git(
        "diff",
        "--name-only",
        "--diff-filter=ACMRD",
        old_revision,
        new_revision,
        "--",
        "cves",
    )
    paths = []
    for relative in output.splitlines():
        path = REPO / relative
        if path.name.startswith("CVE-") and path.suffix == ".json":
            paths.append(path)
    return sorted(paths)


def cve_id_from_path(path: Path) -> str:
    return path.stem.upper()


def parse_changed_files(paths: list[Path]) -> pd.DataFrame:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    with tempfile.TemporaryDirectory(prefix="vulnscope-update-") as directory:
        temporary_root = Path(directory)
        raw_dir = temporary_root / "cves"
        output_dir = temporary_root / "output"
        raw_dir.mkdir()
        for path in existing:
            shutil.copy2(path, raw_dir / path.name)
        environment = os.environ.copy()
        environment.update(
            {
                "VULNSCOPE_RAW_DIR": str(raw_dir),
                "VULNSCOPE_OUTPUT_DIR": str(output_dir),
                "VULNSCOPE_OUTPUT_NAME": "changed.parquet",
            }
        )
        result = subprocess.run(
            [sys.executable, str(PARSER_PATH)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise UpdateError(
                "Changed-record parsing failed; canonical data was not modified.\n"
                + result.stderr[-2000:]
            )
        parsed_path = output_dir / "changed.parquet"
        return pd.read_parquet(parsed_path) if parsed_path.exists() else pd.DataFrame()


def merge_changed_records(
    canonical: pd.DataFrame,
    changed: pd.DataFrame,
    changed_ids: list[str],
) -> pd.DataFrame:
    retained = canonical[~canonical["cve_id"].isin(changed_ids)].copy()
    if not changed.empty:
        missing_columns = set(canonical.columns) - set(changed.columns)
        extra_columns = set(changed.columns) - set(canonical.columns)
        if missing_columns or extra_columns:
            raise UpdateError(
                f"Schema mismatch: missing={sorted(missing_columns)}, "
                f"extra={sorted(extra_columns)}"
            )
        changed = changed[canonical.columns]
        merged = pd.concat([retained, changed], ignore_index=True)
    else:
        merged = retained
    if merged["cve_id"].duplicated().any():
        duplicates = merged.loc[merged["cve_id"].duplicated(), "cve_id"].tolist()
        raise UpdateError(f"Duplicate CVE IDs after merge: {duplicates[:10]}")
    return merged.sort_values("cve_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Fetch and fast-forward the local cvelistV5 checkout before updating.",
    )
    parser.add_argument(
        "--initialize-state",
        action="store_true",
        help="Record the current checkout as the canonical-data baseline.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not REPO.exists() or not CANONICAL_PATH.exists():
        raise UpdateError("The cvelistV5 checkout and canonical Parquet are required.")
    if args.sync:
        run_git("pull", "--ff-only")

    revision = current_revision()
    state = load_state()
    if args.initialize_state:
        payload = {
            "source_revision": revision,
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "canonical_rows": int(len(pd.read_parquet(CANONICAL_PATH, columns=["cve_id"]))),
        }
        if not args.dry_run:
            atomic_json_write(STATE_PATH, payload)
        print(json.dumps({"action": "initialize", **payload}, indent=2))
        return

    old_revision = state.get("source_revision")
    if not old_revision:
        raise UpdateError(
            "No ingestion baseline exists. Run with --initialize-state once."
        )
    changed_paths = changed_cve_paths(old_revision, revision)
    changed_ids = sorted({cve_id_from_path(path) for path in changed_paths})
    preview = {
        "from_revision": old_revision,
        "to_revision": revision,
        "changed_files": len(changed_paths),
        "changed_cve_ids": changed_ids,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if not changed_paths:
        report = {
            **preview,
            "published_rows_parsed": 0,
            "rows_before": int(
                len(pd.read_parquet(CANONICAL_PATH, columns=["cve_id"]))
            ),
            "rows_after": int(
                len(pd.read_parquet(CANONICAL_PATH, columns=["cve_id"]))
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(REPORT_PATH, report)
        print(json.dumps(report, indent=2))
        return

    canonical = pd.read_parquet(CANONICAL_PATH)
    changed = parse_changed_files(changed_paths)
    merged = merge_changed_records(canonical, changed, changed_ids)
    now = datetime.now(timezone.utc).isoformat()
    report = {
        **preview,
        "dry_run": False,
        "published_rows_parsed": int(len(changed)),
        "rows_before": int(len(canonical)),
        "rows_after": int(len(merged)),
        "completed_at": now,
    }
    atomic_parquet_write(merged, CANONICAL_PATH)
    atomic_json_write(REPORT_PATH, report)
    atomic_json_write(
        STATE_PATH,
        {
            "source_revision": revision,
            "last_successful_update": now,
            "canonical_rows": int(len(merged)),
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
