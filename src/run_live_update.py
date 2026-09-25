"""Run the complete incremental sync and scoring workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "src" / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    run("update_cve_data.py", "--sync")
    run("score_changed_cves.py")
    print("VulnScope live data and changed-CVE scores are current.")


if __name__ == "__main__":
    main()
