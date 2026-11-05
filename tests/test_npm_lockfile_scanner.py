"""Offline tests for npm package-lock parsing and OSV result mapping."""

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npm_lockfile_scanner import (  # noqa: E402
    LockfileError,
    parse_package_lock,
    scan_package_lock,
)


@pytest.fixture
def lockfile():
    return {
        "name": "demo-project",
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "demo-project",
                "dependencies": {"direct-package": "^1.0.0"},
            },
            "node_modules/direct-package": {"version": "1.2.3"},
            "node_modules/direct-package/node_modules/nested-package": {
                "version": "2.0.0"
            },
            "node_modules/local-link": {"resolved": "../local", "link": True},
        },
    }


def test_parse_lockfile_extracts_exact_direct_and_transitive_versions(lockfile):
    result = parse_package_lock(json.dumps(lockfile).encode())
    assert result["project_name"] == "demo-project"
    assert result["dependency_count"] == 2
    direct, nested = result["dependencies"]
    assert direct["package"] == "direct-package"
    assert direct["version"] == "1.2.3"
    assert direct["relationship"] == "direct"
    assert nested["package"] == "nested-package"
    assert nested["relationship"] == "transitive"


def test_unsupported_lockfile_version_is_rejected():
    with pytest.raises(LockfileError, match="versions 2 and 3"):
        parse_package_lock({"lockfileVersion": 1, "dependencies": {}})


def test_scan_maps_vulnerabilities_back_to_installed_package(lockfile):
    def fake_batch(dependencies):
        output = []
        for dependency in dependencies:
            vulnerabilities = (
                [{"id": "GHSA-test", "reference": "https://osv.dev/GHSA-test"}]
                if dependency["package"] == "nested-package"
                else []
            )
            output.append(
                {
                    **dependency,
                    "vulnerability_count": len(vulnerabilities),
                    "vulnerabilities": vulnerabilities,
                }
            )
        return output

    result = scan_package_lock(lockfile, batch_query=fake_batch)
    assert result["scanned_count"] == 2
    assert result["vulnerable_package_count"] == 1
    assert result["unique_vulnerability_ids"] == ["GHSA-test"]
    assert result["vulnerable_packages"][0]["relationship"] == "transitive"
