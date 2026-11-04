"""Offline tests for OSV dependency lookup and response normalization."""

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osv_service import OSVError, query_osv, validate_dependency  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_version_ranges_are_rejected():
    with pytest.raises(OSVError, match="exact installed version"):
        validate_dependency("PyPI", "pandas", ">=2.0")


def test_unsupported_ecosystem_is_rejected():
    with pytest.raises(OSVError, match="supported"):
        validate_dependency("Unknown", "example", "1.0.0")


def test_query_payload_and_empty_result():
    captured = {}

    def opener(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({})

    result = query_osv("PyPI", "example", "1.2.3", opener=opener)
    assert captured["payload"] == {
        "package": {"ecosystem": "PyPI", "name": "example"},
        "version": "1.2.3",
    }
    assert captured["timeout"] == 10.0
    assert result["vulnerability_count"] == 0


def test_vulnerability_response_is_normalized():
    payload = {
        "vulns": [
            {
                "id": "GHSA-test-1234",
                "aliases": ["CVE-2026-1234"],
                "summary": "Example vulnerability",
                "published": "2026-01-01T00:00:00Z",
                "modified": "2026-01-02T00:00:00Z",
                "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/test"}],
                "affected": [
                    {
                        "package": {"ecosystem": "PyPI", "name": "example"},
                        "ranges": [
                            {"events": [{"introduced": "0"}, {"fixed": "1.2.4"}]}
                        ],
                    }
                ],
                "references": [{"url": "https://example.test/advisory"}],
            }
        ]
    }

    result = query_osv(
        "PyPI", "example", "1.2.3", opener=lambda *_args, **_kwargs: FakeResponse(payload)
    )
    vulnerability = result["vulnerabilities"][0]
    assert result["vulnerability_count"] == 1
    assert vulnerability["aliases"] == ["CVE-2026-1234"]
    assert vulnerability["fixed_versions"] == ["1.2.4"]
    assert vulnerability["reference"] == "https://example.test/advisory"
