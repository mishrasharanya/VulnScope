"""Minimal OSV client for VulnScope's dependency-security prototype."""

from __future__ import annotations

import json
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OSV_QUERY_URL = "https://api.osv.dev/v1/query"
OSV_BATCH_QUERY_URL = "https://api.osv.dev/v1/querybatch"
SUPPORTED_ECOSYSTEMS = (
    "PyPI",
    "npm",
    "Maven",
    "Go",
    "crates.io",
    "RubyGems",
    "NuGet",
    "Packagist",
)


class OSVError(ValueError):
    """Expected user-facing OSV lookup error."""


def validate_dependency(ecosystem: str, package: str, version: str) -> tuple[str, str, str]:
    """Validate and normalize a dependency lookup without guessing versions."""
    ecosystem = str(ecosystem).strip()
    package = str(package).strip()
    version = str(version).strip()
    if ecosystem not in SUPPORTED_ECOSYSTEMS:
        raise OSVError("Select a supported package ecosystem.")
    if not package or len(package) > 200:
        raise OSVError("Enter a package name between 1 and 200 characters.")
    if not version or len(version) > 100:
        raise OSVError("Enter an exact package version.")
    if re.search(r"[<>=~*^,;\s]", version):
        raise OSVError(
            "Enter one exact installed version, not a version range such as >=2.0."
        )
    return ecosystem, package, version


def _severity_entries(vulnerability: dict) -> list[str]:
    entries = []
    for item in vulnerability.get("severity", []) or []:
        if not isinstance(item, dict):
            continue
        severity_type = item.get("type")
        score = item.get("score")
        if severity_type and score:
            entries.append(f"{severity_type}: {score}")
    return entries


def _fixed_versions(vulnerability: dict, ecosystem: str, package: str) -> list[str]:
    versions = []
    for affected in vulnerability.get("affected", []) or []:
        package_data = affected.get("package", {})
        if (
            package_data.get("ecosystem") != ecosystem
            or package_data.get("name", "").lower() != package.lower()
        ):
            continue
        for item in affected.get("ranges", []) or []:
            for event in item.get("events", []) or []:
                if event.get("fixed"):
                    versions.append(str(event["fixed"]))
    return sorted(set(versions))


def normalize_vulnerability(
    vulnerability: dict,
    ecosystem: str,
    package: str,
) -> dict:
    """Return only the fields the prototype presents to the user."""
    vulnerability_id = str(vulnerability.get("id", "Unknown"))
    aliases = [str(value) for value in vulnerability.get("aliases", []) or []]
    references = [
        str(item["url"])
        for item in vulnerability.get("references", []) or []
        if isinstance(item, dict) and item.get("url")
    ]
    return {
        "id": vulnerability_id,
        "aliases": aliases,
        "summary": vulnerability.get("summary") or "No summary provided.",
        "severity": _severity_entries(vulnerability),
        "fixed_versions": _fixed_versions(vulnerability, ecosystem, package),
        "published": vulnerability.get("published"),
        "modified": vulnerability.get("modified"),
        "reference": references[0] if references else f"https://osv.dev/vulnerability/{vulnerability_id}",
    }


def query_osv(
    ecosystem: str,
    package: str,
    version: str,
    *,
    opener: Callable = urlopen,
    timeout: float = 10.0,
) -> dict:
    """Query OSV for known vulnerabilities affecting one exact dependency."""
    ecosystem, package, version = validate_dependency(ecosystem, package, version)
    payload = {
        "package": {"ecosystem": ecosystem, "name": package},
        "version": version,
    }
    request = Request(
        OSV_QUERY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "VulnScope/2-prototype",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise OSVError(f"OSV rejected the query with HTTP {error.code}.") from error
    except URLError as error:
        raise OSVError(f"OSV could not be reached: {error.reason}") from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise OSVError("OSV returned an invalid response or timed out.") from error

    vulnerabilities = [
        normalize_vulnerability(item, ecosystem, package)
        for item in result.get("vulns", []) or []
    ]
    return {
        "ecosystem": ecosystem,
        "package": package,
        "version": version,
        "vulnerability_count": len(vulnerabilities),
        "vulnerabilities": vulnerabilities,
        "source": "OSV.dev",
        "interpretation": (
            "Results are known published vulnerabilities associated with this "
            "exact package version. No result does not prove the dependency is safe."
        ),
    }


def query_osv_batch(
    dependencies: list[dict],
    *,
    opener: Callable = urlopen,
    timeout: float = 30.0,
    batch_size: int = 500,
) -> list[dict]:
    """Query OSV for many exact package versions while preserving input order."""
    if not dependencies:
        return []
    if not 1 <= batch_size <= 1_000:
        raise OSVError("OSV batch size must be between 1 and 1,000.")

    output = []
    for start in range(0, len(dependencies), batch_size):
        batch = dependencies[start : start + batch_size]
        queries = []
        normalized = []
        for dependency in batch:
            ecosystem, package, version = validate_dependency(
                dependency.get("ecosystem"),
                dependency.get("package"),
                dependency.get("version"),
            )
            normalized.append((ecosystem, package, version))
            queries.append(
                {
                    "package": {"ecosystem": ecosystem, "name": package},
                    "version": version,
                }
            )
        request = Request(
            OSV_BATCH_QUERY_URL,
            data=json.dumps({"queries": queries}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "VulnScope/2-prototype",
            },
            method="POST",
        )
        try:
            with opener(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise OSVError(
                f"OSV rejected the batch query with HTTP {error.code}."
            ) from error
        except URLError as error:
            raise OSVError(f"OSV could not be reached: {error.reason}") from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise OSVError("OSV returned an invalid batch response or timed out.") from error

        results = result.get("results", [])
        if len(results) != len(batch):
            raise OSVError("OSV returned an unexpected number of batch results.")
        for dependency, identity, item in zip(batch, normalized, results):
            ecosystem, package, version = identity
            vulnerabilities = [
                {
                    "id": str(vulnerability.get("id", "Unknown")),
                    "modified": vulnerability.get("modified"),
                    "reference": (
                        "https://osv.dev/vulnerability/"
                        f"{vulnerability.get('id', 'Unknown')}"
                    ),
                }
                for vulnerability in item.get("vulns", []) or []
            ]
            output.append(
                {
                    **dependency,
                    "ecosystem": ecosystem,
                    "package": package,
                    "version": version,
                    "vulnerability_count": len(vulnerabilities),
                    "vulnerabilities": vulnerabilities,
                    "more_results_available": bool(item.get("next_page_token")),
                }
            )
    return output
