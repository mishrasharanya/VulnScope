"""Parse npm package-lock files and query OSV for installed dependencies."""

from __future__ import annotations

import json
from typing import Callable

from osv_service import OSVError, query_osv_batch


class LockfileError(ValueError):
    """Expected user-facing package-lock parsing error."""


def _package_name_from_path(path: str) -> str | None:
    if "node_modules/" not in path:
        return None
    name = path.rsplit("node_modules/", 1)[-1].strip("/")
    return name or None


def parse_package_lock(content: bytes | str | dict) -> dict:
    """Extract exact installed packages from npm lockfile versions 2 and 3."""
    if isinstance(content, dict):
        payload = content
    else:
        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LockfileError("The uploaded file is not valid JSON.") from error

    lockfile_version = payload.get("lockfileVersion")
    if lockfile_version not in {2, 3}:
        raise LockfileError(
            "This prototype supports package-lock.json versions 2 and 3."
        )
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        raise LockfileError("The lockfile does not contain an npm packages map.")

    root = packages.get("", {})
    declared = set()
    for field in [
        "dependencies",
        "devDependencies",
        "optionalDependencies",
        "peerDependencies",
    ]:
        values = root.get(field, {})
        if isinstance(values, dict):
            declared.update(values)

    dependencies = []
    skipped = []
    seen = set()
    for path, package_data in packages.items():
        if not path or not isinstance(package_data, dict):
            continue
        package = package_data.get("name") or _package_name_from_path(path)
        version = str(package_data.get("version", "")).strip()
        if not package or not version or package_data.get("link"):
            skipped.append(path)
            continue
        if version.startswith(("file:", "git+", "http:", "https:", "workspace:")):
            skipped.append(path)
            continue
        identity = (str(package), version, path)
        if identity in seen:
            continue
        seen.add(identity)
        dependencies.append(
            {
                "ecosystem": "npm",
                "package": str(package),
                "version": version,
                "path": path,
                "relationship": (
                    "direct"
                    if path == f"node_modules/{package}" and package in declared
                    else "transitive"
                ),
                "development": bool(package_data.get("dev", False)),
            }
        )
    dependencies.sort(key=lambda item: (item["package"].lower(), item["path"]))
    return {
        "project_name": payload.get("name") or root.get("name") or "Unnamed npm project",
        "lockfile_version": int(lockfile_version),
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
        "skipped_entries": skipped,
    }


def scan_package_lock(
    content: bytes | str | dict,
    *,
    batch_query: Callable = query_osv_batch,
) -> dict:
    """Parse a lockfile and return package-level OSV matches."""
    parsed = parse_package_lock(content)
    try:
        results = batch_query(parsed["dependencies"])
    except OSVError:
        raise
    vulnerable = [item for item in results if item["vulnerability_count"] > 0]
    vulnerability_ids = sorted(
        {
            vulnerability["id"]
            for item in vulnerable
            for vulnerability in item["vulnerabilities"]
        }
    )
    return {
        **parsed,
        "scanned_count": len(results),
        "vulnerable_package_count": len(vulnerable),
        "unique_vulnerability_count": len(vulnerability_ids),
        "unique_vulnerability_ids": vulnerability_ids,
        "vulnerable_packages": vulnerable,
        "interpretation": (
            "Matches show known published vulnerabilities associated with exact "
            "installed versions. They do not prove that vulnerable code is reachable."
        ),
    }
