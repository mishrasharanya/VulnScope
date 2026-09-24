import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/cvelistV5/cves")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INVALID_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "unknown",
    "not available",
    "*"
}


def valid_value(value):
    """Return False for missing/placeholder text such as 'n/a'."""
    if value is None:
        return False

    value = str(value).strip().lower()
    return value not in INVALID_VALUES


def find_key_recursive(obj, target_key):
    """
    Search nested dictionaries/lists for a key.
    Returns all values associated with that key.
    """
    values = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                values.append(value)

            values.extend(find_key_recursive(value, target_key))

    elif isinstance(obj, list):
        for item in obj:
            values.extend(find_key_recursive(item, target_key))

    return values


def has_cvss(obj):
    """Check for any CVSS version anywhere in the CVE record."""
    if isinstance(obj, dict):
        for key, value in obj.items():

            if key.lower().startswith("cvssv"):
                return True

            if has_cvss(value):
                return True

    elif isinstance(obj, list):
        for item in obj:
            if has_cvss(item):
                return True

    return False


def extract_year(date_string, cve_id):
    """
    Prefer actual publication year.
    Fall back to the year embedded in CVE ID.
    """
    if date_string and len(date_string) >= 4:
        try:
            return int(date_string[:4])
        except ValueError:
            pass

    try:
        return int(cve_id.split("-")[1])
    except Exception:
        return None


records = []
parse_errors = 0

files = list(RAW_DIR.rglob("CVE-*.json"))

print(f"Found {len(files):,} CVE files")
print("Profiling...")

for i, file_path in enumerate(files, start=1):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            cve = json.load(f)
    except Exception:
        parse_errors += 1
        continue

    metadata = cve.get("cveMetadata", {})
    containers = cve.get("containers", {})
    cna = containers.get("cna", {})
    adp = containers.get("adp", [])

    cve_id = metadata.get("cveId")
    state = metadata.get("state")
    published_date = metadata.get("datePublished")

    publication_year = extract_year(
        published_date,
        cve_id
    )

    # ----------------------------
    # Vendor / product
    # ----------------------------

    affected = cna.get("affected", [])

    vendors = []
    products = []

    for item in affected:

        vendor = item.get("vendor")
        product = item.get("product")

        if valid_value(vendor):
            vendors.append(vendor)

        if valid_value(product):
            products.append(product)

    has_vendor = len(vendors) > 0
    has_product = len(products) > 0

    # ----------------------------
    # CWE
    # ----------------------------

    cwe_values = find_key_recursive(cve, "cweId")

    has_cwe = any(
        valid_value(value)
        for value in cwe_values
    )

    # ----------------------------
    # CVSS
    # ----------------------------

    cvss_present = has_cvss(cve)

    # ----------------------------
    # CVSS attack vector
    # ----------------------------

    attack_vectors = find_key_recursive(
        cve,
        "attackVector"
    )

    has_attack_vector = any(
        valid_value(value)
        for value in attack_vectors
    )

    # ----------------------------
    # Description
    # ----------------------------

    descriptions = cna.get("descriptions", [])

    has_description = any(
        valid_value(item.get("value"))
        for item in descriptions
    )

    # ----------------------------
    # Other useful metadata
    # ----------------------------

    has_adp = bool(adp)

    has_solution = bool(
        cna.get("solutions")
    )

    has_legacy_v4 = (
        "x_legacyV4Record" in cna
    )

    records.append({
        "cve_id": cve_id,
        "publication_year": publication_year,
        "state": state,

        "has_description": has_description,
        "has_vendor": has_vendor,
        "has_product": has_product,
        "has_cwe": has_cwe,
        "has_cvss": cvss_present,
        "has_attack_vector": has_attack_vector,
        "has_adp": has_adp,
        "has_solution": has_solution,
        "has_legacy_v4": has_legacy_v4
    })

    if i % 10000 == 0:
        print(f"Processed {i:,} files")


df = pd.DataFrame(records)

# Only published vulnerabilities should determine
# feature completeness.
published = df[df["state"] == "PUBLISHED"].copy()


columns_to_measure = [
    "has_description",
    "has_vendor",
    "has_product",
    "has_cwe",
    "has_cvss",
    "has_attack_vector",
    "has_adp",
    "has_solution",
    "has_legacy_v4"
]


summary = (
    published
    .groupby("publication_year")
    .agg(
        total_cves=("cve_id", "count"),
        **{
            column: (column, "mean")
            for column in columns_to_measure
        }
    )
    .reset_index()
)


# Convert proportions to percentages
for column in columns_to_measure:
    summary[column] = (
        summary[column] * 100
    ).round(1)


summary = summary.sort_values(
    "publication_year"
)


output_file = (
    OUTPUT_DIR /
    "cve_completeness_by_year.csv"
)

summary.to_csv(
    output_file,
    index=False
)


print("\nParse errors:", parse_errors)

print("\nFeature completeness by publication year:")
print(summary.tail(15).to_string(index=False))

print(f"\nSaved to: {output_file}")