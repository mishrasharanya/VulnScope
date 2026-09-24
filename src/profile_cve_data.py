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
    """
    Returns False for missing or placeholder values
    such as None, 'n/a', 'unknown', etc.
    """
    if value is None:
        return False

    value = str(value).strip().lower()

    return value not in INVALID_VALUES


def find_key_recursive(obj, target_key):
    """
    Search through nested dictionaries and lists
    and return all values matching target_key.
    """

    values = []

    if isinstance(obj, dict):
        for key, value in obj.items():

            if key == target_key:
                values.append(value)

            values.extend(
                find_key_recursive(value, target_key)
            )

    elif isinstance(obj, list):
        for item in obj:
            values.extend(
                find_key_recursive(item, target_key)
            )

    return values


def has_cvss(obj):
    """
    Detect whether any CVSS version exists
    anywhere inside the CVE JSON.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            # Examples:
            # cvssV3_1
            # cvssV3_0
            # cvssV4_0
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
    Prefer publication year from datePublished.

    If datePublished is missing, fall back
    to the year inside the CVE ID.
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


# Find every CVE JSON file
files = list(
    RAW_DIR.rglob("CVE-*.json")
)

print(f"Found {len(files):,} CVE files")
print("Profiling CVE dataset...")


for i, file_path in enumerate(files, start=1):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            cve = json.load(f)

    except Exception:

        parse_errors += 1
        continue


    # ----------------------------
    # Basic CVE structure
    # ----------------------------

    metadata = cve.get(
        "cveMetadata",
        {}
    )

    containers = cve.get(
        "containers",
        {}
    )

    cna = containers.get(
        "cna",
        {}
    )

    adp = containers.get(
        "adp",
        []
    )


    cve_id = metadata.get(
        "cveId"
    )

    state = metadata.get(
        "state"
    )

    published_date = metadata.get(
        "datePublished"
    )


    publication_year = extract_year(
        published_date,
        cve_id
    )


    # ----------------------------
    # Vendor + Product
    # ----------------------------

    affected = cna.get(
        "affected",
        []
    )

    vendors = []
    products = []


    for item in affected:

        vendor = item.get(
            "vendor"
        )

        product = item.get(
            "product"
        )


        if valid_value(vendor):
            vendors.append(vendor)

        if valid_value(product):
            products.append(product)


    has_vendor = (
        len(vendors) > 0
    )

    has_product = (
        len(products) > 0
    )


    # ----------------------------
    # CWE
    # ----------------------------

    cwe_values = find_key_recursive(
        cve,
        "cweId"
    )

    has_cwe = any(
        valid_value(value)
        for value in cwe_values
    )


    # ----------------------------
    # CVSS
    # ----------------------------

    cvss_present = has_cvss(
        cve
    )


    # ----------------------------
    # Attack Vector
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

    descriptions = cna.get(
        "descriptions",
        []
    )

    has_description = any(
        valid_value(
            item.get("value")
        )
        for item in descriptions
    )


    # ----------------------------
    # CISA ADP enrichment
    # ----------------------------

    has_cisa_adp = any(

        item
        .get(
            "providerMetadata",
            {}
        )
        .get(
            "shortName"
        )
        == "CISA-ADP"

        for item in adp
    )


    # ----------------------------
    # Solution / remediation
    # ----------------------------

    has_solution = bool(
        cna.get(
            "solutions"
        )
    )


    # ----------------------------
    # Legacy CVE V4 record
    # ----------------------------

    has_legacy_v4 = (
        "x_legacyV4Record"
        in cna
    )


    # ----------------------------
    # Save one profiling row
    # ----------------------------

    records.append({

        "cve_id":
            cve_id,

        "publication_year":
            publication_year,

        "state":
            state,

        "has_description":
            has_description,

        "has_vendor":
            has_vendor,

        "has_product":
            has_product,

        "has_cwe":
            has_cwe,

        "has_cvss":
            cvss_present,

        "has_attack_vector":
            has_attack_vector,

        "has_cisa_adp":
            has_cisa_adp,

        "has_solution":
            has_solution,

        "has_legacy_v4":
            has_legacy_v4
    })


    # Show progress every 10k files
    if i % 10000 == 0:

        print(
            f"Processed {i:,} files"
        )


# ----------------------------
# Convert to DataFrame
# ----------------------------

df = pd.DataFrame(
    records
)


# Only published vulnerabilities
# should determine completeness
published = df[
    df["state"] == "PUBLISHED"
].copy()


# ----------------------------
# Fields whose completeness
# we want to measure
# ----------------------------

columns_to_measure = [

    "has_description",

    "has_vendor",

    "has_product",

    "has_cwe",

    "has_cvss",

    "has_attack_vector",

    "has_cisa_adp",

    "has_solution",

    "has_legacy_v4"
]


# ----------------------------
# Aggregate by publication year
# ----------------------------

summary = (

    published

    .groupby(
        "publication_year"
    )

    .agg(

        total_cves=(
            "cve_id",
            "count"
        ),

        **{

            column: (
                column,
                "mean"
            )

            for column
            in columns_to_measure
        }
    )

    .reset_index()
)


# ----------------------------
# Convert proportions → %
# ----------------------------

for column in columns_to_measure:

    summary[column] = (

        summary[column]
        * 100

    ).round(1)


# Sort chronologically
summary = summary.sort_values(
    "publication_year"
)


# ----------------------------
# Save CSV
# ----------------------------

output_file = (

    OUTPUT_DIR
    / "cve_completeness_by_year.csv"
)


summary.to_csv(

    output_file,

    index=False
)


# ----------------------------
# Output results
# ----------------------------

print(
    "\nParse errors:",
    parse_errors
)


print(
    "\nFeature completeness "
    "by publication year:"
)


print(

    summary
    .tail(20)
    .to_string(
        index=False
    )
)


print(
    f"\nSaved to: "
    f"{output_file}"
)