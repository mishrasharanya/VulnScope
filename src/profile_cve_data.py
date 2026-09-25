import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

RAW_DIR = Path(os.getenv("VULNSCOPE_RAW_DIR", "data/raw/cvelistV5/cves"))
OUTPUT_DIR = Path(os.getenv("VULNSCOPE_OUTPUT_DIR", "data/processed"))
OUTPUT_NAME = os.getenv("VULNSCOPE_OUTPUT_NAME", "cves_clean.parquet")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Values that technically exist but are not useful data
# ============================================================

INVALID_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "unknown",
    "not available",
    "*",
}


def valid_value(value):
    if value is None:
        return False

    return (
        str(value)
        .strip()
        .lower()
        not in INVALID_VALUES
    )


def normalize_text(value):
    """
    Normalize human-readable text without discarding its meaning.

    CVE records often contain carriage returns, tabs, repeated newlines,
    and compatibility Unicode characters. These are valid source data but
    make CSV output difficult to inspect and can break some CSV parsers.
    """

    if not valid_value(value):
        return None

    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


# ============================================================
# Generic text extraction
# ============================================================

def get_english_text(items):
    """
    Return the first usable English text entry.
    Used for description and solution fields.
    """

    for item in items or []:

        language = str(
            item.get("lang", "")
        ).lower()

        value = item.get("value")

        if (
            language.startswith("en")
            and valid_value(value)
        ):
            return normalize_text(value)

    return None


# ============================================================
# CWE
# ============================================================

def extract_cwes(cna):
    """
    Extract structured CWE IDs from CNA problemTypes.
    """

    cwes = []

    for problem_type in cna.get(
        "problemTypes",
        []
    ):

        for description in problem_type.get(
            "descriptions",
            []
        ):

            cwe_id = description.get(
                "cweId"
            )

            if valid_value(cwe_id):
                cwes.append(cwe_id)

    return sorted(
        set(cwes)
    )


# ============================================================
# Vendor / product / version information
# ============================================================

def extract_affected(cna):
    """
    Return vendors, products and counts of affected
    version records.
    """

    vendors = []
    products = []

    affected_version_count = 0
    unaffected_version_count = 0

    for item in cna.get(
        "affected",
        []
    ):

        vendor = normalize_text(
            item.get("vendor")
        )

        product = normalize_text(
            item.get("product")
        )

        if valid_value(vendor):
            vendors.append(vendor)

        if valid_value(product):
            products.append(product)

        for version in item.get(
            "versions",
            []
        ):

            status = version.get(
                "status"
            )

            if status == "affected":
                affected_version_count += 1

            elif status == "unaffected":
                unaffected_version_count += 1

    return (
        sorted(set(vendors)),
        sorted(set(products)),
        affected_version_count,
        unaffected_version_count,
    )


# ============================================================
# CVSS
# ============================================================

def find_cvss_in_metrics(metrics):
    """
    Find all CVSS objects in a metrics list.
    """

    results = []

    for metric in metrics or []:

        for key, value in metric.items():

            if (
                key.lower().startswith("cvssv")
                and isinstance(value, dict)
            ):

                results.append({
                    "key": key,
                    "data": value,
                })

    return results


def choose_cvss(cna, adp):
    """
    Prefer CNA-provided CVSS because it is generally
    closer to the original vulnerability disclosure.

    Otherwise use ADP-provided CVSS.

    Within the source, prefer newer CVSS versions.
    """

    candidates = []

    # ----------------------
    # CNA metrics
    # ----------------------

    for result in find_cvss_in_metrics(
        cna.get("metrics", [])
    ):

        candidates.append({
            "source": "CNA",
            "provider_updated":
                cna
                .get("providerMetadata", {})
                .get("dateUpdated"),
            **result,
        })

    # ----------------------
    # ADP metrics
    # ----------------------

    for adp_item in adp:

        provider_metadata = (
            adp_item.get(
                "providerMetadata",
                {}
            )
        )

        provider = (
            provider_metadata
            .get("shortName")
        )

        provider_updated = (
            provider_metadata
            .get("dateUpdated")
        )

        for result in find_cvss_in_metrics(
            adp_item.get("metrics", [])
        ):

            candidates.append({
                "source":
                    provider
                    or "ADP",

                "provider_updated":
                    provider_updated,

                **result,
            })

    if not candidates:
        return None

    version_priority = {
        "cvssV4_0": 4,
        "cvssV3_1": 3,
        "cvssV3_0": 2,
        "cvssV2_0": 1,
    }

    candidates.sort(
        key=lambda x: (
            1
            if x["source"] == "CNA"
            else 0,

            version_priority.get(
                x["key"],
                0
            ),
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# CISA SSVC
# ============================================================

def extract_ssvc(adp):
    """
    Extract CISA SSVC enrichment.

    IMPORTANT:
    These are useful for analysis but are NOT default
    model features because they can contain information
    added after disclosure and therefore create leakage.
    """

    exploitation = None
    automatable = None
    technical_impact = None
    timestamp = None

    for adp_item in adp:

        provider = (
            adp_item
            .get("providerMetadata", {})
            .get("shortName")
        )

        if provider != "CISA-ADP":
            continue

        for metric in adp_item.get(
            "metrics",
            []
        ):

            other = metric.get(
                "other",
                {}
            )

            if (
                other.get("type")
                != "ssvc"
            ):
                continue

            content = other.get(
                "content",
                {}
            )

            timestamp = content.get(
                "timestamp"
            )

            for option in content.get(
                "options",
                []
            ):

                if "Exploitation" in option:
                    exploitation = (
                        option["Exploitation"]
                    )

                if "Automatable" in option:
                    automatable = (
                        option["Automatable"]
                    )

                if "Technical Impact" in option:
                    technical_impact = (
                        option[
                            "Technical Impact"
                        ]
                    )

    return (
        exploitation,
        automatable,
        technical_impact,
        timestamp,
    )


# ============================================================
# Reference features
# Inspired by EPSS reference-count features
# ============================================================

REFERENCE_TAG_FEATURES = {
    "vendor-advisory":
        "ref_vendor_advisory",

    "third-party-advisory":
        "ref_third_party_advisory",

    "patch":
        "ref_patch",

    "exploit":
        "ref_exploit",

    "vdb-entry":
        "ref_vdb",

    "issue-tracking":
        "ref_issue_tracking",

    "mailing-list":
        "ref_mailing_list",

    "government-resource":
        "ref_government",

    "release-notes":
        "ref_release_notes",

    "mitigation":
        "ref_mitigation",

    "technical-description":
        "ref_technical_description",
}


def extract_reference_features(cna):
    """
    Count references and selected reference-tag categories.

    EPSS found reference activity to be highly useful.
    """

    references = cna.get(
        "references",
        []
    )

    features = {
        feature_name: 0
        for feature_name
        in REFERENCE_TAG_FEATURES.values()
    }

    for reference in references:

        tags = [
            str(tag).lower()
            for tag
            in reference.get(
                "tags",
                []
            )
        ]

        for source_tag, feature_name in (
            REFERENCE_TAG_FEATURES.items()
        ):

            if source_tag in tags:
                features[feature_name] += 1

    features["reference_count"] = (
        len(references)
    )

    return features


# ============================================================
# Description-derived features
# Inspired by EPSS text tags
# ============================================================

TEXT_PATTERNS = {

    # Remote exploitation language
    "tag_remote": [
        r"\bremote\b",
        r"\bremote attacker\b",
    ],

    # Code execution
    "tag_code_execution": [
        r"\bcode execution\b",
        r"\barbitrary code\b",
        r"\bexecute arbitrary\b",
        r"\bremote code execution\b",
        r"\brce\b",
    ],

    # Denial of service
    "tag_denial_of_service": [
        r"\bdenial of service\b",
        r"\bdos\b",
    ],

    # Authentication state
    "tag_authenticated": [
        r"\bauthenticated\b",
    ],

    "tag_unauthenticated": [
        r"\bunauthenticated\b",
        r"\bwithout authentication\b",
    ],

    # SQL injection
    "tag_sql_injection": [
        r"\bsql injection\b",
        r"\bsqli\b",
    ],

    # XSS
    "tag_xss": [
        r"\bcross[- ]site scripting\b",
        r"\bxss\b",
    ],

    # Buffer problems
    "tag_buffer_overflow": [
        r"\bbuffer overflow\b",
        r"\bheap overflow\b",
        r"\bstack overflow\b",
    ],

    # Privilege escalation
    "tag_privilege_escalation": [
        r"\bprivilege escalation\b",
        r"\belevation of privilege\b",
    ],

    # Use after free
    "tag_use_after_free": [
        r"\buse after free\b",
        r"\buse-after-free\b",
    ],

    # Path traversal
    "tag_path_traversal": [
        r"\bpath traversal\b",
        r"\bdirectory traversal\b",
    ],

    # Command injection
    "tag_command_injection": [
        r"\bcommand injection\b",
        r"\bos command injection\b",
    ],

    # Information disclosure
    "tag_information_disclosure": [
        r"\binformation disclosure\b",
        r"\binformation exposure\b",
        r"\bsensitive information\b",
    ],

    # Memory corruption
    "tag_memory_corruption": [
        r"\bmemory corruption\b",
        r"\bout[- ]of[- ]bounds\b",
    ],
}


def extract_text_features(text):
    """
    Simple interpretable binary description features.

    For V1 we intentionally keep these transparent
    instead of immediately using embeddings.
    """

    text = (
        str(text).lower()
        if text
        else ""
    )

    features = {}

    for feature_name, patterns in (
        TEXT_PATTERNS.items()
    ):

        found = any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None
            for pattern
            in patterns
        )

        features[feature_name] = int(
            found
        )

    return features


# ============================================================
# Build dataset
# ============================================================

records = []
parse_errors = 0

files = sorted(
    RAW_DIR.rglob(
        "CVE-*.json"
    )
)

print(
    f"Found {len(files):,} CVE files"
)

print(
    "Building VulnScope dataset..."
)


for i, file_path in enumerate(
    files,
    start=1,
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            cve = json.load(f)

    except Exception:

        parse_errors += 1
        continue


    # ========================================================
    # Core CVE structure
    # ========================================================

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

    date_reserved = metadata.get(
        "dateReserved"
    )

    date_published = metadata.get(
        "datePublished"
    )

    date_updated = metadata.get(
        "dateUpdated"
    )

    cna_name = normalize_text(
        metadata.get("assignerShortName")
    )


    # ========================================================
    # Description
    # ========================================================

    title = normalize_text(
        cna.get("title")
    )

    description = get_english_text(
        cna.get(
            "descriptions",
            []
        )
    )


    # ========================================================
    # CWE
    # ========================================================

    cwes = extract_cwes(
        cna
    )


    # ========================================================
    # Affected products
    # ========================================================

    (
        vendors,
        products,
        affected_version_count,
        unaffected_version_count,
    ) = extract_affected(
        cna
    )


    # ========================================================
    # CVSS
    # ========================================================

    selected_cvss = choose_cvss(
        cna,
        adp
    )


    cvss_score = None
    cvss_version = None
    severity = None

    attack_vector = None
    attack_complexity = None

    privileges_required = None
    user_interaction = None

    scope = None

    confidentiality_impact = None
    integrity_impact = None
    availability_impact = None

    cvss_source = None
    cvss_provider_updated = None


    if selected_cvss:

        cvss = selected_cvss[
            "data"
        ]

        cvss_source = selected_cvss[
            "source"
        ]

        cvss_provider_updated = (
            selected_cvss.get(
                "provider_updated"
            )
        )

        cvss_version = cvss.get(
            "version"
        )

        cvss_score = cvss.get(
            "baseScore"
        )

        severity = cvss.get(
            "baseSeverity"
        )

        attack_vector = cvss.get(
            "attackVector"
        )

        attack_complexity = cvss.get(
            "attackComplexity"
        )

        privileges_required = cvss.get(
            "privilegesRequired"
        )

        user_interaction = cvss.get(
            "userInteraction"
        )

        scope = cvss.get(
            "scope"
        )

        confidentiality_impact = cvss.get(
            "confidentialityImpact"
        )

        integrity_impact = cvss.get(
            "integrityImpact"
        )

        availability_impact = cvss.get(
            "availabilityImpact"
        )


    # ========================================================
    # CISA enrichment
    # ========================================================

    has_cisa_adp = any(
        (
            item
            .get("providerMetadata", {})
            .get("shortName")
            == "CISA-ADP"
        )
        for item in adp
    )


    (
        ssvc_exploitation,
        ssvc_automatable,
        ssvc_technical_impact,
        ssvc_timestamp,
    ) = extract_ssvc(
        adp
    )


    # ========================================================
    # References
    # ========================================================

    reference_features = (
        extract_reference_features(
            cna
        )
    )


    # ========================================================
    # Description tags
    # ========================================================

    text_features = (
        extract_text_features(
            description
        )
    )


    # ========================================================
    # Additional metadata
    # ========================================================

    has_solution = bool(
        cna.get(
            "solutions"
        )
    )

    solution = get_english_text(
        cna.get(
            "solutions",
            []
        )
    )

    discovery_type = (
        cna
        .get(
            "source",
            {}
        )
        .get(
            "discovery"
        )
    )

    has_legacy_v4 = (
        "x_legacyV4Record"
        in cna
    )


    # ========================================================
    # Build one CVE row
    # ========================================================

    row = {

        # -----------------------
        # Identification
        # -----------------------

        "cve_id":
            cve_id,

        "state":
            state,

        "date_reserved":
            date_reserved,

        "date_published":
            date_published,

        "date_updated":
            date_updated,

        "cna":
            cna_name,

        # -----------------------
        # Text
        # -----------------------

        "title":
            title,

        "description":
            description,

        # -----------------------
        # CWE
        # -----------------------

        "cwe_ids":
            "|".join(cwes),

        "primary_cwe":
            cwes[0]
            if cwes
            else None,

        "cwe_count":
            len(cwes),

        # -----------------------
        # Vendor/product
        # -----------------------

        "vendors":
            "|".join(vendors),

        "primary_vendor":
            vendors[0]
            if vendors
            else None,

        "vendor_count":
            len(vendors),

        "products":
            "|".join(products),

        "product_count":
            len(products),

        "affected_version_count":
            affected_version_count,

        "unaffected_version_count":
            unaffected_version_count,

        # -----------------------
        # CVSS
        # -----------------------

        "cvss_score":
            cvss_score,

        "cvss_version":
            cvss_version,

        "severity":
            severity,

        "attack_vector":
            attack_vector,

        "attack_complexity":
            attack_complexity,

        "privileges_required":
            privileges_required,

        "user_interaction":
            user_interaction,

        "scope":
            scope,

        "confidentiality_impact":
            confidentiality_impact,

        "integrity_impact":
            integrity_impact,

        "availability_impact":
            availability_impact,

        "cvss_source":
            cvss_source,

        "cvss_provider_updated":
            cvss_provider_updated,

        # -----------------------
        # CISA / analysis-only
        # -----------------------

        "has_cisa_adp":
            has_cisa_adp,

        "ssvc_exploitation":
            ssvc_exploitation,

        "ssvc_automatable":
            ssvc_automatable,

        "ssvc_technical_impact":
            ssvc_technical_impact,

        "ssvc_timestamp":
            ssvc_timestamp,

        # -----------------------
        # Other metadata
        # -----------------------

        "has_solution":
            has_solution,

        "solution":
            solution,

        "discovery_type":
            discovery_type,

        "has_legacy_v4":
            has_legacy_v4,
    }


    # Add reference-derived features
    row.update(
        reference_features
    )

    # Add description-derived features
    row.update(
        text_features
    )

    records.append(
        row
    )


    if i % 10000 == 0:

        print(
            f"Processed {i:,} files"
        )


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    records
)


# Only published vulnerabilities
df = df[
    df["state"]
    == "PUBLISHED"
].copy()


# ============================================================
# Convert dates
# ============================================================

date_columns = [
    "date_reserved",
    "date_published",
    "date_updated",
    "cvss_provider_updated",
    "ssvc_timestamp",
]


for column in date_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=True,
    )


# ============================================================
# Derived time features
# ============================================================

df["publication_year"] = (
    df["date_published"]
    .dt.year
)


# Time between reservation and publication
df["days_reserved_to_published"] = (
    (
        df["date_published"]
        - df["date_reserved"]
    )
    .dt.total_seconds()
    / 86400
)


# IMPORTANT:
# This is useful for data-quality analysis,
# NOT necessarily a prediction feature.
df["days_published_to_updated"] = (
    (
        df["date_updated"]
        - df["date_published"]
    )
    .dt.total_seconds()
    / 86400
)


# ============================================================
# Save complete historical dataset
# ============================================================

df.to_parquet(
    OUTPUT_DIR
    / OUTPUT_NAME,
    index=False,
    compression="zstd",
)


# ============================================================
# Modern modeling population
# 2020–2025
# ============================================================

model_history = df[
    (
        df["publication_year"]
        >= 2020
    )
    &
    (
        df["publication_year"]
        <= 2025
    )
].copy()


# ============================================================
# Training population
# 2020–2024
# ============================================================

train_df = df[
    (
        df["publication_year"]
        >= 2020
    )
    &
    (
        df["publication_year"]
        <= 2024
    )
].copy()


# ============================================================
# Temporal test population
# 2025
# ============================================================

test_df = df[
    df["publication_year"]
    == 2025
].copy()


# ============================================================
# Live scoring population
# 2026
# ============================================================

live_df = df[
    df["publication_year"]
    == 2026
].copy()


# ============================================================
# Summary
# ============================================================

print(
    "\nVulnScope datasets created"
)

print(
    "--------------------------------"
)

print(
    f"Full published CVEs: "
    f"{len(df):,}"
)

print(
    f"2020–2025 modeling population: "
    f"{len(model_history):,}"
)

print(
    f"Train population 2020–2024: "
    f"{len(train_df):,}"
)

print(
    f"Temporal test population 2025: "
    f"{len(test_df):,}"
)

print(
    f"Live scoring population 2026: "
    f"{len(live_df):,}"
)

print(
    f"Parse errors: "
    f"{parse_errors}"
)

print(
    "\nSample 2026 CVEs:"
)

sample_columns = [
    "cve_id",
    "date_published",
    "primary_cwe",
    "cvss_score",
    "attack_vector",
    "privileges_required",
    "reference_count",
    "tag_remote",
    "tag_code_execution",
]


print(
    live_df[
        sample_columns
    ]
    .tail(10)
    .to_string(
        index=False
    )
)
