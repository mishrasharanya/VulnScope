"""Streamlit interface for the VulnScope CVE-prioritization service."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vulnscope_service import VulnScopeError, get_service  # noqa: E402


st.set_page_config(
    page_title="VulnScope",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(ttl=300)
def load_service():
    return get_service()


def probability_label(probability: float) -> str:
    return f"{probability:.1%}"


def show_error(error: Exception) -> None:
    st.error(str(error))


def render_cve_summary(record: dict) -> None:
    st.subheader(record["cve_id"])
    if record.get("title"):
        st.markdown(f"**{record['title']}**")
    st.write(record.get("description") or "No description is available.")
    columns = st.columns(4)
    columns[0].metric("Vendor", record.get("primary_vendor") or "Unknown")
    columns[1].metric("Product", record.get("products") or "Unknown")
    columns[2].metric("CWE", record.get("primary_cwe") or "Unknown")
    cvss = record.get("cvss_score")
    columns[3].metric("Recorded CVSS", "Missing" if cvss is None else f"{cvss:.1f}")
    st.caption(
        f"Published: {record.get('date_published') or 'unknown'} · "
        f"Updated: {record.get('date_updated') or 'unknown'}"
    )


def render_prediction(prediction: dict) -> None:
    probability = prediction["priority_probability"]
    columns = st.columns(3)
    columns[0].metric("Severe-CVE probability", probability_label(probability))
    columns[1].metric(
        "Decision",
        "Flagged" if prediction["flagged"] else "Not flagged",
    )
    columns[2].metric(
        "Threshold", probability_label(prediction["decision_threshold"])
    )
    st.progress(probability)
    st.caption(
        f"{prediction['target_definition']} · "
        f"Model {prediction['model_version']} · "
        f"Source: {prediction['prediction_source']}"
    )
    st.warning(prediction["warning"], icon="⚠️")


def render_explanation(explanation: dict) -> None:
    st.markdown("#### Model explanation")
    left, right = st.columns(2)
    with left:
        st.markdown("**Terms supporting a severe classification**")
        supporting = pd.DataFrame(explanation["supporting_terms"])
        if supporting.empty:
            st.info("No positive term contributions were found.")
        else:
            st.dataframe(supporting, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Terms opposing a severe classification**")
        opposing = pd.DataFrame(explanation["opposing_terms"])
        if opposing.empty:
            st.info("No negative term contributions were found.")
        else:
            st.dataframe(opposing, hide_index=True, use_container_width=True)
    st.caption(explanation["explanation_scope"])


def render_evidence(evidence: list[dict]) -> None:
    for item in evidence:
        with st.expander(
            f"{item['cve_id']} · similarity {item['similarity_score']:.3f}"
        ):
            st.write(item["description"])
            st.markdown("**Published historical solution**")
            st.write(item["historical_solution"])
            st.caption(
                f"Vendor: {item.get('vendor') or 'unknown'} · "
                f"Product: {item.get('products') or 'unknown'} · "
                f"CWE: {item.get('primary_cwe') or 'unknown'}"
            )


service = load_service()

st.title("🔎 VulnScope")
st.write(
    "Prioritize and explain CVEs using a temporally evaluated severity model "
    "and evidence retrieved from historical CVE solutions."
)

with st.sidebar:
    st.header("Model")
    st.write(service.metadata["label_definition"])
    st.metric("2025 temporal-test F1", f"{service.metadata['test_metrics']['f1']:.3f}")
    st.metric(
        "2025 temporal-test recall",
        f"{service.metadata['test_metrics']['recall']:.3f}",
    )
    st.caption(
        "VulnScope predicts technical severity. It does not predict exploitation "
        "or replace vendor remediation guidance."
    )
    if st.button("Reload latest local data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

lookup_tab, brief_tab, compare_tab, alerts_tab, monitoring_tab = st.tabs(
    [
        "Search & analyze",
        "Analyst brief",
        "Compare CVEs",
        "Recent alerts",
        "Data & monitoring",
    ]
)

with lookup_tab:
    search_query = st.text_input(
        "Search CVE, vendor, product, CWE, or vulnerability description",
        value="",
        placeholder="CVE-2026-81537 or Apache remote code execution",
        key="search_query",
    )
    result_limit = st.slider("Maximum results", 5, 30, 15)
    if st.button("Search", type="primary", key="search_button"):
        try:
            with st.spinner("Searching CVE intelligence…"):
                st.session_state["search_results"] = service.search_cves(
                    search_query, limit=result_limit
                )
                st.session_state["search_results_query"] = search_query
        except (VulnScopeError, OSError, KeyError) as error:
            show_error(error)

    results = st.session_state.get("search_results", [])
    if results and st.session_state.get("search_results_query") == search_query:
        display = pd.DataFrame(results)[
            [
                "cve_id", "vendor", "products", "primary_cwe",
                "date_published", "priority_probability", "flagged",
            ]
        ].copy()
        display["priority_probability"] = display[
            "priority_probability"
        ].map(lambda value: f"{value:.1%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        selected_id = st.selectbox(
            "Select a CVE to analyze",
            [item["cve_id"] for item in results],
            key="selected_search_cve",
        )
        if st.button("Analyze selected CVE", key="analyze_button"):
            st.session_state["analyzed_cve"] = selected_id
    elif st.session_state.get("search_results_query") == search_query:
        st.info("No matching CVEs found.")

    analyzed_id = st.session_state.get("analyzed_cve")
    if analyzed_id:
        try:
            record = service.get_cve(analyzed_id)
            explanation = service.explain_prediction(analyzed_id)
            render_cve_summary(record)
            render_prediction(explanation)
            render_explanation(explanation)
        except (VulnScopeError, OSError, KeyError) as error:
            show_error(error)

with brief_tab:
    st.write(
        "Generate a structured, evidence-grounded report from recorded CVE "
        "facts, the saved ML score, model terms, and historical solutions."
    )
    brief_id = st.text_input(
        "CVE ID",
        value="",
        placeholder="CVE-2026-12345",
        key="brief_id",
    )
    brief_evidence_limit = st.slider(
        "Historical sources", 3, 10, 5, key="brief_evidence_limit"
    )
    retrieve_column, generate_column = st.columns(2)
    if retrieve_column.button(
        "Retrieve historical evidence",
        key="brief_retrieve_button",
        help="Runs locally without making a Groq API request.",
    ):
        try:
            evidence = service.retrieve_similar_solutions(
                brief_id, limit=brief_evidence_limit
            )
            st.session_state["brief_evidence"] = evidence
            st.session_state["brief_evidence_cve"] = brief_id.strip().upper()
        except (VulnScopeError, OSError, KeyError) as error:
            show_error(error)

    if generate_column.button(
        "Generate full brief with Groq",
        type="primary",
        key="brief_button",
        help="This makes one Groq API request using retrieved evidence.",
    ):
        try:
            with st.spinner("Building the evidence package and analyst brief…"):
                brief = service.generate_analyst_brief(
                    brief_id, evidence_limit=brief_evidence_limit
                )
            st.session_state["analyst_brief"] = brief
            st.session_state["brief_evidence"] = brief["evidence"]
            st.session_state["brief_evidence_cve"] = brief["cve_id"]
        except (VulnScopeError, OSError, KeyError) as error:
            show_error(error)

    brief = st.session_state.get("analyst_brief")
    if brief and brief["cve_id"] == brief_id.strip().upper():
        prediction = brief["prediction"]
        columns = st.columns(3)
        columns[0].metric(
            "Saved severity probability",
            probability_label(prediction["priority_probability"]),
        )
        columns[1].metric(
            "Decision", "Flagged" if prediction["flagged"] else "Not flagged"
        )
        columns[2].metric("Model", prediction["model_version"])
        st.markdown(brief["analyst_brief"])
        st.warning(brief["warning"], icon="⚠️")
        st.download_button(
            "Download brief as Markdown",
            data=brief["analyst_brief"],
            file_name=f"{brief['cve_id']}_analyst_brief.md",
            mime="text/markdown",
        )
        st.caption(f"Generated with {brief['llm_model']}")

    evidence = st.session_state.get("brief_evidence")
    if (
        evidence
        and st.session_state.get("brief_evidence_cve")
        == brief_id.strip().upper()
    ):
        st.markdown("### Historical evidence sources")
        render_evidence(evidence)

with compare_tab:
    compare_text = st.text_area(
        "Enter 2–10 CVE IDs, separated by commas or new lines",
        placeholder="CVE-2026-12345\nCVE-2026-54321",
    )
    if st.button("Compare", type="primary", key="compare_button"):
        cve_ids = [
            value.strip()
            for value in compare_text.replace(",", "\n").splitlines()
            if value.strip()
        ]
        try:
            comparison = pd.DataFrame(service.compare_cves(cve_ids))
            comparison["priority_probability"] = comparison[
                "priority_probability"
            ].map(lambda value: f"{value:.1%}")
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        except (VulnScopeError, OSError, KeyError) as error:
            show_error(error)

with alerts_tab:
    update_report_path = ROOT / "reports" / "ingestion" / "latest_update.json"
    scoring_report_path = ROOT / "reports" / "ingestion" / "latest_scoring.json"
    if update_report_path.exists():
        update_report = json.loads(update_report_path.read_text())
        columns = st.columns(3)
        columns[0].metric("Changed in last sync", update_report["changed_files"])
        columns[1].metric(
            "Published rows processed", update_report["published_rows_parsed"]
        )
        columns[2].metric("Dataset rows", update_report["rows_after"])
        st.caption(f"Last successful ingestion: {update_report['completed_at']}")
    if scoring_report_path.exists():
        scoring_report = json.loads(scoring_report_path.read_text())
        st.caption(
            f"Latest changed-CVE scoring: {scoring_report['live_2026_scored']:,} "
            f"scored · {scoring_report['live_2026_flagged']:,} flagged · "
            f"model {scoring_report['model_version']}"
        )
    alert_limit = st.slider("Number of alerts", 5, 100, 20, step=5)
    try:
        alerts = pd.DataFrame(service.get_recent_flagged_cves(alert_limit))
        alerts["priority_probability"] = alerts["priority_probability"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(alerts, hide_index=True, use_container_width=True)
        st.caption(
            "These records crossed the severity-model threshold; they are not "
            "confirmed exploitation alerts."
        )
    except (VulnScopeError, OSError, KeyError) as error:
        show_error(error)

with monitoring_tab:
    st.header("Data and model monitoring")
    st.caption(
        "Operational health for the canonical CVE dataset and live severity "
        "predictions. These charts do not retrain the model."
    )

    update_report_path = ROOT / "reports" / "ingestion" / "latest_update.json"
    scoring_report_path = ROOT / "reports" / "ingestion" / "latest_scoring.json"
    update_report = (
        json.loads(update_report_path.read_text())
        if update_report_path.exists()
        else {}
    )
    scoring_report = (
        json.loads(scoring_report_path.read_text())
        if scoring_report_path.exists()
        else {}
    )

    canonical = service.data
    predictions = service.live_predictions
    live_2026 = canonical[canonical["publication_year"].eq(2026)].copy()
    flagged = predictions[predictions["flagged"]]
    last_published = live_2026["date_published"].max()
    freshness_columns = st.columns(5)
    freshness_columns[0].metric("Canonical CVEs", f"{len(canonical):,}")
    freshness_columns[1].metric("Published in 2026", f"{len(live_2026):,}")
    freshness_columns[2].metric("2026 flagged", f"{len(flagged):,}")
    freshness_columns[3].metric(
        "Flag rate",
        f"{len(flagged) / len(predictions):.1%}" if len(predictions) else "N/A",
    )
    freshness_columns[4].metric(
        "Latest publication",
        last_published.strftime("%Y-%m-%d") if pd.notna(last_published) else "N/A",
    )
    if update_report:
        st.success(
            f"Last ingestion completed {update_report.get('completed_at', 'unknown')} · "
            f"{update_report.get('changed_files', 0):,} changed files · "
            f"source {update_report.get('to_revision', 'unknown')[:12]}"
        )
    else:
        st.warning("No successful ingestion report is available.")

    st.subheader("2026 data completeness")
    completeness_fields = {
        "Description": "description",
        "Vendor": "primary_vendor",
        "Product": "products",
        "CWE": "primary_cwe",
        "CVSS": "cvss_score",
        "Solution": "solution",
    }
    completeness = pd.DataFrame(
        {
            "field": list(completeness_fields),
            "complete_share": [
                live_2026[column].notna().mean()
                for column in completeness_fields.values()
            ],
        }
    ).set_index("field")
    st.bar_chart(completeness, y="complete_share")
    st.dataframe(
        completeness.assign(
            complete_share=completeness["complete_share"].map(
                lambda value: f"{value:.1%}"
            )
        ),
        use_container_width=True,
    )

    volume_column, probability_column = st.columns(2)
    with volume_column:
        st.subheader("CVE publication volume")
        monthly = (
            live_2026.dropna(subset=["date_published"])
            .assign(month=lambda frame: frame["date_published"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
            .rename("published_cves")
        )
        st.line_chart(monthly)
    with probability_column:
        st.subheader("Severity probability distribution")
        bins = pd.IntervalIndex.from_breaks(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
            closed="left",
        )
        distribution = (
            pd.cut(predictions["priority_probability"], bins=bins)
            .value_counts(sort=False)
            .rename("cves")
        )
        distribution.index = [
            f"{interval.left:.1f}–{min(interval.right, 1):.1f}"
            for interval in distribution.index
        ]
        st.bar_chart(distribution)

    details = predictions.reset_index(drop=True).merge(
        live_2026[
            ["cve_id", "primary_vendor", "primary_cwe"]
        ].reset_index(drop=True),
        on="cve_id",
        how="left",
        validate="one_to_one",
    )
    flagged_details = details[details["flagged"]]
    vendor_column, cwe_column = st.columns(2)
    with vendor_column:
        st.subheader("Top flagged vendors")
        top_vendors = (
            flagged_details["primary_vendor"]
            .fillna("Missing")
            .value_counts()
            .head(12)
            .sort_values()
        )
        st.bar_chart(top_vendors)
    with cwe_column:
        st.subheader("Top flagged CWEs")
        top_cwes = (
            flagged_details["primary_cwe"]
            .fillna("Missing")
            .value_counts()
            .head(12)
            .sort_values()
        )
        st.bar_chart(top_cwes)

    st.subheader("Saved model evaluation")
    test_metrics = service.metadata["test_metrics"]
    metric_columns = st.columns(5)
    for column, (label, key) in zip(
        metric_columns,
        [
            ("F1", "f1"),
            ("Recall / coverage", "recall"),
            ("Precision / efficiency", "precision"),
            ("PR-AUC", "pr_auc"),
            ("Brier", "brier_score"),
        ],
    ):
        column.metric(label, f"{test_metrics[key]:.3f}")
    st.caption(
        f"Model {service.metadata['model_version']} · "
        f"{service.metadata['label_definition']} · "
        f"threshold {service.metadata['selected_threshold']:.3f}"
    )

    coverage_plot = (
        ROOT / "reports" / "cve_priority_model" / "coverage_vs_efficiency.png"
    )
    if coverage_plot.exists():
        st.subheader("Coverage versus efficiency")
        # Avoid version-specific sizing arguments. Streamlit Cloud may install a
        # newer release where ``use_column_width`` has been removed, while the
        # local environment can still be on a release predating ``width='stretch'``.
        st.image(str(coverage_plot))
        st.caption(
            "Coverage is recall and efficiency is precision for the severe-CVE "
            "target. This is not exploitation coverage."
        )

    with st.expander("Pipeline details"):
        st.json(
            {
                "latest_ingestion": update_report,
                "latest_scoring": scoring_report,
            }
        )
