"""Streamlit interface for the VulnScope CVE-prioritization service."""

from __future__ import annotations

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


@st.cache_resource
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

lookup_tab, brief_tab, compare_tab, alerts_tab = st.tabs(
    [
        "CVE lookup",
        "Analyst brief",
        "Compare CVEs",
        "Recent alerts",
    ]
)

with lookup_tab:
    lookup_id = st.text_input(
        "CVE ID",
        value="",
        placeholder="CVE-2026-12345",
        key="lookup_id",
    )
    if st.button("Analyze CVE", type="primary", key="analyze_button"):
        try:
            record = service.get_cve(lookup_id)
            explanation = service.explain_prediction(lookup_id)
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
