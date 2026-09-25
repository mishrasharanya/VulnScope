"""Offline tests for the VulnScope service; no Groq request is made."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vulnscope_service import VulnScopeError, VulnScopeService  # noqa: E402


@pytest.fixture(scope="module")
def service():
    return VulnScopeService()


def test_rejects_invalid_cve_id(service):
    with pytest.raises(VulnScopeError):
        service.get_cve("not-a-cve")


def test_get_and_predict_known_2026_cve(service):
    cve_id = service.live_predictions.iloc[0]["cve_id"]
    record = service.get_cve(cve_id)
    prediction = service.predict_priority(cve_id)
    assert record["cve_id"] == cve_id
    assert 0 <= prediction["priority_probability"] <= 1
    assert prediction["prediction_source"] == "saved_live_prediction"
    assert "not exploitation" in prediction["warning"]


def test_explanation_has_model_terms(service):
    cve_id = service.live_predictions.sort_values(
        "priority_probability", ascending=False
    ).iloc[0]["cve_id"]
    explanation = service.explain_prediction(cve_id)
    assert explanation["supporting_terms"]
    assert all("term" in item for item in explanation["supporting_terms"])


def test_retrieval_excludes_query_and_has_solutions(service):
    query = service.solution_corpus.sort_values("date_published").iloc[-1]["cve_id"]
    query_date = service._record(query)["date_published"]
    results = service.retrieve_similar_solutions(query, limit=3)
    assert len(results) == 3
    assert all(item["cve_id"] != query for item in results)
    assert all(item["historical_solution"] for item in results)
    assert all(service._record(item["cve_id"])["date_published"] < query_date for item in results)


def test_solution_summary_uses_injected_llm(service):
    query = service.solution_corpus.sort_values("date_published").iloc[-1]["cve_id"]
    captured = {}

    def fake_llm(messages, model_name):
        captured["messages"] = messages
        captured["model"] = model_name
        return "Possible historical pattern [CVE-TEST-0001]. Verify vendor advice."

    result = service.suggest_possible_solutions(
        query, limit=2, llm_callable=fake_llm
    )
    assert result["possible_solution_summary"].startswith("Possible")
    assert len(result["evidence"]) == 2
    assert "HISTORICAL EVIDENCE" in captured["messages"][1]["content"]
    assert "authoritative" in result["warning"]


def test_analyst_brief_uses_saved_score_and_evidence(service):
    query = service.solution_corpus.sort_values("date_published").iloc[-1]["cve_id"]
    expected = service.predict_priority(query)
    captured = {}

    def fake_llm(messages, model_name):
        captured["prompt"] = messages[1]["content"]
        return "1. Executive summary\nEvidence-grounded test brief."

    result = service.generate_analyst_brief(
        query, evidence_limit=2, llm_callable=fake_llm
    )
    assert result["prediction"]["priority_probability"] == expected[
        "priority_probability"
    ]
    assert result["prediction"]["model_version"] == expected["model_version"]
    assert len(result["evidence"]) == 2
    assert "RECORDED CVE FACTS" in captured["prompt"]
    assert "MODEL OUTPUT" in captured["prompt"]
    assert "HISTORICAL EVIDENCE" in captured["prompt"]
