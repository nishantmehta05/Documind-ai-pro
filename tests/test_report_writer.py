"""Tests for local evidence-grounded report generation."""

from src.agents.report_writer import ReportWriter
from src.config import config
from src.retrieval.vector_store import SearchResult


def test_report_writer_uses_configured_token_and_timeout_limits(monkeypatch):
    calls = {}

    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"response": "The paper supports this finding [1]."}

    def fake_post(url, json, timeout):
        calls["payload"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.agents.report_writer.requests.post", fake_post)
    passage = SearchResult(
        chunk_id="paper_p1_c0",
        doc_id="paper",
        filename="paper.pdf",
        page_number=1,
        chunk_index=0,
        text="The paper supports this finding.",
    )
    report = ReportWriter().write_report("Summarize the findings", [passage])

    assert report.is_error is False
    assert calls["timeout"] == config.report_timeout_seconds
    assert calls["payload"]["options"]["num_predict"] == config.max_report_tokens
