"""Tests for local research question planning."""

import requests

from src.agents.research_planner import ResearchPlanner
from src.config import config


def test_research_prompt_formats_json_instructions():
    prompt = ResearchPlanner.PLAN_PROMPT_TEMPLATE.format(topic="compare study findings")
    assert '"sub_questions"' in prompt
    assert "compare study findings" in prompt


def test_research_planner_uses_configured_token_and_timeout_limits(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "response": (
                    '{"reasoning":"Split by topic","sub_questions":["Question one"]}'
                )
            }

    def fake_post(url, json, timeout):
        calls["payload"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.agents.research_planner.requests.post", fake_post)
    plan = ResearchPlanner().plan("compare study findings")

    assert plan.used_heuristic_fallback is False
    assert calls["timeout"] == config.planner_timeout_seconds
    assert calls["payload"]["options"]["num_predict"] == config.max_planning_tokens


def test_research_planner_marks_heuristic_fallback(monkeypatch):
    def fail_request(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Ollama is offline")

    monkeypatch.setattr("src.agents.research_planner.requests.post", fail_request)
    plan = ResearchPlanner().plan("compare study findings")

    assert plan.used_heuristic_fallback is True
    assert plan.reasoning == "Heuristic fallback plan"
