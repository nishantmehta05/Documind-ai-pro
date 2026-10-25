"""Tests for the configured local and hosted model transports."""

import pytest
import requests

from src.config import AppConfig
from src.llm.provider import LLMProvider, ModelProviderError


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(response=self)
            raise error

    def json(self):
        return self.payload


def test_local_ollama_status_and_generation(monkeypatch):
    calls = {}

    def fake_get(url, **kwargs):
        calls["get_url"] = url
        return FakeResponse({"models": [{"name": "qwen2.5:7b"}]})

    def fake_post(url, **kwargs):
        calls["post_url"] = url
        calls["payload"] = kwargs["json"]
        return FakeResponse({"response": "Grounded answer [1]."})

    monkeypatch.setattr("src.llm.provider.requests.get", fake_get)
    monkeypatch.setattr("src.llm.provider.requests.post", fake_post)
    provider = LLMProvider(
        provider="ollama",
        model_name="qwen2.5:7b",
        base_url="http://localhost:11434",
    )

    assert provider.check_status() == (True, True)
    result = provider.generate(
        prompt="Question with evidence",
        system="Ground answers only",
        options={"temperature": 0.1, "num_predict": 128},
    )

    assert result == "Grounded answer [1]."
    assert calls["get_url"] == "http://localhost:11434/api/tags"
    assert calls["post_url"] == "http://localhost:11434/api/generate"
    assert calls["payload"]["model"] == "qwen2.5:7b"
    assert calls["payload"]["system"] == "Ground answers only"
    assert calls["payload"]["prompt"] == "Question with evidence"


def test_hosted_openai_compatible_status_and_generation(monkeypatch):
    calls = {}

    def fake_get(url, **kwargs):
        calls["get_url"] = url
        calls["get_headers"] = kwargs["headers"]
        return FakeResponse({"data": [{"id": "hosted-test-model"}]})

    def fake_post(url, **kwargs):
        calls["post_url"] = url
        calls["post_headers"] = kwargs["headers"]
        calls["payload"] = kwargs["json"]
        return FakeResponse(
            {"choices": [{"message": {"content": "Hosted answer [1]."}}]}
        )

    monkeypatch.setattr("src.llm.provider.requests.get", fake_get)
    monkeypatch.setattr("src.llm.provider.requests.post", fake_post)
    provider = LLMProvider(
        provider="hosted",
        model_name="hosted-test-model",
        base_url="https://llm.example.test/v1",
        api_key="test-secret-do-not-log",
    )

    assert provider.check_status() == (True, True)
    result = provider.generate(
        prompt="Question with evidence",
        system="Ground answers only",
        options={"temperature": 0.1, "top_p": 0.9, "num_ctx": 4096, "num_predict": 128},
    )

    assert result == "Hosted answer [1]."
    assert calls["get_url"] == "https://llm.example.test/v1/models"
    assert calls["post_url"] == "https://llm.example.test/v1/chat/completions"
    assert calls["get_headers"]["Authorization"] == "Bearer test-secret-do-not-log"
    assert calls["post_headers"]["Authorization"] == "Bearer test-secret-do-not-log"
    assert calls["payload"]["messages"] == [
        {"role": "system", "content": "Ground answers only"},
        {"role": "user", "content": "Question with evidence"},
    ]
    assert calls["payload"]["max_tokens"] == 128
    assert "num_ctx" not in calls["payload"]


def test_missing_hosted_credentials_report_safe_configuration_error():
    provider = LLMProvider(
        provider="hosted",
        model_name="hosted-test-model",
        base_url="https://llm.example.test/v1",
        api_key="",
    )

    assert provider.check_status() == (False, False)
    assert "LLM_API_KEY" in provider.last_error
    with pytest.raises(ModelProviderError, match="LLM_API_KEY") as error:
        provider.generate("Question")
    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("base_url", "model_name", "missing_setting"),
    [
        ("", "hosted-test-model", "LLM_API_BASE_URL"),
        ("https://llm.example.test/v1", "", "LLM_MODEL"),
    ],
)
def test_missing_hosted_endpoint_or_model_is_reported(base_url, model_name, missing_setting):
    provider = LLMProvider(
        provider="hosted",
        model_name=model_name,
        base_url=base_url,
        api_key="test-secret-do-not-log",
    )

    assert provider.check_status() == (False, False)
    assert missing_setting in provider.last_error
    with pytest.raises(ModelProviderError) as error:
        provider.generate("Question")
    assert missing_setting in str(error.value)
    assert "test-secret-do-not-log" not in str(error.value)


def test_unavailable_local_and_hosted_services_are_not_fabricated(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("private transport detail")

    monkeypatch.setattr("src.llm.provider.requests.get", unavailable)
    monkeypatch.setattr("src.llm.provider.requests.post", unavailable)

    local = LLMProvider(
        provider="ollama", model_name="qwen2.5:7b", base_url="http://localhost:11434"
    )
    assert local.check_status() == (False, False)
    assert "Could not connect to local Ollama" in local.last_error
    with pytest.raises(ModelProviderError, match="Could not connect to local Ollama"):
        local.generate("Question")

    hosted = LLMProvider(
        provider="hosted",
        model_name="hosted-test-model",
        base_url="https://llm.example.test/v1",
        api_key="test-secret-do-not-log",
    )
    assert hosted.check_status() == (False, False)
    assert "Could not connect to the hosted model endpoint" in hosted.last_error
    with pytest.raises(ModelProviderError, match="Could not connect to the hosted model endpoint") as error:
        hosted.generate("Question")
    assert "test-secret-do-not-log" not in str(error.value)


def test_hosted_auth_failure_is_clear_and_does_not_include_response_body(monkeypatch):
    monkeypatch.setattr(
        "src.llm.provider.requests.post",
        lambda *args, **kwargs: FakeResponse({"error": "test-secret-do-not-log"}, 401),
    )
    provider = LLMProvider(
        provider="hosted",
        model_name="hosted-test-model",
        base_url="https://llm.example.test/v1",
        api_key="test-secret-do-not-log",
    )

    with pytest.raises(ModelProviderError, match="authentication failed") as error:
        provider.generate("Question")
    assert "test-secret-do-not-log" not in str(error.value)


def test_api_key_is_not_in_config_repr():
    settings = AppConfig(hosted_llm_api_key="test-secret-do-not-log")
    assert "test-secret-do-not-log" not in repr(settings)
