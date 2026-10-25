"""Safe transport adapter for local Ollama and hosted OpenAI-compatible APIs."""

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from ..config import config


class ModelProviderError(requests.exceptions.RequestException):
    """An actionable provider error that never includes credentials or response bodies."""


class LLMProvider:
    """Send existing prompts to Ollama or a hosted OpenAI-compatible endpoint."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.provider = (provider or config.llm_provider or "ollama").strip().lower()
        if self.provider not in {"ollama", "hosted"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'hosted'.")

        if self.provider == "ollama":
            self.base_url = (base_url or config.ollama_base_url).strip().rstrip("/")
            self.model_name = (model_name or config.ollama_model).strip()
            self.api_key = None
        else:
            self.base_url = (base_url or config.hosted_llm_base_url).strip().rstrip("/")
            self.model_name = (model_name or config.hosted_llm_model).strip()
            key = api_key if api_key is not None else config.hosted_llm_api_key
            self.api_key = key.strip() if key else ""

        self.timeout = timeout or config.timeout_seconds
        self.last_error: Optional[str] = None

    def _validate_hosted_settings(self) -> Optional[str]:
        if not self.base_url:
            return "Hosted model URL is missing. Set LLM_API_BASE_URL in Streamlit secrets or the environment."
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            return "LLM_API_BASE_URL must be an http(s) endpoint URL without embedded credentials, query parameters, or fragments."
        if not self.model_name:
            return "Hosted model name is missing. Set LLM_MODEL in Streamlit secrets or the environment."
        if not self.api_key:
            return "Hosted model credentials are missing. Set LLM_API_KEY in Streamlit secrets or the environment."
        return None

    def _hosted_chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _hosted_models_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return f"{self.base_url[:-len('/chat/completions')]}/models"
        return f"{self.base_url}/models"

    @staticmethod
    def _hosted_http_error(status_code: int) -> str:
        if status_code in {401, 403}:
            return "Hosted model authentication failed. Verify LLM_API_KEY in Streamlit secrets or the environment."
        if status_code == 404:
            return "The hosted model endpoint was not found. Verify LLM_API_BASE_URL and that it supports OpenAI-compatible endpoints."
        if status_code == 429:
            return "The hosted model provider is rate-limiting requests. Check the account quota and try again."
        return f"The hosted model provider returned HTTP {status_code}. Verify the endpoint and provider configuration."

    def check_status(self) -> Tuple[bool, bool]:
        """Return service reachability and whether the selected model is available."""
        self.last_error = None
        if self.provider == "hosted":
            issue = self._validate_hosted_settings()
            if issue:
                self.last_error = issue
                return False, False

            try:
                response = requests.get(
                    self._hosted_models_url(),
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=min(self.timeout, 10),
                )
                if response.status_code >= 400:
                    self.last_error = self._hosted_http_error(response.status_code)
                    return False, False
                data = response.json()
                models = data.get("data", []) if isinstance(data, dict) else []
                available = any(
                    isinstance(item, dict) and item.get("id") == self.model_name
                    for item in models
                )
                if not available:
                    self.last_error = (
                        "The hosted endpoint is reachable, but the configured LLM_MODEL "
                        "was not listed by the provider. Check the model name and endpoint."
                    )
                return True, available
            except requests.exceptions.Timeout:
                self.last_error = "The hosted model endpoint timed out. Verify provider availability and try again."
            except requests.exceptions.ConnectionError:
                self.last_error = "Could not connect to the hosted model endpoint. Verify LLM_API_BASE_URL and provider availability."
            except (requests.exceptions.RequestException, ValueError, AttributeError, TypeError):
                self.last_error = "The hosted model endpoint returned an invalid status response. Verify its OpenAI-compatible API configuration."
            return False, False

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout, 5))
            response.raise_for_status()
            models = response.json().get("models", [])
            available = any(
                model.get("name") == self.model_name or model.get("model") == self.model_name
                for model in models
                if isinstance(model, dict)
            )
            if not available:
                self.last_error = (
                    f"Ollama is reachable, but model '{self.model_name}' is not installed. "
                    f"Run `ollama pull {self.model_name}`."
                )
            return True, available
        except requests.exceptions.Timeout:
            self.last_error = "The local Ollama service timed out. Verify OLLAMA_BASE_URL and that Ollama is running."
        except requests.exceptions.ConnectionError:
            self.last_error = "Could not connect to local Ollama. Verify OLLAMA_BASE_URL and that Ollama is running."
        except requests.exceptions.HTTPError as error:
            status_code = getattr(error.response, "status_code", 0)
            self.last_error = f"Local Ollama returned HTTP {status_code}. Verify the service configuration."
        except (requests.exceptions.RequestException, ValueError, AttributeError, TypeError):
            self.last_error = "Local Ollama returned an invalid status response. Verify the service configuration."
        return False, False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Generate a response without changing prompt text or exposing credentials."""
        options = options or {}
        request_timeout = timeout or self.timeout

        if self.provider == "ollama":
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            if system is not None:
                payload["system"] = system
            url = f"{self.base_url}/api/generate"
            headers: Dict[str, str] = {}
        else:
            issue = self._validate_hosted_settings()
            if issue:
                raise ModelProviderError(issue)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            }
            for option_name in ("temperature", "top_p"):
                if option_name in options:
                    payload[option_name] = options[option_name]
            if "num_predict" in options:
                payload["max_tokens"] = options["num_predict"]
            url = self._hosted_chat_url()
            headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            request_kwargs: Dict[str, Any] = {
                "json": payload,
                "timeout": request_timeout,
            }
            if headers:
                request_kwargs["headers"] = headers
            response = requests.post(url, **request_kwargs)
            status_code = getattr(response, "status_code", 200)
            if status_code >= 400:
                if self.provider == "hosted":
                    raise ModelProviderError(self._hosted_http_error(status_code))
                if status_code == 404:
                    raise ModelProviderError(
                        f"Ollama model '{self.model_name}' was not found. Run `ollama pull {self.model_name}`."
                    )
                raise ModelProviderError(
                    f"Local Ollama returned HTTP {status_code}. Verify the service and model configuration."
                )
            data = response.json()
            if self.provider == "ollama":
                result = data.get("response", "")
            else:
                result = data["choices"][0]["message"]["content"]
            if not isinstance(result, str) or not result.strip():
                raise ModelProviderError("The model provider returned an empty or invalid response.")
            return result.strip()
        except ModelProviderError:
            raise
        except requests.exceptions.Timeout as error:
            if self.provider == "hosted":
                message = "The hosted model request timed out. Increase the request timeout or check provider availability."
            else:
                message = "The local Ollama request timed out. Verify the service and try again."
            raise ModelProviderError(message) from None
        except requests.exceptions.ConnectionError as error:
            if self.provider == "hosted":
                message = "Could not connect to the hosted model endpoint. Verify LLM_API_BASE_URL and provider availability."
            else:
                message = "Could not connect to local Ollama. Verify OLLAMA_BASE_URL and that Ollama is running."
            raise ModelProviderError(message) from None
        except requests.exceptions.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else 0
            if self.provider == "hosted":
                message = self._hosted_http_error(status_code)
            elif status_code == 404:
                message = f"Ollama model '{self.model_name}' was not found. Run `ollama pull {self.model_name}`."
            else:
                message = f"Local Ollama returned HTTP {status_code}. Verify the service and model configuration."
            raise ModelProviderError(message) from None
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            message = (
                "The hosted model provider returned an invalid response. Verify that the endpoint uses the OpenAI chat completions format."
                if self.provider == "hosted"
                else "Local Ollama returned an invalid response. Verify the model service configuration."
            )
            raise ModelProviderError(message) from None
