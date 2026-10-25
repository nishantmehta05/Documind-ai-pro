"""LLM provider adapters used by DocuMind's existing prompt pipelines."""

from .provider import LLMProvider, ModelProviderError

__all__ = ["LLMProvider", "ModelProviderError"]
