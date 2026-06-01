from __future__ import annotations

from ..config import LLMConfig
from .base import LLMClient
from .mock_client import MockLLMClient


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    backend = (cfg.backend or "mock").lower()
    if backend == "mock":
        return MockLLMClient()
    if backend in {"openai", "azure", "openai-compatible"}:
        from .openai_client import OpenAILLMClient

        return OpenAILLMClient(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )
    raise ValueError(
        f"Unknown LLM backend '{cfg.backend}'. "
        "Supported: 'mock', 'openai', 'azure'."
    )
