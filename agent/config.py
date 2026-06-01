from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: os.getenv("CE_LLM_BACKEND", "mock"))
    model: str = field(default_factory=lambda: os.getenv("CE_LLM_MODEL", "gpt-4o-mini"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("CE_LLM_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("CE_LLM_BASE_URL"))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("CE_LLM_TEMPERATURE", "1.0"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("CE_MAX_RETRIES", "2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("CE_LLM_MAX_TOKENS", "-1"))
    )


@dataclass
class PipelineConfig:
    critic_rounds: int = field(
        default_factory=lambda: int(os.getenv("CE_CRITIC_ROUNDS", "2"))
    )
    enable_repair: bool = field(
        default_factory=lambda: _env_bool("CE_ENABLE_REPAIR", True)
    )
    max_failures: int = 3
    log_level: str = field(default_factory=lambda: os.getenv("CE_LOG_LEVEL", "INFO"))


@dataclass
class FrameworkConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config() -> FrameworkConfig:
    return FrameworkConfig()
