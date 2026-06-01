from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict

from ..llm import LLMClient
from ..utils.logging import get_logger


@dataclass
class AgentMeta:
    name: str
    phase: int


class BaseAgent(abc.ABC):
    meta: AgentMeta

    def __init__(self, llm: LLMClient, **kwargs: Any):
        self.llm = llm
        self.opts: Dict[str, Any] = kwargs
        self.log = get_logger(f"agent.{self.meta.name}")

    @abc.abstractmethod
    def run(self, state: "PipelineState") -> None:
