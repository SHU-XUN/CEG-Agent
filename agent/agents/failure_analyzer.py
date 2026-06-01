from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..prompts import failure_analyzer as P
from ..schemas import FailureNode, FailureType
from .base import AgentMeta, BaseAgent


class FailureAnalyzer(BaseAgent):
    meta = AgentMeta(name="failure_analyzer", phase=2)

    def run(self, state) -> None:
        messages = P.build_messages(state.task or "", state.events)
        data = self.llm.chat_json(messages)
        failures = _parse_failures(data.get("failures", []))
        if not failures:
            failures = [
                FailureNode(
                    id="F1",
                    type=FailureType.OTHER,
                    description="(fallback) failure analyzer returned no failures.",
                )
            ]
        state.failures = failures
        self.log.info("phase 2: identified %d failure(s)", len(failures))


def _parse_failures(raw: List[Dict[str, Any]]) -> List[FailureNode]:
    out: List[FailureNode] = []
    for i, item in enumerate(raw, start=1):
        try:
            item.setdefault("id", f"F{i}")
            out.append(FailureNode.model_validate(item))
        except ValidationError:
            continue
    return out
