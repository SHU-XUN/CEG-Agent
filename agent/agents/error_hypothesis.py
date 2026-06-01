from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..prompts import error_hypothesis as P
from ..schemas import ErrorNode
from ..utils.logging import get_logger
from .base import AgentMeta, BaseAgent


_log = get_logger("agent.error_hypothesis")


class ErrorHypothesisGenerator(BaseAgent):
    meta = AgentMeta(name="error_hypothesis", phase=3)

    def run(self, state) -> None:
        messages = P.build_messages(
            state.task or "", state.events, state.failures
        )
        data = self.llm.chat_json(messages)
        errors = _parse_errors(data.get("errors", []), state.event_ids())
        state.errors = errors
        self.log.info("phase 3: generated %d error hypotheses", len(errors))


def _parse_errors(
    raw: List[Dict[str, Any]], known_event_ids: set[str]
) -> List[ErrorNode]:
    out: List[ErrorNode] = []
    n_dropped_unknown_event = 0
    n_dropped_validation = 0
    for i, item in enumerate(raw, start=1):
        try:
            item.setdefault("id", f"R{i}")
            if item.get("event_id") not in known_event_ids:
                n_dropped_unknown_event += 1
                _log.warning(
                    "dropping error %s: event_id=%r not in events %s",
                    item.get("id"),
                    item.get("event_id"),
                    sorted(known_event_ids),
                )
                continue
            out.append(ErrorNode.model_validate(item))
        except ValidationError as e:
            n_dropped_validation += 1
            _log.warning(
                "dropping error %s: schema validation failed: %s",
                item.get("id"),
                str(e).splitlines()[0] if str(e) else "?",
            )
            continue
    if n_dropped_unknown_event or n_dropped_validation:
        _log.warning(
            "phase 3: dropped %d unknown_event_id + %d validation errors "
            "(of %d candidates); kept %d",
            n_dropped_unknown_event,
            n_dropped_validation,
            len(raw),
            len(out),
        )
    return out
