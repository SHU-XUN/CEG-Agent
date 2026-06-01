from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..prompts import anomaly_detector as P
from ..schemas import AnomalyNode
from .base import AgentMeta, BaseAgent


class AnomalyDetector(BaseAgent):
    meta = AgentMeta(name="anomaly_detector", phase=5)

    def run(self, state) -> None:
        messages = P.build_messages(state.events, state.errors, state.failures)
        data = self.llm.chat_json(messages)
        anomalies = _parse_anomalies(data.get("anomaly", []), state.event_ids())
        state.anomaly = anomalies
        self.log.info("phase 5: detected %d anomalies", len(anomalies))


def _parse_anomalies(
    raw: List[Dict[str, Any]], known_event_ids: set[str]
) -> List[AnomalyNode]:
    out: List[AnomalyNode] = []
    for i, item in enumerate(raw, start=1):
        try:
            item.setdefault("id", f"A{i}")
            item["event_ids"] = [
                eid for eid in (item.get("event_ids") or []) if eid in known_event_ids
            ]
            out.append(AnomalyNode.model_validate(item))
        except ValidationError:
            continue
    return out
