from __future__ import annotations

from typing import List

from ..llm.base import Message
from ..schemas import ErrorNode, EventNode, FailureNode
from ..utils.payload import compact_events_json, compact_models_json
from .base import BASE_SYSTEM, JSON_ONLY_REMINDER


SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:anomaly_detector]\n"
    + """\
Phase 5: Anomaly Detector.
Identify HARMLESS anomalies — events that look unusual / inefficient /
unelegant but do NOT causally contribute to any failure.

Rules:
- An anomaly must NOT correspond to any existing error.
- An anomaly may reference multiple events (event_ids).
- If nothing qualifies, return an empty array.
- Use ids A1, A2, ... contiguously.

Output schema:
{
  "anomaly": [
    {"id": "A1",
     "event_ids": ["E1", "E2"],
     "summary": "...",
     "description": "..."}
  ]
}
"""
)


def build_messages(
    events: List[EventNode],
    errors: List[ErrorNode],
    failures: List[FailureNode],
) -> List[Message]:
    user = (
        f"EVENTS:\n{compact_events_json(events)}\n\n"
        f"ERRORS:\n{compact_models_json(errors, mode='json')}\n\n"
        f"FAILURES:\n{compact_models_json(failures, mode='json')}\n\n"
        f"{JSON_ONLY_REMINDER}"
    )
    return [Message("system", SYSTEM), Message("user", user)]
