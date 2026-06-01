from __future__ import annotations

from typing import List

from ..schemas import EventNode
from .base import AgentMeta, BaseAgent


class EventBuilder(BaseAgent):
    meta = AgentMeta(name="event_builder", phase=1)

    def run(self, state) -> None:
        if not state.turns:
            raise RuntimeError("EventBuilder requires state.turns")
        state.events = _events_from_turns(state.turns)
        self.log.info("phase 1: built %d events (1:1 from turns)", len(state.events))


def _events_from_turns(turns) -> List[EventNode]:
    return [
        EventNode(
            id=f"E{idx}",
            thought=t.thought or "(inferred from action)",
            action=t.action,
            observation=t.observation,
        )
        for idx, t in enumerate(turns, start=1)
    ]
