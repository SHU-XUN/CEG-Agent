from __future__ import annotations

from typing import Any, Dict

from ..prompts import repair_estimator as P
from ..schemas import RepairValue
from .base import AgentMeta, BaseAgent


class RepairValueEstimator(BaseAgent):
    meta = AgentMeta(name="repair_estimator", phase=7)

    def run(self, state) -> None:
        if not state.errors or not state.failures:
            self.log.info("phase 7: skipped (no errors or failures)")
            return
        messages = P.build_messages(state.errors, state.failures, state.edges)
        data = self.llm.chat_json(messages)
        values: Dict[str, str] = data.get("repair_values") or {}
        rationale: Dict[str, str] = data.get("rationale") or {}
        applied = 0
        for r in state.errors:
            v = values.get(r.id)
            if not v:
                continue
            try:
                r.repair_value = RepairValue(v)
                applied += 1
            except ValueError:
                continue
        if rationale:
            for rid, why in rationale.items():
                state.critic_log.append(f"repair_value/{rid}: {why}")
        self.log.info(
            "phase 7: assigned repair_value for %d/%d errors", applied, len(state.errors)
        )
