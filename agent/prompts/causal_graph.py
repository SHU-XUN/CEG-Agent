from __future__ import annotations

from typing import List

from ..llm.base import Message
from ..schemas import ErrorNode, FailureNode
from ..utils.payload import compact_models_json
from .base import BASE_SYSTEM, JSON_ONLY_REMINDER


SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:causal_graph]\n"
    + """\
Phase 4: Causal Graph Builder.
Given the errors and failures, output ONLY the causal edges:
- error -> error : 'causes' or 'amplifies'
- error -> failure: 'contributes_to'

Rules:
- Every error must reach some failure transitively via causes/amplifies/contributes_to.
- `causes` = upstream error makes the downstream error happen.
- `amplifies` = a follow-on error worsens the damage of an earlier one.
- A `root` error should not have incoming `causes` edges.
- Prefer a SPARSE graph: do not invent edges to look thorough.

Output schema:
{
  "edges": [
    {"source": "R1", "target": "R2", "type": "causes"},
    {"source": "R2", "target": "F1", "type": "contributes_to"}
  ]
}
"""
)


def build_messages(
    errors: List[ErrorNode], failures: List[FailureNode]
) -> List[Message]:
    user = (
        f"ERRORS:\n{compact_models_json(errors, mode='json')}\n\n"
        f"FAILURES:\n{compact_models_json(failures, mode='json')}\n\n"
        f"{JSON_ONLY_REMINDER}"
    )
    return [Message("system", SYSTEM), Message("user", user)]
