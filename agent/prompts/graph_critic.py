from __future__ import annotations

from typing import Any, Dict, List

from ..llm.base import Message
from ..utils.payload import compact_json
from .base import BASE_SYSTEM, JSON_ONLY_REMINDER


SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:graph_critic]\n"
    + """\
Phase 6: Graph Critic.
You are given a draft Causal Error Graph and a list of issues raised by
deterministic validators. Produce a minimal JSON patch that fixes them.

Issues you must consider EVEN IF NOT FLAGGED:
1. Anomalies confused with errors (or vice versa).
2. Duplicated failures — two failures that describe the SAME outcome
   in different words. Collapse them. Genuinely distinct outcomes
   (e.g. wrong answer + violated safety rule) stay separate.
3. role does not match graph (root with incoming causes, propagated with none).
4. Errors missing `attached_to` an event.
5. Illegal edge types or endpoints.
6. Duplicate / highly redundant nodes (collapse them).
7. Symptoms mistakenly labeled `root`.

NOTE: the patch language can REMOVE / UPDATE / MERGE / RE-WIRE existing
nodes and edges. It CANNOT add new errors, failures, or anomalies. If
the graph is missing a key error you cannot patch one in here — flag it
in the `issues` list for a human reviewer; the orchestrator will note it
in the bundle's run.log.
8. Obviously missing key errors that the deterministic pass cannot see.

Output schema (any keys may be omitted):
{
  "issues": ["short human-readable issue strings"],
  "patch": {
    "remove_errors":   ["R3"],
    "remove_failures": [],
    "remove_anomalies":[],
    "remove_edges":    [{"source":"R1","target":"R2","type":"causes"}],
    "add_edges":       [{"source":"R1","target":"E1","type":"attached_to"}],
    "update_errors":   [{"id":"R2","role":"propagated"}],
    "update_failures": [],
    "update_anomalies":[],
    "merge_errors":    [{"keep_id":"R1","drop_ids":["R4"]}]
  }
}
If nothing should change, return {"issues": [], "patch": {}}.
"""
)


def build_messages(graph_dict: Dict[str, Any], issues: List[str]) -> List[Message]:
    user = (
        f"DRAFT_GRAPH:\n{compact_json(graph_dict)}\n\n"
        f"VALIDATOR_ISSUES:\n{compact_json(issues)}\n\n"
        f"{JSON_ONLY_REMINDER}"
    )
    return [Message("system", SYSTEM), Message("user", user)]
