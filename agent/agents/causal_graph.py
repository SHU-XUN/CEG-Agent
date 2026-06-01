from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..prompts import causal_graph as P
from ..schemas import Edge, EdgeType
from .base import AgentMeta, BaseAgent


class CausalGraphBuilder(BaseAgent):
    meta = AgentMeta(name="causal_graph", phase=4)

    def run(self, state) -> None:
        edges: List[Edge] = []

        for a, b in zip(state.events, state.events[1:]):
            edges.append(Edge(source=a.id, target=b.id, type=EdgeType.EVENT_NEXT))

        for r in state.errors:
            edges.append(
                Edge(source=r.id, target=r.event_id, type=EdgeType.ATTACHED_TO)
            )

        if state.errors and state.failures:
            messages = P.build_messages(state.errors, state.failures)
            data = self.llm.chat_json(messages)
            edges.extend(
                _parse_edges(data.get("edges", []), state.error_ids(), state.failure_ids())
            )

        edges = _ensure_errors_reach_failure(edges, state)

        state.edges = edges
        self.log.info("phase 4: built %d edges", len(edges))


def _parse_edges(
    raw: List[Dict[str, Any]], error_ids: set[str], failure_ids: set[str]
) -> List[Edge]:
    out: List[Edge] = []
    for item in raw:
        try:
            e = Edge.model_validate(item)
        except (ValidationError, ValueError, KeyError):
            continue
        if e.type == EdgeType.CONTRIBUTES_TO:
            if e.source not in error_ids or e.target not in failure_ids:
                continue
        elif e.type in {EdgeType.CAUSES, EdgeType.AMPLIFIES}:
            if e.source not in error_ids or e.target not in error_ids:
                continue
        else:
            continue
        out.append(e)
    return out


def _ensure_errors_reach_failure(edges: List[Edge], state) -> List[Edge]:
    if not state.errors or not state.failures:
        return edges
    adj: Dict[str, List[str]] = {r.id: [] for r in state.errors}
    for e in edges:
        if e.type in {EdgeType.CAUSES, EdgeType.AMPLIFIES, EdgeType.CONTRIBUTES_TO}:
            adj.setdefault(e.source, []).append(e.target)
    failure_ids = state.failure_ids()
    primary_failure = state.failures[0].id

    for r in state.errors:
        if _reaches(adj, r.id, failure_ids):
            continue
        edges.append(
            Edge(source=r.id, target=primary_failure, type=EdgeType.CONTRIBUTES_TO)
        )
        adj.setdefault(r.id, []).append(primary_failure)
    return edges


def _reaches(adj: Dict[str, List[str]], start: str, targets: set[str]) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in targets:
            return True
        stack.extend(adj.get(node, []))
    return False
