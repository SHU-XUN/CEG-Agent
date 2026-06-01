from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..critic import auto_repair, validate_graph
from ..prompts import graph_critic as P
from ..schemas import (
    AnomalyNode,
    CausalErrorGraph,
    Edge,
    EdgeType,
    ErrorNode,
    FailureNode,
    Mechanism,
    RepairValue,
    Role,
)
from .base import AgentMeta, BaseAgent


class GraphCritic(BaseAgent):
    meta = AgentMeta(name="graph_critic", phase=6)

    def __init__(self, llm, rounds: int = 1, max_failures: int = 3, **kwargs):
        super().__init__(llm, **kwargs)
        self.rounds = max(0, int(rounds))
        self.max_failures = max_failures

    def run(self, state) -> None:
        graph = state.to_graph()
        all_issues: List[str] = []
        graph, det_log = auto_repair(graph)
        all_issues.extend([f"auto_repair: {x}" for x in det_log])

        for round_idx in range(self.rounds):
            issues = validate_graph(graph, max_failures=self.max_failures)
            issue_strs = [str(i) for i in issues]
            if not issue_strs:
                self.log.info("phase 6 round %d: clean", round_idx + 1)
                break
            self.log.info(
                "phase 6 round %d: %d issues, calling critic",
                round_idx + 1,
                len(issue_strs),
            )
            patch = self.llm.chat_json(
                P.build_messages(graph.to_output_dict(), issue_strs)
            )
            graph = apply_patch(graph, patch.get("patch") or {})
            graph, det_log2 = auto_repair(graph)
            all_issues.extend(patch.get("issues", []))
            all_issues.extend([f"auto_repair: {x}" for x in det_log2])

        graph, det_log3 = auto_repair(graph)
        all_issues.extend([f"auto_repair: {x}" for x in det_log3])

        state.load_from_graph(graph)
        state.critic_log.extend(all_issues)



def apply_patch(graph: CausalErrorGraph, patch: Dict[str, Any]) -> CausalErrorGraph:
    g = graph.model_copy(deep=True)
    if not patch:
        return g
    if not isinstance(patch, dict):
        return g

    rm_err = set(patch.get("remove_errors") or [])
    rm_fail = set(patch.get("remove_failures") or [])
    rm_anom = set(patch.get("remove_anomalies") or [])
    if rm_err:
        g.errors = [r for r in g.errors if r.id not in rm_err]
    if rm_fail:
        g.failures = [f for f in g.failures if f.id not in rm_fail]
    if rm_anom:
        g.anomaly = [a for a in g.anomaly if a.id not in rm_anom]

    surviving_ids = g.node_ids()
    g.edges = [
        e for e in g.edges if e.source in surviving_ids and e.target in surviving_ids
    ]

    for spec in patch.get("remove_edges") or []:
        try:
            src = spec["source"]
            tgt = spec["target"]
            typ = spec["type"]
        except (KeyError, TypeError):
            continue
        g.edges = [
            e
            for e in g.edges
            if not (e.source == src and e.target == tgt and e.type.value == typ)
        ]

    surviving_ids = g.node_ids()
    for spec in patch.get("add_edges") or []:
        try:
            e = Edge.model_validate(spec)
        except (ValidationError, ValueError, KeyError):
            continue
        if e.source in surviving_ids and e.target in surviving_ids:
            g.edges.append(e)

    _apply_updates(g.errors, patch.get("update_errors"), ErrorNode)
    _apply_updates(g.failures, patch.get("update_failures"), FailureNode)
    _apply_updates(g.anomaly, patch.get("update_anomalies"), AnomalyNode)

    for spec in patch.get("merge_errors") or []:
        keep_id = spec.get("keep_id")
        drop_ids = set(spec.get("drop_ids") or [])
        if not keep_id or not drop_ids:
            continue
        keep_id_set = {keep_id}
        new_edges: List[Edge] = []
        seen: set[tuple] = set()
        for e in g.edges:
            src = keep_id if e.source in drop_ids else e.source
            tgt = keep_id if e.target in drop_ids else e.target
            if src == tgt and e.type != EdgeType.EVENT_NEXT:
                continue
            key = (src, tgt, e.type.value)
            if key in seen:
                continue
            seen.add(key)
            try:
                new_edges.append(Edge(source=src, target=tgt, type=e.type))
            except ValueError:
                continue
        g.edges = new_edges
        g.errors = [r for r in g.errors if r.id not in drop_ids or r.id in keep_id_set]

    return g


def _apply_updates(items, updates, _cls):
    if not updates:
        return
    by_id = {it.id: it for it in items}
    for u in updates:
        if not isinstance(u, dict):
            continue
        target = by_id.get(u.get("id"))
        if target is None:
            continue
        for k, v in u.items():
            if k == "id":
                continue
            if not hasattr(target, k):
                continue
            try:
                if k == "mechanism":
                    v = Mechanism(v)
                elif k == "role":
                    v = Role(v)
                elif k == "repair_value" and v is not None:
                    v = RepairValue(v)
                setattr(target, k, v)
            except (ValueError, TypeError):
                continue
