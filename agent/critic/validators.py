from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from ..schemas import (
    CausalErrorGraph,
    EdgeType,
    LEGAL_EDGE_ENDPOINTS,
    Role,
)


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def validate_graph(
    graph: CausalErrorGraph, *, max_failures: int = 3
) -> List[Issue]:
    issues: List[Issue] = []
    issues.extend(_validate_edge_endpoints(graph))
    issues.extend(_validate_node_references(graph))
    issues.extend(_validate_failure_count(graph, max_failures))
    issues.extend(_validate_errors_attached(graph))
    issues.extend(_validate_roles_consistent(graph))
    issues.extend(_validate_anomaly_disjoint_from_errors(graph))
    issues.extend(_validate_errors_reach_failure(graph))
    issues.extend(_validate_no_duplicate_edges(graph))
    return issues



def _validate_edge_endpoints(graph: CausalErrorGraph) -> List[Issue]:
    out: List[Issue] = []
    for e in graph.edges:
        src_p, tgt_p = e.source[:1], e.target[:1]
        expected = LEGAL_EDGE_ENDPOINTS.get(e.type)
        if expected is None:
            out.append(
                Issue(
                    "edge.unknown_type",
                    f"unknown edge type '{e.type}' on {e.source}->{e.target}",
                    "error",
                )
            )
            continue
        if (src_p, tgt_p) != expected:
            out.append(
                Issue(
                    "edge.bad_endpoints",
                    f"edge {e.source} -{e.type.value}-> {e.target} expects "
                    f"prefixes {expected} but got ({src_p},{tgt_p})",
                    "error",
                )
            )
    return out


def _validate_node_references(graph: CausalErrorGraph) -> List[Issue]:
    out: List[Issue] = []
    known = graph.node_ids()
    for e in graph.edges:
        if e.source not in known:
            out.append(
                Issue("edge.dangling_source", f"edge source '{e.source}' missing", "error")
            )
        if e.target not in known:
            out.append(
                Issue("edge.dangling_target", f"edge target '{e.target}' missing", "error")
            )
    event_ids = graph.event_ids()
    for r in graph.errors:
        if r.event_id not in event_ids:
            out.append(
                Issue(
                    "error.bad_event_id",
                    f"error {r.id} attached to unknown event {r.event_id}",
                    "error",
                )
            )
    for a in graph.anomaly:
        for eid in a.event_ids:
            if eid not in event_ids:
                out.append(
                    Issue(
                        "anomaly.bad_event_id",
                        f"anomaly {a.id} references unknown event {eid}",
                        "error",
                    )
                )
    return out


def _validate_failure_count(graph: CausalErrorGraph, limit: int) -> List[Issue]:
    if len(graph.failures) > limit:
        return [
            Issue(
                "failure.too_many",
                f"{len(graph.failures)} failures exceed soft cap {limit}; "
                "collapse symptoms into the deepest task-level failure.",
                "warning",
            )
        ]
    if not graph.failures:
        return [
            Issue(
                "failure.missing",
                "no failures emitted — pipeline must always produce ≥ 1 failure.",
                "error",
            )
        ]
    return []


def _validate_errors_attached(graph: CausalErrorGraph) -> List[Issue]:
    out: List[Issue] = []
    attached: Set[str] = set()
    for e in graph.edges:
        if e.type == EdgeType.ATTACHED_TO:
            attached.add(e.source)
    for r in graph.errors:
        if r.id not in attached:
            out.append(
                Issue(
                    "error.missing_attached_to",
                    f"error {r.id} has no attached_to edge to an event",
                    "error",
                )
            )
    return out


def _validate_roles_consistent(graph: CausalErrorGraph) -> List[Issue]:
    out: List[Issue] = []
    incoming_causes: dict[str, int] = {r.id: 0 for r in graph.errors}
    participates_amplifies: dict[str, bool] = {r.id: False for r in graph.errors}
    for e in graph.edges:
        if e.type == EdgeType.CAUSES and e.target in incoming_causes:
            incoming_causes[e.target] += 1
        if e.type == EdgeType.AMPLIFIES:
            if e.source in participates_amplifies:
                participates_amplifies[e.source] = True
            if e.target in participates_amplifies:
                participates_amplifies[e.target] = True
    for r in graph.errors:
        if r.role == Role.ROOT and incoming_causes.get(r.id, 0) > 0:
            out.append(
                Issue(
                    "role.root_has_incoming_causes",
                    f"error {r.id} is 'root' but has incoming causes edges",
                    "warning",
                )
            )
        if r.role == Role.PROPAGATED and incoming_causes.get(r.id, 0) == 0:
            out.append(
                Issue(
                    "role.propagated_no_upstream",
                    f"error {r.id} is 'propagated' but has no upstream causes",
                    "warning",
                )
            )
        if r.role == Role.AMPLIFICATION and not participates_amplifies.get(r.id, False):
            out.append(
                Issue(
                    "role.amplification_no_amplifies_edge",
                    f"error {r.id} is 'amplification' but participates in no "
                    "amplifies edge",
                    "warning",
                )
            )
    return out


def _validate_anomaly_disjoint_from_errors(graph: CausalErrorGraph) -> List[Issue]:
    out: List[Issue] = []
    err_events: Set[str] = {r.event_id for r in graph.errors}
    err_keys: Set[str] = {
        r.event_id + "::" + r.description.strip().lower()
        for r in graph.errors
    }
    for a in graph.anomaly:
        for eid in a.event_ids:
            if eid in err_events:
                key = eid + "::" + a.description.strip().lower()
                if key in err_keys:
                    out.append(
                        Issue(
                            "anomaly.overlap_with_error",
                            f"anomaly {a.id} overlaps an error at event {eid}",
                            "warning",
                        )
                    )
    return out


def _validate_errors_reach_failure(graph: CausalErrorGraph) -> List[Issue]:
    """Every error should reach a failure transitively via causes/amplifies/contributes_to."""
    out: List[Issue] = []
    if not graph.failures:
        return out
    adj: dict[str, List[str]] = {r.id: [] for r in graph.errors}
    for e in graph.edges:
        if e.type in {EdgeType.CAUSES, EdgeType.AMPLIFIES, EdgeType.CONTRIBUTES_TO}:
            adj.setdefault(e.source, []).append(e.target)
    failure_ids = graph.failure_ids()
    for r in graph.errors:
        seen: Set[str] = set()
        stack = [r.id]
        reaches = False
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in failure_ids:
                reaches = True
                break
            stack.extend(adj.get(node, []))
        if not reaches:
            out.append(
                Issue(
                    "error.unreachable_from_failure",
                    f"error {r.id} does not reach any failure node",
                    "warning",
                )
            )
    return out


def _validate_no_duplicate_edges(graph: CausalErrorGraph) -> List[Issue]:
    seen: Set[tuple[str, str, str]] = set()
    out: List[Issue] = []
    for e in graph.edges:
        key = (e.source, e.target, e.type.value)
        if key in seen:
            out.append(
                Issue(
                    "edge.duplicate",
                    f"duplicate edge {e.source} -{e.type.value}-> {e.target}",
                    "warning",
                )
            )
        seen.add(key)
    return out
