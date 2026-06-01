from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..schemas import (
    CausalErrorGraph,
    Edge,
    EdgeType,
    LEGAL_EDGE_ENDPOINTS,
    Role,
)

def auto_repair(graph: CausalErrorGraph) -> Tuple[CausalErrorGraph, List[str]]:
    log: List[str] = []
    g = graph.model_copy(deep=True)

    legal_edges: List[Edge] = []
    known = g.node_ids()
    for e in g.edges:
        expected = LEGAL_EDGE_ENDPOINTS.get(e.type)
        if expected is None:
            log.append(f"drop edge with unknown type: {e}")
            continue
        if (e.source[:1], e.target[:1]) != expected:
            log.append(f"drop edge with bad endpoints: {e}")
            continue
        if e.source not in known or e.target not in known:
            log.append(f"drop dangling edge: {e}")
            continue
        legal_edges.append(e)
    g.edges = legal_edges

    seen: Set[Tuple[str, str, str]] = set()
    dedup: List[Edge] = []
    for e in g.edges:
        key = (e.source, e.target, e.type.value)
        if key in seen:
            log.append(f"dedup edge {key}")
            continue
        seen.add(key)
        dedup.append(e)
    g.edges = dedup

    attached: Set[str] = {
        e.source for e in g.edges if e.type == EdgeType.ATTACHED_TO
    }
    for r in g.errors:
        if r.id in attached:
            continue
        if r.event_id in g.event_ids():
            g.edges.append(
                Edge(source=r.id, target=r.event_id, type=EdgeType.ATTACHED_TO)
            )
            log.append(f"add missing attached_to {r.id}->{r.event_id}")

    incoming_causes: Dict[str, int] = {r.id: 0 for r in g.errors}
    in_amplifies: Dict[str, bool] = {r.id: False for r in g.errors}
    for e in g.edges:
        if e.type == EdgeType.CAUSES and e.target in incoming_causes:
            incoming_causes[e.target] += 1
        if e.type == EdgeType.AMPLIFIES:
            if e.source in in_amplifies:
                in_amplifies[e.source] = True
            if e.target in in_amplifies:
                in_amplifies[e.target] = True
    for r in g.errors:
        if r.role == Role.ROOT and incoming_causes.get(r.id, 0) > 0:
            log.append(f"reclassify {r.id}: root -> propagated (has upstream causes)")
            r.role = Role.PROPAGATED
        elif r.role == Role.PROPAGATED and incoming_causes.get(r.id, 0) == 0:
            if in_amplifies.get(r.id, False):
                log.append(f"reclassify {r.id}: propagated -> amplification")
                r.role = Role.AMPLIFICATION
            else:
                log.append(f"reclassify {r.id}: propagated -> root (no upstream)")
                r.role = Role.ROOT
        elif r.role == Role.AMPLIFICATION and not in_amplifies.get(r.id, False):
            if incoming_causes.get(r.id, 0) > 0:
                log.append(
                    f"reclassify {r.id}: amplification -> propagated "
                    "(no amplifies edge but has upstream causes)"
                )
                r.role = Role.PROPAGATED
            else:
                log.append(
                    f"reclassify {r.id}: amplification -> root "
                    "(no amplifies edge and no upstream causes)"
                )
                r.role = Role.ROOT

    if g.events:
        existing_next: Set[Tuple[str, str]] = {
            (e.source, e.target)
            for e in g.edges
            if e.type == EdgeType.EVENT_NEXT
        }
        for a, b in zip(g.events, g.events[1:]):
            if (a.id, b.id) not in existing_next:
                g.edges.append(
                    Edge(source=a.id, target=b.id, type=EdgeType.EVENT_NEXT)
                )
                log.append(f"add missing event_next {a.id}->{b.id}")

    return g, log
