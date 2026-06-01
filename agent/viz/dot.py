from __future__ import annotations

import html
from typing import List

from ..schemas import CausalErrorGraph, EdgeType




_NODE_STYLES = {
    "event":  'shape=oval,    style="filled,setlinewidth(1)", fillcolor="#cfe8ff", color="#3a7abf", fontname="Helvetica"',
    "error":  'shape=box,     style="filled,rounded",          fillcolor="#ffd6cc", color="#c0392b", fontname="Helvetica"',
    "failure":'shape=doubleoctagon, style="filled",            fillcolor="#ff8080", color="#7a0000", fontname="Helvetica", fontcolor="#ffffff"',
    "anomaly":'shape=ellipse, style="filled,dashed",          fillcolor="#fff3b0", color="#a07a00", fontname="Helvetica"',
}

_EDGE_STYLES = {
    EdgeType.EVENT_NEXT:     'color="#888888", style=solid, penwidth=1, arrowsize=0.7',
    EdgeType.ATTACHED_TO:    'color="#888888", style=dashed, penwidth=1, arrowsize=0.6',
    EdgeType.CAUSES:         'color="#c0392b", style=solid, penwidth=2, arrowsize=0.9',
    EdgeType.AMPLIFIES:      'color="#e67e22", style=dashed, penwidth=2, arrowsize=0.9',
    EdgeType.CONTRIBUTES_TO: 'color="#7a0000", style=bold,  penwidth=2.5, arrowsize=1.0',
}



def render_dot(graph: CausalErrorGraph) -> str:
    lines: List[str] = [
        f'digraph "causal_error_graph_{_q(graph.trace_id)}" {{',
        '  graph [rankdir=LR, fontname="Helvetica", labelloc="t", '
        f'label=<<b>Causal Error Graph</b><br/>trace {_q(graph.trace_id)}>];',
        '  node [fontsize=11];',
        '  edge [fontsize=9, fontname="Helvetica"];',
        '',
        '  // ===== events (timeline) =====',
        '  subgraph cluster_events {',
        '    label="events"; style=dotted; color="#888888";',
    ]
    for e in graph.events:
        label = _node_label(e.id, e.action or e.thought, max_len=46)
        lines.append(
            f'    "{_q(e.id)}" [label=<{label}>, {_NODE_STYLES["event"]}];'
        )
    lines.append('  }')

    if graph.errors:
        lines += ['', '  // ===== errors =====']
        for r in graph.errors:
            sub_text = f"{r.mechanism.value} | {r.role.value}"
            label = _error_label(r.id, sub_text, r.sub_mechanism, r.repair_value)
            lines.append(
                f'  "{_q(r.id)}" [label=<{label}>, {_NODE_STYLES["error"]}];'
            )

    if graph.failures:
        lines += ['', '  // ===== failures =====']
        for f in graph.failures:
            label = _failure_label(f.id, f.type.value, f.description)
            lines.append(
                f'  "{_q(f.id)}" [label=<{label}>, {_NODE_STYLES["failure"]}];'
            )

    if graph.anomaly:
        lines += ['', '  // ===== anomalies =====']
        for a in graph.anomaly:
            label = _node_label(a.id, a.summary, max_len=46)
            lines.append(
                f'  "{_q(a.id)}" [label=<{label}>, {_NODE_STYLES["anomaly"]}];'
            )

    lines += ['', '  // ===== edges =====']
    for e in graph.edges:
        style = _EDGE_STYLES.get(e.type, "color=black")
        attrs = f'[{style}, label="{e.type.value}"]'
        lines.append(f'  "{_q(e.source)}" -> "{_q(e.target)}" {attrs};')

    for a in graph.anomaly:
        for eid in a.event_ids:
            lines.append(
                f'  "{_q(a.id)}" -> "{_q(eid)}" '
                '[color="#a07a00", style=dotted, penwidth=1, arrowsize=0.5, label="anomaly@"];'
            )

    lines.append('}')
    return "\n".join(lines) + "\n"



def _q(s: str) -> str:
    return s.replace('"', '\\"')


def _esc(s: str) -> str:
    """HTML-escape inside a Graphviz HTML-like label."""
    return html.escape(s or "", quote=False)


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _node_label(node_id: str, body: str, max_len: int = 46) -> str:
    return (
        f'<b>{_esc(node_id)}</b><br/>'
        f'<font point-size="9">{_esc(_truncate(body, max_len))}</font>'
    )


def _error_label(node_id: str, taxonomy: str, sub: str, repair) -> str:
    rv = (
        f'<br/><font point-size="9" color="#7a0000"><i>repair={_esc(repair.value)}</i></font>'
        if repair else ""
    )
    return (
        f'<b>{_esc(node_id)}</b> <font point-size="9">[{_esc(taxonomy)}]</font>'
        f'<br/><font point-size="9">{_esc(_truncate(sub, 46))}</font>'
        f'{rv}'
    )


def _failure_label(node_id: str, ftype: str, desc: str) -> str:
    return (
        f'<b>{_esc(node_id)}</b> <font point-size="9">({_esc(ftype)})</font><br/>'
        f'<font point-size="9">{_esc(_truncate(desc, 60))}</font>'
    )
