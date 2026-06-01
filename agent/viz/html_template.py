from __future__ import annotations

import json
from typing import Any, Dict, List

from ..schemas import CausalErrorGraph


_VIS_URL = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Causal Error Graph — {trace_id}</title>
<script src="{vis_url}"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  header {{ padding: 10px 14px; background: #1f2933; color: #f5f7fa; font-size: 14px; }}
  header b {{ color: #ffd6cc; }}
  header .meta {{ font-size: 12px; color: #cbd2d9; margin-top: 4px; }}
  #legend {{ padding: 6px 14px; background: #f5f7fa; border-bottom: 1px solid #d9e2ec; font-size: 12px; color: #334e68; }}
  #legend span {{ display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 12px; }}
  .l-event   {{ background:#cfe8ff; color:#1f4e79; }}
  .l-error   {{ background:#ffd6cc; color:#7a2920; }}
  .l-failure {{ background:#c0392b; color:#fff; font-weight:600; }}
  .l-anomaly {{ background:#fff3b0; color:#7a5b00; border:1px dashed #a07a00; }}
  .l-causes      {{ background:transparent; color:#c0392b; font-weight:600; }}
  .l-amplifies   {{ background:transparent; color:#e67e22; font-weight:600; }}
  .l-contributes {{ background:transparent; color:#7a0000; font-weight:600; }}
  #network {{ width: 100%; height: calc(100vh - 90px); background: #fafbfc; }}
  #detail {{ position: fixed; right: 12px; top: 100px; max-width: 360px; max-height: 70vh; overflow:auto;
             background: #ffffff; border: 1px solid #d9e2ec; padding: 10px; border-radius: 6px;
             font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); display:none; }}
  #detail h3 {{ margin: 0 0 6px 0; font-size: 13px; }}
  #detail .meta {{ color: #627d98; margin-bottom: 6px; }}
  #detail pre  {{ background: #f5f7fa; padding: 6px; border-radius: 4px; overflow:auto; font-size: 11px; }}
</style>
</head>
<body>
<header>
  <b>Causal Error Graph</b> &nbsp; trace_id = <code>{trace_id}</code>
  <div class="meta">events={n_events} · errors={n_errors} · failures={n_failures} · edges={n_edges} · anomalies={n_anomalies}</div>
</header>
<div id="legend">
  <span class="l-event">event</span>
  <span class="l-error">error</span>
  <span class="l-failure">failure</span>
  <span class="l-anomaly">anomaly</span>
  &nbsp;|&nbsp;
  <span class="l-causes">→ causes</span>
  <span class="l-amplifies">⇢ amplifies</span>
  <span class="l-contributes">⇒ contributes_to</span>
</div>
<div id="network"></div>
<div id="detail"></div>
<script>
const NODES = {nodes_json};
const EDGES = {edges_json};
const FULL  = {full_json};

const data = {{
  nodes: new vis.DataSet(NODES),
  edges: new vis.DataSet(EDGES)
}};
const opts = {{
  layout: {{ improvedLayout: true }},
  physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -3500, springLength: 140 }} }},
  nodes:  {{ shape: 'box', font: {{ size: 12 }} }},
  edges:  {{ smooth: {{ type: 'cubicBezier' }}, arrows: 'to' }},
  interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }}
}};
const network = new vis.Network(document.getElementById('network'), data, opts);

const detail = document.getElementById('detail');
network.on('click', params => {{
  if (!params.nodes.length) {{ detail.style.display = 'none'; return; }}
  const id = params.nodes[0];
  const obj = FULL[id];
  if (!obj) {{ detail.style.display = 'none'; return; }}
  detail.style.display = 'block';
  detail.innerHTML = '<h3>' + id + '</h3>' +
    '<div class="meta">' + (obj._kind || '') + '</div>' +
    '<pre>' + JSON.stringify(obj, null, 2) + '</pre>';
}});
</script>
</body>
</html>
"""



_NODE_STYLE = {
    "event": {
        "shape": "ellipse",
        "color": {"background": "#cfe8ff", "border": "#3a7abf"},
        "font": {"color": "#1f4e79", "size": 12, "face": "Helvetica"},
        "widthConstraint": {"maximum": 220},
    },
    "error": {
        "shape": "box",
        "color": {"background": "#ffd6cc", "border": "#c0392b"},
        "font": {"color": "#7a2920", "size": 12, "face": "Helvetica"},
        "borderWidth": 2,
        "shapeProperties": {"borderRadius": 6},
        "widthConstraint": {"maximum": 260},
    },
    "failure": {
        "shape": "box",
        "color": {"background": "#c0392b", "border": "#7a0000"},
        "font": {"color": "#ffffff", "size": 13, "face": "Helvetica", "bold": True},
        "borderWidth": 3,
        "shapeProperties": {"borderRadius": 4},
        "widthConstraint": {"minimum": 160, "maximum": 280},
    },
    "anomaly": {
        "shape": "ellipse",
        "color": {"background": "#fff3b0", "border": "#a07a00"},
        "font": {"color": "#7a5b00", "size": 12, "face": "Helvetica"},
        "shapeProperties": {"borderDashes": [4, 4]},
        "widthConstraint": {"maximum": 220},
    },
}

_EDGE_STYLE = {
    "event_next":     {"color": "#888888", "dashes": False, "width": 1, "arrows": {"to": {"scaleFactor": 0.6}}},
    "attached_to":    {"color": "#888888", "dashes": True,  "width": 1},
    "causes":         {"color": "#c0392b", "dashes": False, "width": 2.2, "arrows": {"to": {"scaleFactor": 0.9}}},
    "amplifies":      {"color": "#e67e22", "dashes": True,  "width": 2.2, "arrows": {"to": {"scaleFactor": 0.9}}},
    "contributes_to": {"color": "#7a0000", "dashes": False, "width": 2.6, "arrows": {"to": {"scaleFactor": 1.1}}},
}


def render_html(graph: CausalErrorGraph) -> str:
    nodes: List[Dict[str, Any]] = []
    full: Dict[str, Dict[str, Any]] = {}

    for e in graph.events:
        label = f"{e.id}\n{(e.action or e.thought or '')[:38]}"
        nodes.append({"id": e.id, "label": label, **_NODE_STYLE["event"]})
        full[e.id] = {"_kind": "event", **e.model_dump()}

    for r in graph.errors:
        rv = f"\nrepair={r.repair_value.value}" if r.repair_value else ""
        label = f"{r.id} [{r.mechanism.value} | {r.role.value}]\n{r.sub_mechanism}{rv}"
        nodes.append({"id": r.id, "label": label, **_NODE_STYLE["error"]})
        full[r.id] = {"_kind": "error", **r.model_dump(mode="json")}

    for f in graph.failures:
        desc = f.description.strip()
        if len(desc) > 240:
            desc = desc[:237] + "…"
        label = f"{f.id} · {f.type.value}\n{desc}"
        nodes.append({"id": f.id, "label": label, **_NODE_STYLE["failure"]})
        full[f.id] = {"_kind": "failure", **f.model_dump(mode="json")}

    for a in graph.anomaly:
        label = f"{a.id}\n{a.summary[:48]}"
        nodes.append({"id": a.id, "label": label, **_NODE_STYLE["anomaly"]})
        full[a.id] = {"_kind": "anomaly", **a.model_dump()}

    edges: List[Dict[str, Any]] = []
    for e in graph.edges:
        edges.append(
            {
                "from": e.source,
                "to": e.target,
                "label": e.type.value,
                "font": {"size": 10, "align": "middle", "color": "#444"},
                **_EDGE_STYLE.get(e.type.value, {}),
            }
        )
    for a in graph.anomaly:
        for eid in a.event_ids:
            edges.append(
                {
                    "from": a.id,
                    "to": eid,
                    "label": "anomaly@",
                    "color": "#a07a00",
                    "dashes": [2, 4],
                    "width": 1,
                    "font": {"size": 9, "color": "#a07a00"},
                }
            )

    return _HTML_TEMPLATE.format(
        trace_id=graph.trace_id,
        vis_url=_VIS_URL,
        n_events=len(graph.events),
        n_errors=len(graph.errors),
        n_failures=len(graph.failures),
        n_edges=len(graph.edges),
        n_anomalies=len(graph.anomaly),
        nodes_json=json.dumps(nodes, ensure_ascii=False),
        edges_json=json.dumps(edges, ensure_ascii=False),
        full_json=json.dumps(full, ensure_ascii=False),
    )
