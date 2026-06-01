from __future__ import annotations

import os
from typing import Any, Dict, List

from ..viz import render_dot, render_html, render_svg
from .base import Tool, ToolContext


_VALID_FORMATS = {"dot", "html", "svg"}


def make_render_graph_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        formats = args.get("formats") or ["dot", "html", "svg"]
        formats = [f for f in formats if f in _VALID_FORMATS]
        if not formats:
            return {"error": "no valid formats; pick from dot/html/svg"}
        write = bool(args.get("write", False))
        out_dir = args.get("output_dir") or ctx.bundle_dir
        preview_lines = int(args.get("preview_lines") or 12)

        graph = ctx.state.to_graph()
        report: Dict[str, Any] = {
            "trace_id": graph.trace_id,
            "counts": {
                "events":    len(graph.events),
                "errors":    len(graph.errors),
                "failures":  len(graph.failures),
                "edges":     len(graph.edges),
                "anomalies": len(graph.anomaly),
            },
            "formats": {},
        }
        artifacts: Dict[str, str] = {}

        dot_src = render_dot(graph)
        report["formats"]["dot"] = {
            "lines": dot_src.count("\n"),
            "bytes": len(dot_src.encode("utf-8")),
            "preview": "\n".join(dot_src.splitlines()[:preview_lines]),
        }
        artifacts["graph.dot"] = dot_src

        if "html" in formats:
            html = render_html(graph)
            report["formats"]["html"] = {
                "bytes": len(html.encode("utf-8")),
                "viewer": "vis-network from CDN",
            }
            artifacts["graph.html"] = html

        if "svg" in formats:
            svg = render_svg(dot_src)
            if svg is None:
                report["formats"]["svg"] = {
                    "ok": False,
                    "reason": "graphviz `dot` binary not found on PATH",
                }
            else:
                report["formats"]["svg"] = {
                    "ok": True,
                    "bytes": len(svg.encode("utf-8")),
                }
                artifacts["graph.svg"] = svg

        if write:
            target_dir = out_dir
            if not target_dir:
                return {
                    **report,
                    "error": (
                        "write=true but no output_dir was provided and no "
                        "bundle_dir is active on the context"
                    ),
                }
            os.makedirs(target_dir, exist_ok=True)
            written: Dict[str, str] = {}
            for fname, content in artifacts.items():
                fmt = fname.split(".")[-1]
                if fmt != "dot" and fmt not in formats:
                    continue
                path = os.path.join(target_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                written[fname] = path
            report["written"] = written
            report["output_dir"] = target_dir
        return report

    return Tool(
        name="render_graph",
        description=(
            "Materialize a visualization of the current Causal Error Graph. "
            "Returns size info + a short DOT preview. Use `write=true` to save "
            "files to disk — they go to `output_dir` if you pass it, otherwise "
            "to the bundle directory the orchestrator already set up. "
            "DOT is always rendered (cheap); HTML uses vis-network from CDN; "
            "SVG only works if the `dot` binary is on PATH."
        ),
        parameters={
            "type": "object",
            "properties": {
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(_VALID_FORMATS)},
                    "description": (
                        "Subset of {dot,html,svg}. Default: ['dot','html','svg']. "
                        "Always include 'svg' unless you know Graphviz is unavailable; "
                        "render_svg returns a skip marker if `dot` isn't on PATH so "
                        "asking for it is always safe."
                    ),
                },
                "write": {
                    "type": "boolean",
                    "description": "If true, write files to disk.",
                    "default": False,
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Directory to write into. Defaults to the active "
                        "bundle directory."
                    ),
                },
                "preview_lines": {
                    "type": "integer",
                    "description": "How many DOT lines to include in the preview field.",
                    "default": 12,
                },
            },
            "additionalProperties": False,
        },
        handler=handler,
    )
