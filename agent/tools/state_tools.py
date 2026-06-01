from __future__ import annotations

from typing import Any, Dict, List

from ..agents.graph_critic import apply_patch
from ..critic import auto_repair, validate_graph as run_validators
from .base import NO_ARGS, Tool, ToolContext



def make_read_trace_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        text = ctx.state.normalized_trace or ""
        return {
            "trace_id": ctx.state.trace_id,
            "task": ctx.state.task,
            "chars": len(text),
            "text": text,
            "truncated": bool(getattr(ctx.state, "trace_truncated", False)),
        }

    return Tool(
        name="read_trace",
        description=(
            "Return the normalized trace text in full (the agent's run, with "
            "thought/action/observation per step). The whole trace is "
            "returned every call — no per-call truncation. The `truncated` "
            "flag only reflects whether the upstream parser hit its hard "
            "ceiling (CE_TRACE_MAX_CHARS, default 800k chars)."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handler,
    )



_INSPECT_FIELDS = {"events", "errors", "failures", "edges", "anomaly", "todos"}


def make_inspect_state_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        fields = args.get("fields")
        if not fields:
            fields = sorted(_INSPECT_FIELDS)
        out: Dict[str, Any] = {
            "trace_id": ctx.state.trace_id,
        }
        if "events" in fields:
            out["events"] = [e.model_dump() for e in ctx.state.events]
        if "errors" in fields:
            out["errors"] = [e.model_dump(mode="json") for e in ctx.state.errors]
        if "failures" in fields:
            out["failures"] = [f.model_dump(mode="json") for f in ctx.state.failures]
        if "edges" in fields:
            out["edges"] = [e.model_dump(mode="json") for e in ctx.state.edges]
        if "anomaly" in fields:
            out["anomaly"] = [a.model_dump() for a in ctx.state.anomaly]
        if "todos" in fields:
            out["todos"] = ctx.todos.snapshot()
        return out

    return Tool(
        name="inspect_state",
        description=(
            "Return the current state of the graph. Optional `fields` array "
            "subselects from {events, errors, failures, edges, anomaly, todos}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_INSPECT_FIELDS),
                    },
                }
            },
            "additionalProperties": False,
        },
        handler=handler,
    )



def make_validate_graph_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        run_repair = bool(args.get("run_auto_repair", True))
        graph = ctx.state.to_graph()
        repair_log: List[str] = []
        if run_repair:
            graph, repair_log = auto_repair(graph)
            ctx.state.load_from_graph(graph)
        issues = run_validators(graph)
        return {
            "n_issues": len(issues),
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in issues
            ],
            "auto_repair_log": repair_log,
        }

    return Tool(
        name="validate_graph",
        description=(
            "Run deterministic schema/role/edge validators on the current "
            "graph. By default also runs auto_repair which fixes machine-"
            "fixable issues (illegal edges, missing attached_to, role "
            "consistency). Returns the remaining issue list — use this to "
            "decide whether to call apply_critic_patch."
        ),
        parameters={
            "type": "object",
            "properties": {
                "run_auto_repair": {
                    "type": "boolean",
                    "description": "If false, only report issues; do not fix.",
                    "default": True,
                }
            },
            "additionalProperties": False,
        },
        handler=handler,
    )




_PATCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "object",
            "description": (
                "Compact JSON patch. Any of these keys may appear: "
                "remove_errors[ids], remove_failures[ids], remove_anomalies[ids], "
                "remove_edges[{source,target,type}], add_edges[{source,target,type}], "
                "update_errors[{id,...partial fields}], update_failures[...], "
                "update_anomalies[...], merge_errors[{keep_id,drop_ids}]."
            ),
        }
    },
    "required": ["patch"],
    "additionalProperties": False,
}


def make_apply_patch_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        patch = args.get("patch") or {}
        graph = ctx.state.to_graph()
        graph = apply_patch(graph, patch)
        graph, repair_log = auto_repair(graph)
        ctx.state.load_from_graph(graph)
        issues = run_validators(graph)
        return {
            "applied": True,
            "n_issues_remaining": len(issues),
            "auto_repair_log": repair_log,
        }

    return Tool(
        name="apply_critic_patch",
        description=(
            "Apply a JSON patch to the graph (add/remove/update/merge nodes "
            "and edges) and re-run auto_repair. Use this to fix issues "
            "surfaced by validate_graph."
        ),
        parameters=_PATCH_PARAMETERS,
        handler=handler,
    )



def make_write_todo_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        items = args.get("items") or []
        replace = bool(args.get("replace", False))
        if replace:
            ctx.todos.replace(items)
        else:
            for it in items:
                ctx.todos.upsert(it)
        return {"todos": ctx.todos.snapshot()}

    return Tool(
        name="write_todo",
        description=(
            "Claude-Code-style task tracker. `items` is a list of "
            "{id, content, status} where status ∈ {pending, in_progress, "
            "completed}. By default upserts; pass replace=true to overwrite."
        ),
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "content"],
                        "additionalProperties": False,
                    },
                },
                "replace": {"type": "boolean", "default": False},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        handler=handler,
    )



def make_spawn_subagent_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        if ctx.sub_invoker is None:
            return {
                "ok": False,
                "error": "no sub-agent invoker is wired into this runtime",
            }
        focus = args.get("focus") or "general check"
        instructions = args.get("instructions") or ""
        max_iters = int(args.get("max_iters") or 5)
        return ctx.sub_invoker(
            focus=focus, instructions=instructions, max_iters=max_iters
        )

    return Tool(
        name="spawn_subagent",
        description=(
            "Delegate a focused diagnosis question to a sub-agent that has "
            "read-only access to the current graph plus inspection tools. "
            "Use this for: 'is R3 actually a root or is it caused by R1?', "
            "'are the anomalies in events E5-E7 actually errors?', etc. "
            "The sub-agent returns a structured report as a JSON dict."
        ),
        parameters={
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Short tag for what the sub-agent is investigating.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Free-text instructions for the sub-agent.",
                },
                "max_iters": {
                    "type": "integer",
                    "description": "Max tool-call iterations for the sub-agent.",
                    "default": 5,
                },
            },
            "required": ["focus", "instructions"],
            "additionalProperties": False,
        },
        handler=handler,
    )



def make_finalize_tool() -> Tool:
    def handler(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        ctx.finalize_request["finalized"] = True
        ctx.finalize_request["note"] = args.get("note", "")
        return {
            "finalized": True,
            "summary": {
                "events": len(ctx.state.events),
                "errors": len(ctx.state.errors),
                "failures": len(ctx.state.failures),
                "edges": len(ctx.state.edges),
                "anomalies": len(ctx.state.anomaly),
            },
        }

    return Tool(
        name="finalize",
        description=(
            "Emit the final causal error graph and exit the agent loop. "
            "Call this exactly once, when validate_graph reports no errors "
            "(warnings are acceptable)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Optional one-line note explaining why you finalized now.",
                }
            },
            "additionalProperties": False,
        },
        handler=handler,
        is_terminal=True,
    )
