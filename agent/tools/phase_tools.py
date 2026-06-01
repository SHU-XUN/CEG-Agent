from __future__ import annotations

from typing import Any, Dict

from ..agents import (
    AnomalyDetector,
    CausalGraphBuilder,
    ErrorHypothesisGenerator,
    EventBuilder,
    FailureAnalyzer,
    RepairValueEstimator,
)
from .base import NO_ARGS, Tool, ToolContext


def _summary_after_event_builder(ctx: ToolContext) -> Dict[str, Any]:
    return {"n_events": len(ctx.state.events)}


def _summary_after_failure_analyzer(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "n_failures": len(ctx.state.failures),
        "ids": [f.id for f in ctx.state.failures],
    }


def _summary_after_error_hypothesis(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "n_errors": len(ctx.state.errors),
        "ids": [r.id for r in ctx.state.errors],
    }


def _summary_after_causal_graph(ctx: ToolContext) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    for e in ctx.state.edges:
        by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
    return {"n_edges": len(ctx.state.edges), "by_type": by_type}


def _summary_after_anomaly(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "n_anomalies": len(ctx.state.anomaly),
        "ids": [a.id for a in ctx.state.anomaly],
    }


def _summary_after_repair(ctx: ToolContext) -> Dict[str, Any]:
    n_total = len(ctx.state.errors)
    n_with = sum(1 for r in ctx.state.errors if r.repair_value is not None)
    return {"n_with_value": n_with, "n_total": n_total}




def make_event_builder_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        EventBuilder(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_event_builder(ctx)}

    return Tool(
        name="build_events",
        description=(
            "Phase 1. Segment the normalized trace into thought-action-"
            "observation events E1..En and store them in state. Call this "
            "once, first. Returns only `n_events`; full event list lives in "
            "state — call inspect_state(fields=[\"events\"]) if you need it."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )


def make_failure_analyzer_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        FailureAnalyzer(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_failure_analyzer(ctx)}

    return Tool(
        name="find_failures",
        description=(
            "Phase 2. Identify the task-level failure(s) of the trace. "
            "Output is sparse — usually one F1. Requires events first. "
            "Returns only counts + ids; full failures live in state — "
            "call inspect_state(fields=[\"failures\"]) for details."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )


def make_error_hypothesis_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        ErrorHypothesisGenerator(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_error_hypothesis(ctx)}

    return Tool(
        name="propose_errors",
        description=(
            "Phase 3. Generate failure-relevant errors with mechanism, "
            "sub_mechanism, role, and description. Requires events + failures. "
            "Returns only counts + ids; full errors live in state — call "
            "inspect_state(fields=[\"errors\"]) for details."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )


def make_causal_graph_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        CausalGraphBuilder(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_causal_graph(ctx)}

    return Tool(
        name="build_causal_graph",
        description=(
            "Phase 4. Build event_next, attached_to, causes, amplifies, and "
            "contributes_to edges. Requires events + errors + failures."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )


def make_anomaly_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        AnomalyDetector(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_anomaly(ctx)}

    return Tool(
        name="find_anomalies",
        description=(
            "Phase 5. Identify harmless anomalies that do NOT contribute to "
            "any failure. Requires events + errors + failures. Returns only "
            "counts + ids; full anomalies live in state — call "
            "inspect_state(fields=[\"anomaly\"]) for details."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )


def make_repair_estimator_tool() -> Tool:
    def handler(_args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        RepairValueEstimator(ctx.llm).run(ctx.state)
        return {"ok": True, **_summary_after_repair(ctx)}

    return Tool(
        name="estimate_repair_values",
        description=(
            "Phase 7 (optional). Counterfactually estimate repair_value "
            "(high/medium/low) for every error. Skip if you don't need it."
        ),
        parameters=NO_ARGS,
        handler=handler,
    )
