from __future__ import annotations

from typing import List

from .base import Tool
from .phase_tools import (
    make_anomaly_tool,
    make_causal_graph_tool,
    make_error_hypothesis_tool,
    make_event_builder_tool,
    make_failure_analyzer_tool,
    make_repair_estimator_tool,
)
from .state_tools import (
    make_apply_patch_tool,
    make_finalize_tool,
    make_inspect_state_tool,
    make_read_trace_tool,
    make_spawn_subagent_tool,
    make_validate_graph_tool,
    make_write_todo_tool,
)
from .viz_tool import make_render_graph_tool


def build_default_tools(*, enable_repair: bool = True) -> List[Tool]:
    tools: List[Tool] = [
        make_read_trace_tool(),
        make_inspect_state_tool(),
        make_event_builder_tool(),
        make_failure_analyzer_tool(),
        make_error_hypothesis_tool(),
        make_causal_graph_tool(),
        make_anomaly_tool(),
        make_validate_graph_tool(),
        make_apply_patch_tool(),
        make_write_todo_tool(),
        make_spawn_subagent_tool(),
    ]
    if enable_repair:
        tools.append(make_repair_estimator_tool())
    tools.append(make_render_graph_tool())
    tools.append(make_finalize_tool())
    return tools


def build_subagent_tools() -> List[Tool]:
    
    return [
        make_read_trace_tool(),
        make_inspect_state_tool(),
        make_validate_graph_tool(),
    ]
