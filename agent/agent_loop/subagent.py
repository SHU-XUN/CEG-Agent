from __future__ import annotations

import json
from typing import Any, Dict, List

from ..llm import LLMClient, Message
from ..pipeline.state import PipelineState
from ..prompts.base import BASE_SYSTEM
from ..tools import build_subagent_tools
from ..tools.base import ToolContext
from ..utils.logging import get_logger
from .todo import TodoList


_SUBAGENT_SYSTEM = (
    BASE_SYSTEM
    + "\n[PHASE:subagent]\n"
    + (
        "You are a focused diagnosis sub-agent. The parent agent has spawned "
        "you to investigate a narrow question against a snapshot of the "
        "current causal error graph. You have read-only inspection tools.\n\n"
        "Workflow:\n"
        "  1. Use inspect_state / read_trace to gather the evidence you need.\n"
        "  2. Reason concretely about the focus question.\n"
        "  3. Stop tool calls and emit a final assistant message containing a "
        "JSON report with this shape:\n"
        '       {"finding": "<short answer>",\n'
        '        "evidence": ["...", "..."],\n'
        '        "suggested_patch": {...optional same patch language as the parent...}}\n\n'
        "Do not ask the user questions. Do not finalize anything — only the "
        "parent agent can finalize."
    )
)


def invoke_subagent(
    *,
    parent_llm: LLMClient,
    parent_state: PipelineState,
    parent_todos: TodoList,
    focus: str,
    instructions: str,
    max_iters: int = 5,
) -> Dict[str, Any]:
    log = get_logger("subagent")
    log.info("spawn focus=%s max_iters=%d", focus, max_iters)

    snapshot = PipelineState(
        trace_id=parent_state.trace_id,
        task=parent_state.task,
        turns=list(parent_state.turns),
        normalized_trace=parent_state.normalized_trace,
        events=list(parent_state.events),
        failures=list(parent_state.failures),
        errors=list(parent_state.errors),
        edges=list(parent_state.edges),
        anomaly=list(parent_state.anomaly),
    )
    sub_todos = TodoList()

    tools = build_subagent_tools()
    for t in tools:
        if t.name == "validate_graph":
            original = t.handler

            def _readonly_handler(args, ctx, _orig=original):
                args = dict(args or {})
                args["run_auto_repair"] = False
                return _orig(args, ctx)

            t.handler = _readonly_handler
    ctx = ToolContext(
        state=snapshot, llm=parent_llm, todos=sub_todos
    )
    tool_specs = [t.to_spec() for t in tools]
    by_name = {t.name: t for t in tools}

    user_prompt = (
        f"FOCUS: {focus}\n\nINSTRUCTIONS:\n{instructions}\n\n"
        "Investigate, then emit a single JSON report (no prose around it)."
    )
    messages: List[Message] = [
        Message.system(_SUBAGENT_SYSTEM),
        Message.user(user_prompt),
    ]

    last_text = ""
    for step in range(max_iters):
        resp = parent_llm.chat(messages, tools=tool_specs)
        last_text = resp.text or last_text
        if not resp.has_tool_calls:
            break
        messages.append(
            Message.assistant(content=resp.text or "", tool_calls=resp.tool_calls)
        )
        for tc in resp.tool_calls:
            tool = by_name.get(tc.name)
            if tool is None:
                result: Dict[str, Any] = {"error": f"unknown tool: {tc.name}"}
            else:
                try:
                    result = tool.run(tc.arguments, ctx)
                except Exception as e:
                    result = {"error": str(e)}
            messages.append(
                Message.tool(tc.id, tc.name, json.dumps(result, ensure_ascii=False))
            )

    from ..llm.base import _extract_json

    report = _extract_json(last_text) if last_text else {}
    if not report:
        report = {"finding": last_text.strip()[:1000] or "(no report)", "evidence": []}
    log.info("sub-agent returned: %s", str(report)[:200])
    return {"ok": True, "focus": focus, "report": report}
