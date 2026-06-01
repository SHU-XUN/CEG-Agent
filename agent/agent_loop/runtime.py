from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm import LLMClient, Message, ToolCall
from ..pipeline.state import PipelineState
from ..tools.base import Tool, ToolContext
from ..utils.display import Console
from ..utils.logging import get_logger
from ..utils.tokens import TokenAccountant
from .todo import TodoList


@dataclass
class TraceStep:

    iteration: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    elapsed_s: float = 0.0


class AgentRuntime:

    DEFAULT_KEEP_RECENT_TOOL_MSGS = 4
    DEFAULT_TOOL_RESULT_THRESHOLD_CHARS = 4000
    _ELIDED_PREFIX = "[elided "

    def __init__(
        self,
        *,
        llm: LLMClient,
        state: PipelineState,
        tools: List[Tool],
        initial_messages: List[Message],
        max_iters: int = 24,
        sub_invoker: Optional[Callable[..., Dict[str, Any]]] = None,
        log_name: str = "agent_loop",
        console: Optional[Console] = None,
        accountant: Optional[TokenAccountant] = None,
        bundle_dir: Optional[str] = None,
        keep_recent_tool_msgs: Optional[int] = None,
        tool_result_threshold_chars: Optional[int] = None,
    ):
        self.llm = llm
        self.state = state
        self.tools = tools
        self._tools_by_name = {t.name: t for t in tools}
        self.messages: List[Message] = list(initial_messages)
        self.max_iters = max_iters
        self.todos = TodoList()
        self.finalize_request: Dict[str, Any] = {}
        self.sub_invoker = sub_invoker
        self.steps: List[TraceStep] = []
        self.log = get_logger(log_name)
        self.console = console
        self.accountant = accountant
        self.bundle_dir = bundle_dir
        self.keep_recent_tool_msgs = (
            keep_recent_tool_msgs
            if keep_recent_tool_msgs is not None
            else self.DEFAULT_KEEP_RECENT_TOOL_MSGS
        )
        self.tool_result_threshold_chars = (
            tool_result_threshold_chars
            if tool_result_threshold_chars is not None
            else self.DEFAULT_TOOL_RESULT_THRESHOLD_CHARS
        )

    def _label(self, name: str):
        return self.accountant.label(name) if self.accountant else nullcontext()

    def _last_call(self):
        if not self.accountant:
            return None
        calls = self.accountant.calls
        return calls[-1] if calls else None

    def run(self) -> Dict[str, Any]:
        tool_specs = [t.to_spec() for t in self.tools]
        ctx = ToolContext(
            state=self.state,
            llm=self.llm,
            todos=self.todos,
            finalize_request=self.finalize_request,
            sub_invoker=self.sub_invoker,
            bundle_dir=self.bundle_dir,
        )

        for it in range(self.max_iters):
            step = TraceStep(iteration=it + 1)
            iter_t0 = time.time()
            self._compact_old_tool_messages()
            with self._label(f"iter{it + 1}/decide"):
                resp = self.llm.chat(self.messages, tools=tool_specs)
            step.assistant_text = resp.text or ""

            if not resp.has_tool_calls:
                self.messages.append(Message.assistant(content=resp.text or ""))
                step.elapsed_s = round(time.time() - iter_t0, 4)
                self.steps.append(step)
                self.log.info(
                    "iter %d: no tool calls; assistant text=%r",
                    it + 1,
                    (resp.text or "")[:120],
                )
                if self.console:
                    self.console.iter_start(
                        iteration=it + 1, assistant_text=resp.text or ""
                    )
                    self.console.notice("(no tool calls — loop exits)")
                break

            if self.console:
                self.console.iter_start(
                    iteration=it + 1, assistant_text=resp.text or ""
                )

            self.messages.append(
                Message.assistant(content=resp.text or "", tool_calls=resp.tool_calls)
            )

            terminal_called = False
            for tc in resp.tool_calls:
                if self.console:
                    self.console.tool_call(name=tc.name, arguments=tc.arguments)
                with self._label(f"iter{it + 1}/{tc.name}"):
                    result, terminal = self._dispatch(tc, ctx)
                terminal_called = terminal_called or terminal
                step.tool_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": _preview(result),
                    }
                )
                self.messages.append(
                    Message.tool(
                        tc.id, tc.name, json.dumps(result, ensure_ascii=False)
                    )
                )
                self.log.info(
                    "iter %d: %s(%s) -> %s",
                    it + 1,
                    tc.name,
                    _short(tc.arguments),
                    _preview(result),
                )
                if self.console:
                    self.console.tool_result(
                        name=tc.name,
                        result=result,
                        is_error="error" in result,
                        token_call=self._last_call(),
                    )

            step.elapsed_s = round(time.time() - iter_t0, 4)
            self.steps.append(step)
            if terminal_called or self.finalize_request.get("finalized"):
                break
        else:
            self.log.warning("agent loop hit max_iters=%d without finalize", self.max_iters)
            if self.console:
                self.console.warn(
                    f"agent loop hit max_iters={self.max_iters} without finalize"
                )

        return {
            "finalized": bool(self.finalize_request.get("finalized")),
            "iterations": len(self.steps),
            "todos": self.todos.snapshot(),
            "messages": self.messages,
            "steps": self.steps,
        }

    def _dispatch(self, tc: ToolCall, ctx: ToolContext):
        tool = self._tools_by_name.get(tc.name)
        if tool is None:
            return {"error": f"unknown tool: {tc.name}"}, False
        try:
            result = tool.run(tc.arguments, ctx)
        except Exception as e:  # noqa: BLE001
            self.log.exception("tool %s raised", tc.name)
            return {"error": f"{type(e).__name__}: {e}"}, False
        return result, tool.is_terminal

    def _compact_old_tool_messages(self) -> None:
        if self.keep_recent_tool_msgs < 0 or self.tool_result_threshold_chars <= 0:
            return
        tool_indices = [
            i for i, m in enumerate(self.messages) if m.role == "tool"
        ]
        if len(tool_indices) <= self.keep_recent_tool_msgs:
            return
        cutoff = len(tool_indices) - self.keep_recent_tool_msgs
        for k in tool_indices[:cutoff]:
            msg = self.messages[k]
            if msg.content.startswith(self._ELIDED_PREFIX):
                continue
            if len(msg.content) <= self.tool_result_threshold_chars:
                continue
            n_chars = len(msg.content)
            tool_name = msg.name or "tool"
            msg.content = (
                f"{self._ELIDED_PREFIX}{tool_name} result: "
                f"{n_chars} chars dropped from history. "
                f"The structured data is still in state — call "
                f"inspect_state(...) or re-run {tool_name} if needed.]"
            )
            self.log.debug(
                "compacted msg #%d (%s): %d chars -> %d chars",
                k, tool_name, n_chars, len(msg.content),
            )

def _short(args: Any, limit: int = 60) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _preview(result: Any, limit: int = 200) -> str:
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 3] + "..."
