from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent_loop import AgentRuntime, invoke_subagent
from ..agents import GraphCritic, RepairValueEstimator
from ..config import FrameworkConfig, load_config
from ..critic import auto_repair, validate_graph
from ..llm import LLMClient, build_llm_client
from ..llm.tracked import TrackedLLMClient
from ..prompts.orchestrator import build_initial_messages
from ..schemas import CausalErrorGraph
from ..tools import build_default_tools
from ..utils import parse_raw_trace
from ..utils.display import Console
from ..utils.logging import get_logger
from ..utils.tokens import TokenAccountant
from .state import PipelineState


@dataclass
class AgenticDiagnosisResult:
    graph: CausalErrorGraph
    issues: List[str] = field(default_factory=list)
    iterations: int = 0
    finalized: bool = False
    todos: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)
    accountant: Optional[TokenAccountant] = None
    trace_chars_sent: int = 0
    trace_chars_full: int = 0
    trace_truncated: bool = False


class AgenticDiagnosisOrchestrator:

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        llm: Optional[LLMClient] = None,
        max_iters: int = 24,
        console: Optional[Console] = None,
    ):
        self.config = config or load_config()
        self.log = get_logger("agentic_orchestrator")
        self.accountant = TokenAccountant()
        inner_llm = llm or build_llm_client(self.config.llm)
        self.llm = TrackedLLMClient(
            inner_llm, self.accountant, model=self.config.llm.model
        )
        self.max_iters = max_iters
        self.console = console
        self.bundle_dir: Optional[str] = None
        self.log.info(
            "agentic mode: backend=%s max_iters=%d", self.llm.name, self.max_iters
        )

    def diagnose(self, raw_trace: Dict[str, Any]) -> AgenticDiagnosisResult:
        normalized = parse_raw_trace(raw_trace)
        state = PipelineState(
            trace_id=normalized["trace_id"],
            task=normalized["task"],
            turns=normalized["turns"],
            normalized_trace=normalized["text"],
            trace_chars_full=normalized["chars_full"],
            trace_truncated=normalized["truncated"],
        )
        self.accountant.reset()

        if self.console:
            chars_field = (
                f"{normalized['chars']:,} of {normalized['chars_full']:,} "
                f"({(100*normalized['chars']/normalized['chars_full']):.0f}%)"
                if normalized["truncated"] else f"{normalized['chars']:,}"
            )
            self.console.run_header(
                title="Causal Error Diagnosis — agentic mode",
                fields={
                    "trace_id": state.trace_id,
                    "backend": f"{self.llm.name} ({self.config.llm.model})",
                    "task": (state.task or "")[:80] + ("..." if len(state.task or "") > 80 else ""),
                    "max_iters": self.max_iters,
                    "trace_chars": chars_field,
                },
            )
            if normalized["truncated"]:
                self.console.warn(
                    f"trace clipped to {normalized['chars']:,} chars "
                    f"(full size {normalized['chars_full']:,}). "
                    "Bump CE_TRACE_MAX_CHARS to send more."
                )

        tools = build_default_tools(
            enable_repair=self.config.pipeline.enable_repair
        )

        runtime = AgentRuntime(
            llm=self.llm,
            state=state,
            tools=tools,
            initial_messages=build_initial_messages(state.trace_id, state.task or ""),
            max_iters=self.max_iters,
            sub_invoker=lambda **kw: _wrap_subagent(
                invoke_subagent(
                    parent_llm=self.llm,
                    parent_state=state,
                    parent_todos=runtime.todos,
                    **kw,
                ),
                self.console,
                kw,
            ),
            console=self.console,
            accountant=self.accountant,
            bundle_dir=self.bundle_dir,
        )

        with self.accountant.label("agentic"):
            t0 = time.time()
            loop_result = runtime.run()
            elapsed = time.time() - t0

        called_tools = {
            tc["name"]
            for s in runtime.steps
            for tc in s.tool_calls
        }


        critic_was_run = "apply_critic_patch" in called_tools or "validate_graph" in called_tools
        if state.events and (state.errors or state.failures) and not critic_was_run:
            self.log.info(
                "post-loop fallback: agent skipped critic; running 1 round"
            )
            with self.accountant.label("post_loop/graph_critic"):
                GraphCritic(
                    self.llm,
                    rounds=self.config.pipeline.critic_rounds,
                    max_failures=self.config.pipeline.max_failures,
                ).run(state)

        if (
            self.config.pipeline.enable_repair
            and state.errors
            and state.failures
            and not any(r.repair_value is not None for r in state.errors)
            and "estimate_repair_values" not in called_tools
        ):
            self.log.info(
                "post-loop fallback: agent skipped estimate_repair_values; "
                "running it now"
            )
            with self.accountant.label("post_loop/repair_estimator"):
                RepairValueEstimator(self.llm).run(state)


        graph = state.to_graph()
        graph, repair_log = auto_repair(graph)
        state.load_from_graph(graph)
        if repair_log and self.console:
            self.console.repair_log(repair_log)

        issues = [
            str(i)
            for i in validate_graph(
                state.to_graph(), max_failures=self.config.pipeline.max_failures
            )
        ]
        if issues:
            self.log.warning("post-loop issues remain: %s", issues)
            if self.console:
                self.console.issue_list(issues)

        tool_call_log = [
            {
                "iteration": s.iteration,
                "tool_calls": s.tool_calls,
                "assistant_text": s.assistant_text[:200],
                "elapsed_s": s.elapsed_s,
            }
            for s in runtime.steps
        ]

        if self.console:
            self.console.run_footer(
                events=len(state.events),
                errors=len(state.errors),
                failures=len(state.failures),
                edges=len(state.edges),
                anomalies=len(state.anomaly),
                iterations=loop_result["iterations"],
                elapsed_s=elapsed,
                finalized=bool(loop_result["finalized"]),
                issues_remaining=len(issues),
                token_totals=self.accountant.totals(),
            )

        return AgenticDiagnosisResult(
            graph=state.to_graph(),
            issues=issues,
            iterations=loop_result["iterations"],
            finalized=bool(loop_result["finalized"]),
            todos=loop_result["todos"],
            elapsed_s=elapsed,
            tool_call_log=tool_call_log,
            accountant=self.accountant,
            trace_chars_sent=len(state.normalized_trace),
            trace_chars_full=state.trace_chars_full,
            trace_truncated=state.trace_truncated,
        )


def _wrap_subagent(
    invoke_result: Dict[str, Any],
    console: Optional[Console],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if console:
        report = invoke_result.get("report") or {}
        console.subagent_done(
            focus=kwargs.get("focus", "?"),
            finding=str(report.get("finding") or report)[:300],
        )
    return invoke_result
