from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents import (
    AnomalyDetector,
    CausalGraphBuilder,
    ErrorHypothesisGenerator,
    EventBuilder,
    FailureAnalyzer,
    GraphCritic,
    RepairValueEstimator,
)
from ..config import FrameworkConfig, load_config
from ..critic import auto_repair, validate_graph
from ..llm import LLMClient, build_llm_client
from ..llm.tracked import TrackedLLMClient
from ..schemas import CausalErrorGraph
from ..utils import parse_raw_trace
from ..utils.display import Console
from ..utils.logging import get_logger
from ..utils.tokens import TokenAccountant, TokenTotals
from .state import PipelineState


@dataclass
class DiagnosisResult:
    graph: CausalErrorGraph
    issues: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    critic_log: List[str] = field(default_factory=list)
    accountant: Optional[TokenAccountant] = None
    elapsed_s: float = 0.0
    trace_chars_sent: int = 0
    trace_chars_full: int = 0
    trace_truncated: bool = False


_PHASE_SUMMARY = {
    "event_builder":     lambda s: f"events={len(s.events)}",
    "failure_analyzer":  lambda s: f"failures={len(s.failures)}",
    "error_hypothesis":  lambda s: f"errors={len(s.errors)}",
    "causal_graph":      lambda s: f"edges={len(s.edges)}",
    "anomaly_detector":  lambda s: f"anomalies={len(s.anomaly)}",
    "graph_critic":      lambda s: (
        f"graph clean (events={len(s.events)} errors={len(s.errors)} "
        f"failures={len(s.failures)} edges={len(s.edges)} anomalies={len(s.anomaly)})"
    ),
    "repair_estimator":  lambda s: (
        "repair_values: " +
        ", ".join(
            f"{r.id}={r.repair_value.value if r.repair_value else '-'}"
            for r in s.errors
        ) if s.errors else "(no errors)"
    ),
}

_PHASE_LABELS = {
    "event_builder":     "Event Builder",
    "failure_analyzer":  "Failure Analyzer",
    "error_hypothesis":  "Error Hypothesis Generator",
    "causal_graph":      "Causal Graph Builder",
    "anomaly_detector":  "Anomaly Detector",
    "graph_critic":      "Graph Critic",
    "repair_estimator":  "Repair Value Estimator",
}


class DiagnosisOrchestrator:

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        llm: Optional[LLMClient] = None,
        console: Optional[Console] = None,
    ):
        self.config = config or load_config()
        self.log = get_logger("orchestrator")
        self.console = console
        self.accountant = TokenAccountant()
        inner_llm = llm or build_llm_client(self.config.llm)
        self.llm = TrackedLLMClient(inner_llm, self.accountant, model=self.config.llm.model)
        self.log.info("LLM backend: %s", self.llm.name)

        self.event_builder = EventBuilder(self.llm)
        self.failure_analyzer = FailureAnalyzer(self.llm)
        self.error_hypothesis = ErrorHypothesisGenerator(self.llm)
        self.causal_graph = CausalGraphBuilder(self.llm)
        self.anomaly_detector = AnomalyDetector(self.llm)
        self.graph_critic = GraphCritic(
            self.llm,
            rounds=self.config.pipeline.critic_rounds,
            max_failures=self.config.pipeline.max_failures,
        )
        self.repair_estimator = RepairValueEstimator(self.llm)


    def diagnose(self, raw_trace: Dict[str, Any]) -> DiagnosisResult:
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
                title="Causal Error Diagnosis — pipeline mode",
                fields={
                    "trace_id": state.trace_id,
                    "backend": f"{self.llm.name} ({self.config.llm.model})",
                    "task": (state.task or "")[:80] + ("..." if len(state.task or "") > 80 else ""),
                    "critic_rounds": self.config.pipeline.critic_rounds,
                    "trace_chars": chars_field,
                },
            )
            if normalized["truncated"]:
                self.console.warn(
                    f"trace clipped to {normalized['chars']:,} chars "
                    f"(full size {normalized['chars_full']:,}). "
                    "Bump CE_TRACE_MAX_CHARS to send more."
                )

        timings: Dict[str, float] = {}
        phases = [
            ("event_builder", 1, self.event_builder),
            ("failure_analyzer", 2, self.failure_analyzer),
            ("error_hypothesis", 3, self.error_hypothesis),
            ("causal_graph", 4, self.causal_graph),
            ("anomaly_detector", 5, self.anomaly_detector),
            ("graph_critic", 6, self.graph_critic),
        ]
        if self.config.pipeline.enable_repair:
            phases.append(("repair_estimator", 7, self.repair_estimator))

        run_t0 = time.time()
        phase_errors: Dict[str, str] = {}
        for name, phase_no, agent in phases:
            if self.console:
                self.console.phase_start(
                    phase=phase_no, name=_PHASE_LABELS.get(name, name)
                )
            t0 = time.time()
            calls_before = len(self.accountant.calls)
            with self.accountant.label(name):
                try:
                    agent.run(state)
                except Exception as e:  
                    self.log.exception("phase %s failed", name)
                    phase_errors[name] = f"{type(e).__name__}: {e}"
                    if self.console:
                        self.console.error(f"phase {name} failed: {e}")
            elapsed = time.time() - t0
            timings[name] = elapsed
            if self.console:
                phase_calls = self.accountant.calls[calls_before:]
                phase_totals = TokenTotals(
                    calls=len(phase_calls),
                    input_tokens=sum(c.input_tokens for c in phase_calls),
                    output_tokens=sum(c.output_tokens for c in phase_calls),
                    elapsed_s=sum(c.elapsed_s for c in phase_calls),
                )
                summary = _PHASE_SUMMARY.get(name, lambda _s: "ok")(state)
                self.console.phase_done(
                    summary=summary, elapsed_s=elapsed, tokens=phase_totals
                )
            self.log.debug("phase %s took %.2fs", name, elapsed)
        total_elapsed = time.time() - run_t0

        graph = state.to_graph()

        graph, _ = auto_repair(graph)
        state.load_from_graph(graph)
        graph = state.to_graph()

        issues = [str(i) for i in validate_graph(
            graph, max_failures=self.config.pipeline.max_failures
        )]

        for name, err in phase_errors.items():
            issues.append(f"[ERROR] phase.{name}_crashed: {err}")
        if issues:
            self.log.warning("post-pipeline issues remain: %s", issues)
            if self.console:
                self.console.issue_list(issues)

        if self.console:
            self.console.run_footer(
                events=len(state.events),
                errors=len(state.errors),
                failures=len(state.failures),
                edges=len(state.edges),
                anomalies=len(state.anomaly),
                elapsed_s=total_elapsed,
                issues_remaining=len(issues),
                token_totals=self.accountant.totals(),
            )

        return DiagnosisResult(
            graph=graph,
            issues=issues,
            timings=timings,
            critic_log=list(state.critic_log),
            accountant=self.accountant,
            elapsed_s=total_elapsed,
            trace_chars_sent=len(state.normalized_trace),
            trace_chars_full=state.trace_chars_full,
            trace_truncated=state.trace_truncated,
        )
