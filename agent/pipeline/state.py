from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..schemas import (
    AnomalyNode,
    CausalErrorGraph,
    Edge,
    ErrorNode,
    EventNode,
    FailureNode,
)
from ..utils.trace_parser import NormalizedTurn


@dataclass
class PipelineState:
    trace_id: str
    task: Optional[str] = None
    turns: List[NormalizedTurn] = field(default_factory=list)
    normalized_trace: str = ""
    trace_chars_full: int = 0
    trace_truncated: bool = False

    events: List[EventNode] = field(default_factory=list)
    failures: List[FailureNode] = field(default_factory=list)
    errors: List[ErrorNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    anomaly: List[AnomalyNode] = field(default_factory=list)

    critic_log: List[str] = field(default_factory=list)

   
    def event_ids(self) -> Set[str]:
        return {e.id for e in self.events}

    def error_ids(self) -> Set[str]:
        return {r.id for r in self.errors}

    def failure_ids(self) -> Set[str]:
        return {f.id for f in self.failures}

   
    def to_graph(self) -> CausalErrorGraph:
        return CausalErrorGraph(
            trace_id=self.trace_id,
            task=self.task,
            events=list(self.events),
            errors=list(self.errors),
            failures=list(self.failures),
            edges=list(self.edges),
            anomaly=list(self.anomaly),
        )

    def load_from_graph(self, graph: CausalErrorGraph) -> None:
        self.events = list(graph.events)
        self.errors = list(graph.errors)
        self.failures = list(graph.failures)
        self.edges = list(graph.edges)
        self.anomaly = list(graph.anomaly)
