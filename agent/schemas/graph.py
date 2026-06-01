from __future__ import annotations

from enum import Enum
from typing import List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator




class Mechanism(str, Enum):
    REPRESENTATION = "representation"
    PLANNING = "planning"
    EXECUTION = "execution"
    EVIDENCE_INTEGRATION = "evidence_integration"
    CONTROL = "control"
    SELF_EVALUATION = "self_evaluation"
    OMISSION = "omission"
    ENVIRONMENT = "environment"


class Role(str, Enum):
    ROOT = "root"
    PROPAGATED = "propagated"
    AMPLIFICATION = "amplification"


class FailureType(str, Enum):
    CORRECTNESS = "correctness"
    COMPLETION = "completion"
    CONSTRAINT = "constraint"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    OTHER = "other"



_FAILURE_TYPE_ALIASES = {
    "constraint_violation": FailureType.CONSTRAINT,
    "repo_pollution":       FailureType.CONSTRAINT,
    "patch_hygiene":        FailureType.CONSTRAINT,
    "unrelated_changes":    FailureType.CONSTRAINT,
    "unauthorized_action":  FailureType.SAFETY,
    "cost":                 FailureType.EFFICIENCY,
}


def coerce_failure_type(raw) -> FailureType:
    if isinstance(raw, FailureType):
        return raw
    s = (str(raw or "")).strip().lower()
    if not s:
        return FailureType.OTHER
    if s in _FAILURE_TYPE_ALIASES:
        return _FAILURE_TYPE_ALIASES[s]
    try:
        return FailureType(s)
    except ValueError:
        return FailureType.OTHER


class EdgeType(str, Enum):
    EVENT_NEXT = "event_next"
    ATTACHED_TO = "attached_to"
    CAUSES = "causes"
    AMPLIFIES = "amplifies"
    CONTRIBUTES_TO = "contributes_to"


class RepairValue(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


LEGAL_EDGE_ENDPOINTS = {
    EdgeType.EVENT_NEXT: ("E", "E"),
    EdgeType.ATTACHED_TO: ("R", "E"),
    EdgeType.CAUSES: ("R", "R"),
    EdgeType.AMPLIFIES: ("R", "R"),
    EdgeType.CONTRIBUTES_TO: ("R", "F"),
}




class EventNode(BaseModel):

    id: str = Field(..., pattern=r"^E\d+$")
    thought: str = ""
    action: str = ""
    observation: str = ""


class ErrorNode(BaseModel):

    id: str = Field(..., pattern=r"^R\d+$")
    event_id: str = Field(..., pattern=r"^E\d+$")
    mechanism: Mechanism
    sub_mechanism: str
    role: Role
    description: str
    repair_value: Optional[RepairValue] = None


class FailureNode(BaseModel):
    id: str = Field(..., pattern=r"^F\d+$")
    type: FailureType
    description: str

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        return coerce_failure_type(v)


class AnomalyNode(BaseModel):
    id: str = Field(..., pattern=r"^A\d+$")
    event_ids: List[str] = Field(default_factory=list)
    summary: str
    description: str




class Edge(BaseModel):
    source: str
    target: str
    type: EdgeType

    @model_validator(mode="after")
    def _endpoints_match_edge_type(self) -> "Edge":
        src_prefix, tgt_prefix = self.source[:1], self.target[:1]
        expected = LEGAL_EDGE_ENDPOINTS[self.type]
        if (src_prefix, tgt_prefix) != expected:
            raise ValueError(
                f"Illegal edge {self.source}->{self.target} for type "
                f"{self.type.value}: expected prefixes {expected}, got "
                f"({src_prefix}, {tgt_prefix})"
            )
        return self




class CausalErrorGraph(BaseModel):

    trace_id: str
    task: Optional[str] = None
    events: List[EventNode] = Field(default_factory=list)
    errors: List[ErrorNode] = Field(default_factory=list)
    failures: List[FailureNode] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    anomaly: List[AnomalyNode] = Field(default_factory=list)


    def node_ids(self) -> Set[str]:
        ids: Set[str] = set()
        ids.update(e.id for e in self.events)
        ids.update(e.id for e in self.errors)
        ids.update(f.id for f in self.failures)
        ids.update(a.id for a in self.anomaly)
        return ids

    def event_ids(self) -> Set[str]:
        return {e.id for e in self.events}

    def error_ids(self) -> Set[str]:
        return {r.id for r in self.errors}

    def failure_ids(self) -> Set[str]:
        return {f.id for f in self.failures}

    def to_output_dict(self) -> dict:
        out: dict = {
            "trace_id": self.trace_id,
        }
        if self.task is not None:
            out["task"] = self.task
        out["events"] = [e.model_dump() for e in self.events]
        out["errors"] = [
            {k: v for k, v in r.model_dump(mode="json").items() if v is not None}
            for r in self.errors
        ]
        out["failures"] = [f.model_dump(mode="json") for f in self.failures]
        out["edges"] = [e.model_dump(mode="json") for e in self.edges]
        out["anomaly"] = [a.model_dump() for a in self.anomaly]
        return out
