from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Set

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec


_PHASE_ORDER = [
    "build_events",
    "find_failures",
    "propose_errors",
    "build_causal_graph",
    "find_anomalies",
    "validate_graph",
    "estimate_repair_values",
    "render_graph",
    "finalize",
]


class MockLLMClient(LLMClient):
    name = "mock"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        if tools:
            resp = self._simulate_tool_use(messages, tools)
        else:
            phase = _detect_phase(messages)
            handler = getattr(self, f"_handle_{phase}", self._handle_default)
            payload = handler(messages)
            resp = LLMResponse(text=json.dumps(payload), raw=payload)
        resp.usage = _fake_usage(messages, resp)
        return resp

    def _simulate_tool_use(
        self, messages: List[Message], tools: List[ToolSpec]
    ) -> LLMResponse:
        available: Set[str] = {t.name for t in tools}
        called: Set[str] = set()
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    called.add(tc.name)

        for name in _PHASE_ORDER:
            if name in called or name not in available:
                continue
            args = self._args_for(name, messages)
            return _single_tool_call(name, args)

        if "finalize" in available and "finalize" not in called:
            return _single_tool_call("finalize", {})
        return LLMResponse(text="(mock) all phases complete; nothing more to do.")

    @staticmethod
    def _args_for(name: str, messages: List[Message]) -> Dict[str, Any]:
        if name == "render_graph":
            return {"formats": ["dot", "html", "svg"], "write": True}
        return {}

    def _handle_event_builder(self, messages: List[Message]) -> dict:
        return {
            "events": [
                {
                    "id": "E1",
                    "thought": "(mock) consider the task",
                    "action": "(mock) plan an action",
                    "observation": "(mock) record an observation",
                }
            ]
        }

    def _handle_failure_analyzer(self, messages: List[Message]) -> dict:
        return {
            "failures": [
                {
                    "id": "F1",
                    "type": "completion",
                    "description": "(mock) task did not reach a final answer.",
                }
            ]
        }

    def _handle_error_hypothesis(self, messages: List[Message]) -> dict:
        return {
            "errors": [
                {
                    "id": "R1",
                    "event_id": "E1",
                    "mechanism": "control",
                    "sub_mechanism": "no finish action",
                    "role": "root",
                    "description": "(mock) agent never emitted a finish event.",
                }
            ]
        }

    def _handle_causal_graph(self, messages: List[Message]) -> dict:
        return {
            "edges": [
                {"source": "R1", "target": "F1", "type": "contributes_to"},
            ]
        }

    def _handle_anomaly_detector(self, messages: List[Message]) -> dict:
        return {"anomaly": []}

    def _handle_graph_critic(self, messages: List[Message]) -> dict:
        return {"patch": {}, "issues": []}

    def _handle_repair_estimator(self, messages: List[Message]) -> dict:
        return {"repair_values": {"R1": "high"}}

    def _handle_default(self, messages: List[Message]) -> dict:
        return {}



def _detect_phase(messages: List[Message]) -> str:
    for msg in messages:
        if msg.role != "system":
            continue
        for line in msg.content.splitlines():
            line = line.strip()
            if line.startswith("[PHASE:") and line.endswith("]"):
                return line[len("[PHASE:") : -1].strip().lower()
    return "default"


def _single_tool_call(name: str, args: Dict[str, Any]) -> LLMResponse:
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
    )


def _fake_usage(messages: List[Message], resp: LLMResponse) -> Dict[str, int]:
    in_chars = 0
    for m in messages:
        in_chars += len(m.content or "")
        if m.tool_calls:
            for tc in m.tool_calls:
                in_chars += len(tc.name) + len(json.dumps(tc.arguments))
    out_chars = len(resp.text or "")
    if resp.tool_calls:
        for tc in resp.tool_calls:
            out_chars += len(tc.name) + len(json.dumps(tc.arguments))
    return {
        "input_tokens": max(1, in_chars // 4),
        "output_tokens": max(1, out_chars // 4),
    }
