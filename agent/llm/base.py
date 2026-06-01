from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class ToolCall:

    id: str
    name: str
    arguments: Dict[str, Any]



@dataclass
class Message:

    role: str 
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None 

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: Optional[List[ToolCall]] = None
    ) -> "Message":
        return cls("assistant", content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, call_id: str, name: str, content: str) -> "Message":
        return cls("tool", content, tool_call_id=call_id, name=name)



@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)



@dataclass
class ToolSpec:

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }



class LLMClient(abc.ABC):

    name: str = "abstract"

    @abc.abstractmethod
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


    def chat_json(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        resp = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
        )
        parsed = _extract_json(resp.text)
        if not parsed and resp.text:
            import logging
            logging.getLogger("llm").warning(
                "chat_json: failed to parse JSON from %d-char response "
                "(starts with %r)",
                len(resp.text),
                resp.text[:120],
            )
        return parsed


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    for match in _JSON_FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidate = _largest_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}
    return {}


def _largest_json_object(text: str) -> str:
    best = ""
    stack: List[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                candidate = text[start : i + 1]
                if len(candidate) > len(best):
                    best = candidate
    return best
