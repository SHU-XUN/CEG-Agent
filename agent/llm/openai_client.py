from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec


class OpenAILLMClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = -1,
        max_retries: int = 2,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is not installed. Either `pip install openai` "
                "or use CE_LLM_BACKEND=mock for offline runs."
            ) from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
        kwargs: dict = {
            "model": self.model,
            "messages": [_msg_to_openai(m) for m in messages],
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
        }
        eff_max = max_tokens if max_tokens is not None else self.max_tokens
        if eff_max and eff_max > 0:
            kwargs["max_tokens"] = eff_max
        if response_format == "json_object" and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content or ""
                tool_calls: List[ToolCall] = []
                for tc in (getattr(msg, "tool_calls", None) or []):
                    args_raw = tc.function.arguments or "{}"
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"_raw": args_raw}
                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                    )
                usage = {}
                if getattr(resp, "usage", None):
                    usage = {
                        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                    }
                return LLMResponse(
                    text=content, tool_calls=tool_calls, usage=usage, raw=resp
                )
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        assert last_err is not None
        raise last_err


def _msg_to_openai(m: Message) -> Dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content,
        }
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content}
