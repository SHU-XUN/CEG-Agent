from __future__ import annotations

import time
from typing import List, Optional

from ..utils.tokens import TokenAccountant
from .base import LLMClient, LLMResponse, Message, ToolSpec


class TrackedLLMClient(LLMClient):

    def __init__(
        self,
        inner: LLMClient,
        accountant: TokenAccountant,
        model: Optional[str] = None,
    ):
        self.inner = inner
        self.accountant = accountant
        self.name = inner.name
        self._model = model

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
        t0 = time.time()
        resp = self.inner.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
        )
        elapsed = time.time() - t0
        self.accountant.record(
            resp.usage or {}, model=self._model, elapsed_s=elapsed
        )
        return resp
