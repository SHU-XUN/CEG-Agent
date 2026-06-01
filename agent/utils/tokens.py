from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TokenCall:
    label: str         
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    model: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    by_label: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenAccountant:

    def __init__(self):
        self._calls: List[TokenCall] = []
        self._stack: List[str] = []
        self._lock = threading.Lock()


    @contextmanager
    def label(self, name: str) -> Iterator[None]:
        with self._lock:
            self._stack.append(name)
        try:
            yield
        finally:
            with self._lock:
                if self._stack:
                    if self._stack[-1] != name:
                        import logging
                        logging.getLogger("tokens").warning(
                            "label stack out-of-order pop: top=%r, popping=%r",
                            self._stack[-1], name,
                        )
                    self._stack.pop()

    def current_label(self) -> str:
        with self._lock:
            return "/".join(self._stack) if self._stack else "default"


    def record(
        self,
        usage: Dict[str, int],
        *,
        model: Optional[str] = None,
        elapsed_s: float = 0.0,
        label: Optional[str] = None,
    ) -> TokenCall:
        call = TokenCall(
            label=label or self.current_label(),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            elapsed_s=elapsed_s,
            model=model,
        )
        with self._lock:
            self._calls.append(call)
        return call


    @property
    def calls(self) -> List[TokenCall]:
        with self._lock:
            return list(self._calls)

    def totals(self) -> TokenTotals:
        with self._lock:
            calls = list(self._calls)
        out = TokenTotals(calls=len(calls))
        for c in calls:
            out.input_tokens += c.input_tokens
            out.output_tokens += c.output_tokens
            out.elapsed_s += c.elapsed_s
            entry = out.by_label.setdefault(
                c.label,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0},
            )
            entry["calls"] += 1
            entry["input_tokens"] += c.input_tokens
            entry["output_tokens"] += c.output_tokens
            entry["elapsed_s"] = round(entry["elapsed_s"] + c.elapsed_s, 4)
        return out

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
