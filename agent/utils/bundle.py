from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO

from ..schemas import CausalErrorGraph
from ..viz import render_dot, render_html, render_svg
from .display import Console
from .tokens import TokenAccountant


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


@dataclass
class BundleWriter:
    root_dir: str           
    trace_id: str           
    bundle_dir: str = field(init=False)
    _log_buffer: io.StringIO = field(default_factory=io.StringIO, init=False)
    _console_buffer: io.StringIO = field(default_factory=io.StringIO, init=False)
    _log_handler: Optional[logging.Handler] = field(default=None, init=False)
    _console: Optional[Console] = field(default=None, init=False)
    _orig_console_streams: List[TextIO] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.bundle_dir = os.path.join(self.root_dir, str(self.trace_id))
        os.makedirs(self.bundle_dir, exist_ok=True)

    def __enter__(self) -> "BundleWriter":
        self._attach_logger()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._detach_logger()
        self._restore_console()

    def _attach_logger(self) -> None:
        h = logging.StreamHandler(self._log_buffer)
        h.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        h.setLevel(logging.DEBUG) 
        logging.getLogger().addHandler(h)
        self._log_handler = h

    def _detach_logger(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    def attach_console(self, console: Console) -> None:
        self._console = console
        self._orig_console_streams = [console.stream]
        original_w = console._w

        buf = self._console_buffer

        def _tee(s: str = "", end: str = "\n") -> None:
            original_w(s, end=end)
            buf.write(_strip_ansi(s) + end)

        console._w = _tee  
        self._original_w = original_w  

    def _restore_console(self) -> None:
        if self._console is not None and hasattr(self, "_original_w"):
            self._console._w = self._original_w  


    def write(
        self,
        graph: CausalErrorGraph,
        *,
        accountant: Optional[TokenAccountant] = None,
        summary_extra: Optional[Dict[str, Any]] = None,
        tool_call_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}

        paths["diagnosis"] = self._write(
            "diagnosis.json",
            json.dumps(graph.to_output_dict(), ensure_ascii=False, indent=2) + "\n",
        )

        log_text = (
            "=== console (pretty step-by-step) ===\n"
            + self._console_buffer.getvalue()
            + "\n=== logger output ===\n"
            + self._log_buffer.getvalue()
        )
        paths["log"] = self._write("run.log", log_text)

        if accountant is not None:
            lines = [
                json.dumps(c.to_dict(), ensure_ascii=False)
                for c in accountant.calls
            ]
            paths["tool_trace"] = self._write("tool_trace.jsonl", "\n".join(lines) + ("\n" if lines else ""))

        summary = {
            "trace_id": graph.trace_id,
            "task": graph.task,
            "counts": {
                "events": len(graph.events),
                "errors": len(graph.errors),
                "failures": len(graph.failures),
                "edges": len(graph.edges),
                "anomalies": len(graph.anomaly),
            },
        }
        if accountant is not None:
            tot = accountant.totals()
            summary["tokens"] = {
                "calls": tot.calls,
                "input_tokens": tot.input_tokens,
                "output_tokens": tot.output_tokens,
                "total_tokens": tot.total_tokens,
                "elapsed_s": round(tot.elapsed_s, 3),
                "by_label": tot.by_label,
            }
        if summary_extra:
            summary.update(summary_extra)
        if tool_call_log:
            summary["agentic_steps"] = len(tool_call_log)
        paths["summary"] = self._write(
            "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )

        if tool_call_log:
            lines = [json.dumps(s, ensure_ascii=False) for s in tool_call_log]
            paths["agentic_steps"] = self._write(
                "agentic_steps.jsonl", "\n".join(lines) + "\n"
            )

        dot_src = render_dot(graph)
        paths["dot"] = self._write("graph.dot", dot_src)

        svg = render_svg(dot_src)
        if svg is not None:
            paths["svg"] = self._write("graph.svg", svg)

        html_doc = render_html(graph)
        paths["html"] = self._write("graph.html", html_doc)

        return paths

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.bundle_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
