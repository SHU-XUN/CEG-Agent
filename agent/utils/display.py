from __future__ import annotations

import io
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO



_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
}


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CE_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())



QUIET = 0   
NORMAL = 1 
VERBOSE = 2 
DEBUG = 3   



@dataclass
class Console:
    stream: TextIO = sys.stderr
    verbosity: int = NORMAL
    use_color: Optional[bool] = None  

    def __post_init__(self):
        if self.use_color is None:
            self.use_color = _supports_color(self.stream)


    def _c(self, code: str, s: str) -> str:
        if not self.use_color:
            return s
        return _ANSI[code] + s + _ANSI["reset"]

    def _w(self, s: str = "", end: str = "\n") -> None:
        self.stream.write(s + end)
        self.stream.flush()

    def _rule(self, char: str = "═", width: int = 72) -> str:
        return char * width


    def run_header(self, *, title: str, fields: Dict[str, Any]) -> None:
        if self.verbosity < NORMAL:
            return
        self._w()
        self._w(self._c("cyan", self._rule()))
        self._w("  " + self._c("bold", title))
        for k, v in fields.items():
            self._w(f"  {self._c('dim', k+':'):<22} {v}")
        self._w(self._c("cyan", self._rule()))

    def run_footer(
        self,
        *,
        events: int,
        errors: int,
        failures: int,
        edges: int,
        anomalies: int,
        iterations: Optional[int] = None,
        elapsed_s: Optional[float] = None,
        finalized: Optional[bool] = None,
        issues_remaining: int = 0,
        token_totals: Optional[Any] = None,  
        bundle_dir: Optional[str] = None,
    ) -> None:
        if self.verbosity < QUIET:
            return
        self._w()
        self._w(self._c("cyan", self._rule()))
        summary = (
            f"events={events}  errors={errors}  failures={failures}  "
            f"edges={edges}  anomalies={anomalies}"
        )
        self._w("  " + self._c("bold", "Result: ") + summary)
        meta_bits: List[str] = []
        if iterations is not None:
            meta_bits.append(f"iterations={iterations}")
        if elapsed_s is not None:
            meta_bits.append(f"elapsed={elapsed_s:.2f}s")
        if finalized is not None:
            meta_bits.append(
                ("finalized=" + ("✓" if finalized else "✗"))
                if self.use_color
                else f"finalized={finalized}"
            )
        if issues_remaining:
            meta_bits.append(
                self._c("yellow", f"issues_remaining={issues_remaining}")
            )
        if meta_bits:
            self._w("  " + "  ".join(meta_bits))
        if token_totals is not None and getattr(token_totals, "calls", 0):
            self._w(
                "  "
                + self._c("bold", "Tokens: ")
                + f"calls={token_totals.calls}  "
                + f"in={token_totals.input_tokens}  "
                + f"out={token_totals.output_tokens}  "
                + f"total={token_totals.total_tokens}"
            )
            if self.verbosity >= VERBOSE and token_totals.by_label:
                for label, agg in sorted(
                    token_totals.by_label.items(),
                    key=lambda kv: -(kv[1]["input_tokens"] + kv[1]["output_tokens"]),
                ):
                    el = agg.get("elapsed_s", 0.0) or 0.0
                    self._w(
                        "    "
                        + self._c("dim", f"{label:<32}")
                        + f"calls={agg['calls']:<3}  "
                        + f"in={agg['input_tokens']:<6}  "
                        + f"out={agg['output_tokens']:<6}  "
                        + f"total={agg['input_tokens']+agg['output_tokens']:<6}  "
                        + f"elapsed={el:.2f}s"
                    )
        if bundle_dir:
            self._w("  " + self._c("dim", "Bundle:  ") + bundle_dir)
        self._w(self._c("cyan", self._rule()))
        self._w()


    def phase_start(self, *, phase: int, name: str) -> None:
        if self.verbosity < NORMAL:
            return
        self._w()
        marker = self._c("blue", f"▶ phase {phase}")
        self._w(f"{marker}  {self._c('bold', name)}")

    def phase_done(
        self,
        *,
        summary: str,
        elapsed_s: Optional[float] = None,
        tokens: Optional[Any] = None,  
    ) -> None:
        if self.verbosity < NORMAL:
            return
        suffix_parts: List[str] = []
        if elapsed_s is not None:
            suffix_parts.append(f"{elapsed_s:.2f}s")
        if tokens and getattr(tokens, "calls", 0):
            suffix_parts.append(
                f"tok in={tokens.input_tokens} out={tokens.output_tokens}"
            )
        suffix = self._c("dim", "  (" + ", ".join(suffix_parts) + ")") if suffix_parts else ""
        self._w(f"  {self._c('green', '✓')} {summary}{suffix}")


    def iter_start(self, *, iteration: int, assistant_text: str = "") -> None:
        if self.verbosity < NORMAL:
            return
        self._w()
        self._w(f"{self._c('blue', '▶')} {self._c('bold', f'iter {iteration}')}")
        if assistant_text and self.verbosity >= VERBOSE:
            for line in textwrap.wrap(
                assistant_text.strip(), width=68,
                initial_indent="  " + self._c("dim", "say  "),
                subsequent_indent="       ",
            ):
                self._w(line)

    def tool_call(self, *, name: str, arguments: Dict[str, Any]) -> None:
        if self.verbosity < NORMAL:
            return
        arg_repr = _format_args(arguments)
        self._w(f"  {self._c('magenta', '→')} {self._c('bold', name)}({arg_repr})")

    def tool_result(
        self,
        *,
        name: str,
        result: Dict[str, Any],
        is_error: bool = False,
        token_call: Optional[Any] = None,  
    ) -> None:
        if self.verbosity < NORMAL:
            return
        if is_error or "error" in result:
            self._w("  " + self._c("red", "✗ ") + json.dumps(result, ensure_ascii=False))
            return
        renderer = _RENDERERS.get(name, _render_default)
        lines = renderer(result, self.verbosity)
        for line in lines:
            self._w("    " + line)
        if token_call is not None:
            self._w(
                "    "
                + self._c(
                    "dim",
                    f"tok in={token_call.input_tokens} out={token_call.output_tokens} "
                    f"({token_call.elapsed_s:.2f}s)",
                )
            )
        if self.verbosity >= DEBUG:
            self._w(self._c("dim", "    raw: ") + json.dumps(result, ensure_ascii=False)[:600])


    def subagent_start(self, *, focus: str) -> None:
        if self.verbosity < NORMAL:
            return
        self._w("  " + self._c("yellow", "↳ spawn subagent: ") + focus)

    def subagent_done(self, *, focus: str, finding: str) -> None:
        if self.verbosity < NORMAL:
            return
        finding_clip = (finding or "").strip().replace("\n", " ")
        if len(finding_clip) > 200:
            finding_clip = finding_clip[:197] + "..."
        self._w("  " + self._c("yellow", "↳ subagent done: ") + finding_clip)


    def issue_list(self, issues: List[str]) -> None:
        if not issues or self.verbosity < NORMAL:
            return
        self._w("  " + self._c("yellow", "issues:"))
        for s in issues:
            self._w("    - " + s)

    def repair_log(self, repairs: List[str]) -> None:
        if not repairs or self.verbosity < VERBOSE:
            return
        self._w("  " + self._c("dim", "auto_repair:"))
        for r in repairs:
            self._w("    " + self._c("dim", "- " + r))

    def notice(self, msg: str) -> None:
        if self.verbosity < NORMAL:
            return
        self._w("  " + self._c("dim", msg))

    def warn(self, msg: str) -> None:
        if self.verbosity < QUIET:
            return
        self._w(self._c("yellow", "WARN: ") + msg)

    def error(self, msg: str) -> None:
        self._w(self._c("red", "ERROR: ") + msg)




def _format_args(args: Dict[str, Any], limit: int = 80) -> str:
    if not args:
        return ""
    if len(args) == 1:
        k, v = next(iter(args.items()))
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else f'"{v}"'
        if len(s) > limit:
            s = s[: limit - 3] + "..."
        return f"{k}={s}"
    parts = []
    for k, v in args.items():
        sv = json.dumps(v, ensure_ascii=False)
        if len(sv) > 30:
            sv = sv[:27] + "..."
        parts.append(f"{k}={sv}")
    out = ", ".join(parts)
    return out if len(out) <= limit else out[: limit - 3] + "..."


def _render_default(result: Dict[str, Any], verbosity: int) -> List[str]:
    s = json.dumps(result, ensure_ascii=False)
    limit = 800 if verbosity >= VERBOSE else 200
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return [s]


def _render_build_events(result: Dict[str, Any], verbosity: int) -> List[str]:
    lines = [f"ok  n_events={result.get('n_events', '?')}"]
    if verbosity >= VERBOSE:
        for ev in (result.get("events") or [])[:10]:
            lines.append(f"  {ev.get('id'):>4}  {ev.get('action_preview','')}")
        more = max(0, len(result.get("events") or []) - 10)
        if more:
            lines.append(f"  ... ({more} more)")
    return lines


def _render_find_failures(result: Dict[str, Any], verbosity: int) -> List[str]:
    failures = result.get("failures") or []
    lines = [f"ok  n_failures={result.get('n_failures', len(failures))}"]
    for f in failures:
        desc = (f.get("description") or "").replace("\n", " ")
        if len(desc) > 100 and verbosity < VERBOSE:
            desc = desc[:97] + "..."
        lines.append(f"  {f.get('id')} [{f.get('type')}]: {desc}")
    return lines


def _render_propose_errors(result: Dict[str, Any], verbosity: int) -> List[str]:
    errs = result.get("errors") or []
    lines = [f"ok  n_errors={result.get('n_errors', len(errs))}"]
    for r in errs:
        sub = r.get("sub_mechanism") or ""
        if len(sub) > 60:
            sub = sub[:57] + "..."
        lines.append(
            f"  {r.get('id')} [{r.get('mechanism')}|{r.get('role')}] "
            f"→ {r.get('event_id')}: {sub}"
        )
    return lines


def _render_build_causal_graph(result: Dict[str, Any], verbosity: int) -> List[str]:
    lines = [f"ok  n_edges={result.get('n_edges', '?')}"]
    by = result.get("by_type") or {}
    if by:
        bits = [f"{k}={v}" for k, v in sorted(by.items())]
        lines.append("  " + "  ".join(bits))
    return lines


def _render_find_anomalies(result: Dict[str, Any], verbosity: int) -> List[str]:
    anos = result.get("anomalies") or []
    lines = [f"ok  n_anomalies={result.get('n_anomalies', len(anos))}"]
    for a in anos:
        events = ",".join(a.get("events", []))
        summary = a.get("summary") or ""
        if len(summary) > 80 and verbosity < VERBOSE:
            summary = summary[:77] + "..."
        lines.append(f"  {a.get('id')} ({events}): {summary}")
    return lines


def _render_validate_graph(result: Dict[str, Any], verbosity: int) -> List[str]:
    n = result.get("n_issues", 0)
    lines = [f"ok  n_issues={n}"]
    for issue in (result.get("issues") or [])[:8]:
        sev = issue.get("severity", "warning")
        sev_short = "ERR" if sev == "error" else "warn"
        lines.append(f"  [{sev_short}] {issue.get('code')}: {issue.get('message')}")
    repairs = result.get("auto_repair_log") or []
    if repairs and verbosity >= VERBOSE:
        lines.append("  auto_repair:")
        for r in repairs:
            lines.append("    - " + r)
    elif repairs:
        lines.append(f"  auto_repair applied {len(repairs)} change(s)")
    return lines


def _render_apply_critic_patch(result: Dict[str, Any], verbosity: int) -> List[str]:
    return [
        f"ok  n_issues_remaining={result.get('n_issues_remaining', '?')}",
    ]


def _render_estimate_repair(result: Dict[str, Any], verbosity: int) -> List[str]:
    rv = result.get("repair_values") or {}
    if not rv:
        return ["ok  (no repair_values assigned)"]
    bits = [f"{k}={v}" for k, v in rv.items()]
    return ["ok  " + "  ".join(bits)]


def _render_finalize(result: Dict[str, Any], verbosity: int) -> List[str]:
    s = result.get("summary") or {}
    return [
        "✓ FINALIZED  "
        f"events={s.get('events', '?')}  errors={s.get('errors', '?')}  "
        f"failures={s.get('failures', '?')}  edges={s.get('edges', '?')}  "
        f"anomalies={s.get('anomalies', '?')}"
    ]


def _render_render_graph(result: Dict[str, Any], verbosity: int) -> List[str]:
    if "error" in result:
        return [f"✗ {result['error']}"]
    formats = result.get("formats") or {}
    bits: List[str] = []
    for fmt, info in formats.items():
        if fmt == "svg" and not info.get("ok", True):
            bits.append(f"svg=skipped({info.get('reason', '?')})")
        elif "bytes" in info:
            bits.append(f"{fmt}={info['bytes']}B")
        else:
            bits.append(fmt)
    line = "ok  " + "  ".join(bits)
    out = [line]
    written = result.get("written") or {}
    if written:
        out.append(
            f"  wrote {len(written)} file(s) → {result.get('output_dir')}"
        )
        if verbosity >= VERBOSE:
            for fname, path in written.items():
                out.append(f"    {fname}")
    return out


def _render_read_trace(result: Dict[str, Any], verbosity: int) -> List[str]:
    return [
        f"ok  trace_id={result.get('trace_id')}  chars={result.get('chars')}  "
        f"truncated={result.get('truncated', False)}"
    ]


def _render_inspect_state(result: Dict[str, Any], verbosity: int) -> List[str]:
    out = []
    if "events" in result:
        out.append(f"events={len(result['events'])}")
    if "errors" in result:
        out.append(f"errors={len(result['errors'])}")
    if "failures" in result:
        out.append(f"failures={len(result['failures'])}")
    if "edges" in result:
        out.append(f"edges={len(result['edges'])}")
    if "anomaly" in result:
        out.append(f"anomaly={len(result['anomaly'])}")
    if "todos" in result:
        out.append(f"todos={len(result['todos'])}")
    return ["ok  " + "  ".join(out)] if out else ["ok"]


def _render_write_todo(result: Dict[str, Any], verbosity: int) -> List[str]:
    todos = result.get("todos") or []
    lines = [f"ok  n_todos={len(todos)}"]
    if verbosity >= NORMAL:
        for t in todos:
            status = t.get("status", "?")
            mark = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(
                status, "?"
            )
            lines.append(f"  {mark} {t.get('id')}: {t.get('content')}")
    return lines


_RENDERERS = {
    "build_events":          _render_build_events,
    "find_failures":         _render_find_failures,
    "propose_errors":        _render_propose_errors,
    "build_causal_graph":    _render_build_causal_graph,
    "find_anomalies":        _render_find_anomalies,
    "validate_graph":        _render_validate_graph,
    "apply_critic_patch":    _render_apply_critic_patch,
    "estimate_repair_values":_render_estimate_repair,
    "finalize":              _render_finalize,
    "render_graph":          _render_render_graph,
    "read_trace":            _render_read_trace,
    "inspect_state":         _render_inspect_state,
    "write_todo":            _render_write_todo,
}



class NullConsole(Console):

    def __init__(self):
        super().__init__(stream=io.StringIO(), verbosity=-1, use_color=False)

    def _w(self, s: str = "", end: str = "\n") -> None:  # noqa: D401
        return
