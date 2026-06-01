from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, List, Optional

from .config import load_config
from .pipeline import AgenticDiagnosisOrchestrator, DiagnosisOrchestrator
from .utils import (
    DEBUG,
    NORMAL,
    QUIET,
    VERBOSE,
    BundleWriter,
    Console,
    NullConsole,
    dump_json,
    get_logger,
    load_json,
    set_global_level,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.main",
        description="Causal Error Graph diagnosis pipeline.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", "-i", help="path to a single raw trace JSON")
    grp.add_argument(
        "--input-dir", "-I", help="directory of raw trace JSONs to process"
    )
    p.add_argument(
        "--output",
        "-o",
        help="single JSON output path (legacy mode; only the diagnosis graph)",
    )
    p.add_argument(
        "--output-dir",
        "-O",
        help="bundle dir; each trace becomes <DIR>/<trace_id>/{diagnosis.json,"
             "run.log,tool_trace.jsonl,summary.json,graph.dot,graph.html,...}",
    )
    p.add_argument(
        "--mode",
        "-m",
        choices=["pipeline", "agentic"],
        default="agentic",
        help="execution mode (default: agentic)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=24,
        help="max agent loop iterations (agentic mode only)",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="limit number of files in batch mode",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort batch on the first error instead of skipping",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip traces whose <out_dir>/<trace_id>/diagnosis.json already "
            "exists and is valid JSON. Prior index.json rows are merged into "
            "the new one so the final index keeps a full picture."
        ),
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "filter the input list to only the rows whose status != 'ok' in "
            "<out_dir>/index.json. Prior ok rows are preserved verbatim in "
            "the new index. Requires an existing index.json."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="-v = verbose tool results, -vv = debug (raw JSON), default = normal",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="only emit final summary line; suppress per-step rendering",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI color in step-by-step rendering",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the pretty step-by-step renderer entirely (use logging only)",
    )
    p.add_argument(
        "--trace-max-chars",
        type=int,
        default=None,
        help=(
            "max chars of the rendered trace to feed the LLM. Default 800000 "
            "(or CE_TRACE_MAX_CHARS). The renderer preserves every field in "
            "full; the ceiling is only a last-resort guardrail against extreme "
            "outliers — when exceeded, the middle is replaced by a "
            "`[truncated]` marker. `summary.json` always records the original "
            "size + truncated flag."
        ),
    )
    return p


def _resolve_verbosity(args: argparse.Namespace) -> int:
    if args.quiet:
        return QUIET
    if args.verbose >= 2:
        return DEBUG
    if args.verbose == 1:
        return VERBOSE
    return NORMAL


def _build_console(args: argparse.Namespace) -> Console:
    if args.no_progress:
        return NullConsole()
    use_color = None
    if args.no_color:
        use_color = False
    return Console(
        stream=sys.stderr,
        verbosity=_resolve_verbosity(args),
        use_color=use_color,
    )


def _build_orchestrator(args: argparse.Namespace, console: Console) -> Any:
    cfg = load_config()
    if args.mode == "agentic":
        return AgenticDiagnosisOrchestrator(
            cfg, max_iters=args.max_iters, console=console
        )
    return DiagnosisOrchestrator(cfg, console=console)


def _run_one(
    orch: Any,
    in_path: str,
    *,
    out_file: Optional[str] = None,
    out_dir: Optional[str] = None,
    console: Optional[Console] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    raw = load_json(in_path)
    trace_id = str(raw.get("oid") or raw.get("trace_id") or "unknown")

    if out_dir is not None:
        with BundleWriter(out_dir, trace_id) as bundle:
            if console is not None:
                bundle.attach_console(console)
            if hasattr(orch, "bundle_dir"):
                orch.bundle_dir = bundle.bundle_dir
            t0 = time.time()
            result = orch.diagnose(raw)
            wall_s = round(time.time() - t0, 4)
            extra: dict = {
                "mode": getattr(args, "mode", None),
                "backend": getattr(orch.llm, "name", None),
                "model": orch.config.llm.model,
                "elapsed_s": round(getattr(result, "elapsed_s", 0.0) or 0.0, 3),
                "wall_elapsed_s": wall_s,
                "finalized": getattr(result, "finalized", None),
                "iterations": getattr(result, "iterations", None),
                "issues_remaining": len(result.issues),
                "trace": {
                    "chars_sent": getattr(result, "trace_chars_sent", 0),
                    "chars_full": getattr(result, "trace_chars_full", 0),
                    "truncated": bool(getattr(result, "trace_truncated", False)),
                },
            }
            timings = getattr(result, "timings", None)
            if timings:
                extra["phase_timings_s"] = {k: round(v, 4) for k, v in timings.items()}
            tool_call_log = getattr(result, "tool_call_log", None) or []
            if tool_call_log:
                extra["iteration_timings_s"] = [
                    {
                        "iteration": s["iteration"],
                        "elapsed_s": s.get("elapsed_s", 0.0),
                        "tools": [tc["name"] for tc in s.get("tool_calls", [])],
                    }
                    for s in tool_call_log
                ]
            paths = bundle.write(
                result.graph,
                accountant=getattr(result, "accountant", None),
                summary_extra=extra,
                tool_call_log=tool_call_log,
            )
            if console is not None:
                console.notice(
                    f"wrote bundle → {bundle.bundle_dir}/  "
                    f"({', '.join(sorted(os.path.basename(p) for p in paths.values()))})"
                )
        return {
            "trace_id": trace_id,
            "bundle_dir": bundle.bundle_dir,
            "wall_elapsed_s": wall_s,
            "diagnose_elapsed_s": extra["elapsed_s"],
            "finalized": extra["finalized"],
            "iterations": extra["iterations"],
            "counts": {
                "events": len(result.graph.events),
                "errors": len(result.graph.errors),
                "failures": len(result.graph.failures),
                "edges": len(result.graph.edges),
                "anomalies": len(result.graph.anomaly),
            },
            "issues_remaining": extra["issues_remaining"],
            "tokens": (
                {
                    "calls": result.accountant.totals().calls,
                    "input_tokens": result.accountant.totals().input_tokens,
                    "output_tokens": result.accountant.totals().output_tokens,
                }
                if getattr(result, "accountant", None) else None
            ),
        }

    result = orch.diagnose(raw)
    if out_file:
        dump_json(result.graph.to_output_dict(), out_file)
    return None


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_config()
    if args.output and args.output_dir:
        print(
            "ERROR: --output and --output-dir are mutually exclusive. "
            "Use --output FILE for a single JSON, or --output-dir DIR for "
            "the per-trace bundle (recommended).",
            file=sys.stderr,
        )
        return 2
    if args.input_dir and args.output:
        print(
            "ERROR: --output FILE doesn't make sense with --input-dir. "
            "Use --output-dir DIR for batch mode.",
            file=sys.stderr,
        )
        return 2

    if args.quiet:
        set_global_level("WARNING")
    elif args.verbose >= 2:
        set_global_level("DEBUG")
    else:
        set_global_level(cfg.pipeline.log_level)

    if not args.no_progress and args.verbose < 2:
        import logging
        for noisy in (
            "agent_loop",
            "orchestrator",
            "agentic_orchestrator",
            "agent.event_builder",
            "agent.failure_analyzer",
            "agent.error_hypothesis",
            "agent.causal_graph",
            "agent.anomaly_detector",
            "agent.graph_critic",
            "agent.repair_estimator",
            "subagent",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    log = get_logger("main")
    console = _build_console(args)
    if args.trace_max_chars is not None:
        os.environ["CE_TRACE_MAX_CHARS"] = str(args.trace_max_chars)
    orch = _build_orchestrator(args, console)

    if args.input:
        if args.output_dir:
            if args.resume and _bundle_already_done(args.input, args.output_dir):
                log.info("skip %s (resume): bundle already exists", args.input)
                return 0
            _run_one(
                orch, args.input, out_dir=args.output_dir, console=console, args=args
            )
        else:
            out_file = args.output or _default_out(args.input)
            _run_one(orch, args.input, out_file=out_file, args=args)
        return 0

    in_dir = args.input_dir
    out_dir = args.output_dir
    if not out_dir:
        log.error("--output-dir is required with --input-dir")
        return 2
    files = sorted(
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.endswith(".json")
    )

    prior_rows_by_input: dict = {}
    if args.resume or args.retry_failed:
        idx_path = os.path.join(out_dir, "index.json")
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    old_index = json.load(f)
                for r in old_index.get("rows", []):
                    if r.get("input"):
                        prior_rows_by_input[r["input"]] = r
            except Exception as e:  # noqa: BLE001
                log.warning("could not read prior index.json: %s", e)
        elif args.retry_failed:
            log.error(
                "--retry-failed needs an existing index.json at %s", idx_path
            )
            return 2

    if args.retry_failed:
        failed_inputs = {
            inp for inp, r in prior_rows_by_input.items()
            if r.get("status") != "ok"
        }
        files = [f for f in files if os.path.basename(f) in failed_inputs]
        log.info("retry-failed: %d previously-failed traces", len(files))

    if args.limit:
        files = files[: args.limit]
    log.info("batch: %d files to process", len(files))
    batch_t0 = time.time()
    rows_by_input: dict = dict(prior_rows_by_input)
    processed = 0
    skipped = 0
    for in_path in files:
        basename = os.path.basename(in_path)
        if args.resume and _bundle_already_done(in_path, out_dir):
            log.info("  skip %s (resume: bundle exists)", basename)
            skipped += 1
            if basename not in rows_by_input:
                rows_by_input[basename] = {
                    "input": basename,
                    "status": "ok",
                    "note": "skipped (resume)",
                }
            continue
        try:
            row = _run_one(
                orch, in_path, out_dir=out_dir, console=console, args=args
            )
            processed += 1
            log.info(
                "  ok  %s (%.2fs)",
                basename,
                row["wall_elapsed_s"] if row else 0.0,
            )
            if row is not None:
                rows_by_input[basename] = {
                    "input": basename, "status": "ok", **row
                }
        except Exception as e:  # noqa: BLE001
            log.exception("  FAIL %s", basename)
            rows_by_input[basename] = {
                "input": basename,
                "status": "fail",
                "error": f"{type(e).__name__}: {e}",
            }
            if args.stop_on_error:
                rows = sorted(
                    rows_by_input.values(), key=lambda r: r.get("input", "")
                )
                ok = sum(1 for r in rows if r.get("status") == "ok")
                _write_batch_index(
                    out_dir, rows, time.time() - batch_t0, ok, len(rows)
                )
                return 1
    batch_elapsed = time.time() - batch_t0
    rows = sorted(rows_by_input.values(), key=lambda r: r.get("input", ""))
    ok = sum(1 for r in rows if r.get("status") == "ok")
    _write_batch_index(out_dir, rows, batch_elapsed, ok, len(rows))
    log.info(
        "batch finished: %d processed (+%d skipped), %d/%d ok overall in %.2fs",
        processed, skipped, ok, len(rows), batch_elapsed,
    )
    return 0 if ok == len(rows) else 1


def _bundle_already_done(in_path: str, out_dir: str) -> bool:
    basename = os.path.basename(in_path)
    try:
        raw_peek = load_json(in_path)
        trace_id = str(
            raw_peek.get("oid")
            or raw_peek.get("trace_id")
            or os.path.splitext(basename)[0]
        )
    except Exception:
        trace_id = os.path.splitext(basename)[0]
    diag_path = os.path.join(out_dir, trace_id, "diagnosis.json")
    if not os.path.exists(diag_path):
        return False
    try:
        with open(diag_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _write_batch_index(
    out_dir: str, rows: List[dict], elapsed_s: float, ok: int, total: int
) -> None:
    successful = [r for r in rows if r.get("status") == "ok"]
    timings = [r.get("wall_elapsed_s", 0.0) for r in successful]
    tokens = [r.get("tokens") for r in successful if r.get("tokens")]
    index = {
        "n_files": total,
        "n_ok": ok,
        "n_fail": total - ok,
        "elapsed_s": round(elapsed_s, 3),
        "trace_timing_s": (
            {
                "min": round(min(timings), 3),
                "max": round(max(timings), 3),
                "mean": round(sum(timings) / len(timings), 3),
                "p50": round(sorted(timings)[len(timings) // 2], 3),
                "total": round(sum(timings), 3),
            }
            if timings else None
        ),
        "tokens_total": (
            {
                "calls": sum(t["calls"] for t in tokens),
                "input_tokens": sum(t["input_tokens"] for t in tokens),
                "output_tokens": sum(t["output_tokens"] for t in tokens),
            }
            if tokens else None
        ),
        "rows": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _default_out(in_path: str) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(os.getcwd(), f"{base}.diagnosed.json")


if __name__ == "__main__":
    sys.exit(main())
