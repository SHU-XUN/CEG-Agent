from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from ...utils.trace_parser import parse_raw_trace


def _turn_to_dict(turn: Any) -> Dict[str, Any]:
    return {
        "thought": turn.thought,
        "action": turn.action,
        "observation": turn.observation,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent.tools.cli.read_trace")
    parser.add_argument(
        "--input", "-i",
        help="path to raw trace JSON; if omitted, read from stdin",
    )
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="emit only the events array (E1..En), one object per line "
             "(JSONL), suitable for piping into other tools",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="omit the full rendered text field — useful when only the "
             "event list is needed and the trace is large",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="hard total-char ceiling on the rendered text (overrides "
             "CE_TRACE_MAX_CHARS, default 800000)",
    )
    args = parser.parse_args(argv)

    raw_text = (
        open(args.input, "r", encoding="utf-8").read()
        if args.input else sys.stdin.read()
    )
    raw = json.loads(raw_text)
    parsed = parse_raw_trace(raw, max_total_chars=args.max_chars)

    events = [
        {"id": f"E{i + 1}", **_turn_to_dict(t)}
        for i, t in enumerate(parsed["turns"])
    ]

    if args.events_only:
        for ev in events:
            print(json.dumps(ev, ensure_ascii=False))
        return 0

    out = {
        "trace_id": parsed["trace_id"],
        "task": parsed["task"],
        "n_events": len(events),
        "events": events,
        "chars": parsed["chars"],
        "chars_full": parsed["chars_full"],
        "truncated": parsed["truncated"],
    }
    if not args.no_text:
        out["text"] = parsed["text"]
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
