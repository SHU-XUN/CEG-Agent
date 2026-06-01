from __future__ import annotations

import argparse
import json
import sys

from ...critic.rules import auto_repair
from ...critic.validators import validate_graph
from ...schemas.graph import CausalErrorGraph


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent.tools.cli.validate_graph")
    parser.add_argument(
        "--input", "-i",
        help="path to draft graph JSON; if omitted, read from stdin",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="report issues without running auto_repair (read-only check)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="soft cap on number of failures before flagging "
             "`failure.too_many` (default 3, matches pipeline default)",
    )
    args = parser.parse_args(argv)

    raw_text = (
        open(args.input, "r", encoding="utf-8").read()
        if args.input else sys.stdin.read()
    )
    draft = json.loads(raw_text)

    graph = CausalErrorGraph.model_validate(draft)
    repair_log: list[str] = []
    if not args.no_repair:
        graph, repair_log = auto_repair(graph)

    issues = validate_graph(graph, max_failures=args.max_failures)

    out = {
        "graph": graph.to_output_dict(),
        "auto_repair_log": repair_log,
        "n_issues": len(issues),
        "issues": [
            {"code": i.code, "severity": i.severity, "message": i.message}
            for i in issues
        ],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
