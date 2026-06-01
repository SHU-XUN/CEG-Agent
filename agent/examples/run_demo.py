from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agent.pipeline import DiagnosisOrchestrator  
from agent.utils import load_json 


def main() -> int:
    sample = os.path.join(ROOT, "data", "init", "1007.json")
    if not os.path.exists(sample):
        raw = {
            "oid": "demo",
            "task": "A small synthetic task.",
            "messages": [
                {"role": "user", "content": "demo task"},
                {
                    "role": "assistant",
                    "reasoning_content": "plan a search",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}
                    ],
                    "content": "",
                },
                {"role": "tool", "content": "irrelevant result"},
            ],
        }
    else:
        raw = load_json(sample)

    orch = DiagnosisOrchestrator()
    result = orch.diagnose(raw)
    print(json.dumps(result.graph.to_output_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
