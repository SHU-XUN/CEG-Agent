from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple


_BASH_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?THOUGHT(?:\*\*)?\s*:\s*(.*?)(?=\n\s*```|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _split_assistant_content(content: str) -> Tuple[str, str]:
    if not content:
        return "", ""
    bash_blocks = _BASH_BLOCK_RE.findall(content)
    action = "\n".join(b.strip() for b in bash_blocks).strip()
    thought_match = _THOUGHT_RE.search(content)
    if thought_match:
        thought = thought_match.group(1).strip()
    elif bash_blocks:
        head = _BASH_BLOCK_RE.split(content, maxsplit=1)[0]
        thought = head.strip()
    else:
        thought = content.strip()
    return thought, action


def _format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or "unknown_tool"
        args = fn.get("arguments")
        if args is None:
            args = ""
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        parts.append(f"{name}({args})")
    return "\n".join(parts)


def _format_observation_chunk(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(content)
    if msg.get("role") == "tool":
        name = msg.get("name")
        if name:
            return f"[{name}] {content}"
    return content


def _extract_thought_action(msg: Dict[str, Any]) -> Tuple[str, str]:
    reasoning = (msg.get("reasoning_content") or "").strip()
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        action = _format_tool_calls(tool_calls)
        head_text = content.strip()
    else:
        sb_thought, sb_action = _split_assistant_content(content)
        if sb_action:
            action = sb_action
            head_text = sb_thought
        else:
            action = content.strip()
            head_text = ""
    thought = reasoning or head_text
    return thought, action




def _default_max_total_chars() -> int:
    raw = os.getenv("CE_TRACE_MAX_CHARS")
    if not raw:
        return 800_000
    try:
        v = int(raw)
        return v if v > 0 else 800_000
    except ValueError:
        return 800_000




@dataclass
class NormalizedTurn:
    
    index: int                    
    thought: str = ""
    action: str = ""
    observation: str = ""
    obs_chunks: List[str] = dc_field(default_factory=list)




def parse_raw_trace(
    raw: Dict[str, Any],
    *,
    max_total_chars: Optional[int] = None,
) -> Dict[str, Any]:
    
    trace_id = str(raw.get("oid") or raw.get("trace_id") or "unknown")
    messages = raw.get("messages") or []

    first_user = ""
    start_idx = 0
    if messages and messages[0].get("role") == "user":
        first_user = (messages[0].get("content") or "").strip()
        start_idx = 1

    task = (raw.get("task") or first_user or "").strip()

    turns = _build_turns(messages, start_idx)

    limit = max_total_chars if max_total_chars is not None else _default_max_total_chars()
    text, chars_full = render_turns(turns, max_total_chars=limit)
    return {
        "trace_id": trace_id,
        "task": task,
        "turns": turns,
        "text": text,
        "chars": len(text),
        "chars_full": chars_full,
        "truncated": chars_full > len(text),
    }


def _build_turns(messages: List[Dict[str, Any]], start_idx: int) -> List[NormalizedTurn]:
    out: List[NormalizedTurn] = []
    n = len(messages)
    i = start_idx
    while i < n:
        msg = messages[i]
        if msg.get("role") != "assistant":
            i += 1
            continue
        thought, action = _extract_thought_action(msg)

        j = i + 1
        chunks: List[str] = []
        while j < n and messages[j].get("role") != "assistant":
            piece = _format_observation_chunk(messages[j])
            if piece:
                chunks.append(piece)
            j += 1
        observation = "\n".join(chunks)

        out.append(NormalizedTurn(
            index=i,
            thought=thought,
            action=action,
            observation=observation,
            obs_chunks=chunks,
        ))
        i = j
    return out




def render_turns(
    turns: List[NormalizedTurn],
    *,
    max_total_chars: Optional[int] = None,
) -> Tuple[str, int]:
    
    full_text = _render_full(turns)
    chars_full = len(full_text)
    if max_total_chars is None or chars_full <= max_total_chars:
        return full_text, chars_full

    half = max_total_chars // 2
    head = full_text[:half]
    tail = full_text[-half:] if half else ""
    return head + "\n...[truncated]...\n" + tail, chars_full


def _render_full(turns: List[NormalizedTurn]) -> str:
    lines: List[str] = []
    for step_no, t in enumerate(turns, 1):
        lines.append(f"--- step {step_no} (msg #{t.index}) ---")
        if t.thought:
            lines.append(f"THOUGHT: {t.thought}")
        if t.action:
            lines.append(f"ACTION:  {t.action}")
        if t.obs_chunks:
            lines.append("OBSERVATION: " + "\n  | ".join(t.obs_chunks))
        elif t.observation:
            lines.append(f"OBSERVATION: {t.observation}")
    return "\n".join(lines)
