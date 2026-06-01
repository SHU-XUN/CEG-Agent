from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List



def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _obs_clip_threshold() -> int:
    return _int_env("CE_OBS_CLIP_THRESHOLD", 1500)


def _obs_clip_keep() -> int:
    return _int_env("CE_OBS_CLIP_KEEP", 600)


def _keep_tail_events() -> int:
    return _int_env("CE_KEEP_TAIL_EVENTS", 1)



_COMPACT_SEPARATORS = (",", ":")


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=_COMPACT_SEPARATORS)


def _clip_observation(obs: str) -> str:
    threshold = _obs_clip_threshold()
    if not obs or len(obs) <= threshold:
        return obs
    keep = _obs_clip_keep()
    head = obs[:keep]
    tail = obs[-keep:]
    elided = len(obs) - 2 * keep
    return f"{head}\n...[{elided} chars elided]...\n{tail}"


def _compact_event_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    obs = d.get("observation")
    if isinstance(obs, str):
        clipped = _clip_observation(obs)
        if clipped is not obs:
            d = dict(d)
            d["observation"] = clipped
    return d


def compact_events_json(events: Iterable[Any]) -> str:
    items = list(events)
    n = len(items)
    tail_keep = max(0, _keep_tail_events())
    tail_start = max(0, n - tail_keep)
    payload: List[Dict[str, Any]] = []
    for idx, e in enumerate(items):
        d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
        if idx < tail_start:
            d = _compact_event_dict(d)
        payload.append(d)
    return compact_json(payload)


def compact_models_json(items: Iterable[Any], *, mode: str = "python") -> str:
    payload = [
        m.model_dump(mode=mode) if hasattr(m, "model_dump") else dict(m)
        for m in items
    ]
    return compact_json(payload)
