import os
import re
import json
import argparse
import threading
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


_TEXT_SIM_MODE = os.environ.get("CEG_TEXT_SIM", "lexical").lower()
_EMBED_MODEL_PATH = os.environ.get(
    "CEG_EMBED_MODEL",
    "/checkpoints/bge-m3"
)

_embed_lock = threading.Lock()
_embed_state = {"tokenizer": None, "model": None, "device": None}
_embed_cache: Dict[str, Any] = {}


def set_text_sim_mode(mode: str, model_path: Optional[str] = None) -> None:
    global _TEXT_SIM_MODE, _EMBED_MODEL_PATH
    mode = mode.lower()
    if mode not in ("lexical", "semantic"):
        raise ValueError(f"unknown text sim mode: {mode}")
    _TEXT_SIM_MODE = mode
    if model_path:
        _EMBED_MODEL_PATH = model_path


def _load_embedder():
    if _embed_state["model"] is not None:
        return
    with _embed_lock:
        if _embed_state["model"] is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(_EMBED_MODEL_PATH)
        mdl = AutoModel.from_pretrained(_EMBED_MODEL_PATH)
        mdl.eval()
        mdl.to(device)
        _embed_state["tokenizer"] = tok
        _embed_state["model"] = mdl
        _embed_state["device"] = device


def _embed_one(text: str):
    import torch
    cached = _embed_cache.get(text)
    if cached is not None:
        return cached
    _load_embedder()
    tok = _embed_state["tokenizer"]
    mdl = _embed_state["model"]
    dev = _embed_state["device"]
    with torch.no_grad():
        enc = tok(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = mdl(**enc)
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vec = emb[0].detach().cpu().numpy()
    _embed_cache[text] = vec
    return vec


def _semantic_similarity(a: str, b: str) -> float:
    va = _embed_one(a)
    vb = _embed_one(b)
    return float((va * vb).sum())


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _TEXT_SIM_MODE == "semantic":
        try:
            return _semantic_similarity(a, b)
        except Exception:
            return SequenceMatcher(None, a, b).ratio()
    return SequenceMatcher(None, a, b).ratio()


def safe_get(d: Dict, key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def get_event_index(event_id: str) -> Optional[int]:
    if not event_id:
        return None
    m = re.match(r"E(\d+)", str(event_id))
    if not m:
        return None
    return int(m.group(1))


def event_proximity(event_a: str, event_b: str) -> float:
    ia = get_event_index(event_a)
    ib = get_event_index(event_b)
    if ia is None or ib is None:
        return 0.0
    dist = abs(ia - ib)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.5
    return 0.0


def f1_score(pred_count: int, gold_count: int, match_count: int) -> Dict[str, float]:
    if pred_count == 0 and gold_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if pred_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if gold_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p = match_count / pred_count
    r = match_count / gold_count
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def greedy_bipartite_match(
    sim_matrix: List[List[float]],
    threshold: float
) -> List[Tuple[int, int, float]]:
    pairs = []
    for i, row in enumerate(sim_matrix):
        for j, score in enumerate(row):
            if score >= threshold:
                pairs.append((i, j, score))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_gold = set()
    used_pred = set()
    matches = []

    for i, j, score in pairs:
        if i not in used_gold and j not in used_pred:
            matches.append((i, j, score))
            used_gold.add(i)
            used_pred.add(j)

    return matches


def hungarian_match(
    sim_matrix: List[List[float]],
    threshold: float
) -> List[Tuple[int, int, float]]:
    if not sim_matrix or not sim_matrix[0]:
        return []

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        mat = np.array(sim_matrix)
        row_ind, col_ind = linear_sum_assignment(-mat)

        matches = []
        for i, j in zip(row_ind, col_ind):
            score = float(mat[i, j])
            if score >= threshold:
                matches.append((int(i), int(j), score))

        return matches

    except Exception:
        return greedy_bipartite_match(sim_matrix, threshold)


def event_to_text(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return str(event)

    preferred_keys = [
        "thought",
        "action",
        "observation",
        "input",
        "output",
        "tool",
        "content",
        "description",
        "summary",
    ]

    parts = []
    for k in preferred_keys:
        v = event.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            parts.append(str(v))
        else:
            parts.append(json.dumps(v, ensure_ascii=False, sort_keys=True))

    if parts:
        return " ".join(parts)

    fallback_parts = []
    for k, v in event.items():
        if k == "id":
            continue
        if isinstance(v, (str, int, float, bool)):
            fallback_parts.append(str(v))
        else:
            fallback_parts.append(json.dumps(v, ensure_ascii=False, sort_keys=True))

    return " ".join(fallback_parts)


def relative_position(idx: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return idx / (total - 1)


def event_similarity(
    gold_event: Dict[str, Any],
    pred_event: Dict[str, Any],
    gold_idx: int,
    pred_idx: int,
    gold_total: int,
    pred_total: int,
    weights: Dict[str, float] = None
) -> float:
    if weights is None:
        weights = {
            "text": 0.75,
            "position": 0.20,
            "id": 0.05,
        }

    gold_text = event_to_text(gold_event)
    pred_text = event_to_text(pred_event)
    text_sim = text_similarity(gold_text, pred_text)

    gp = relative_position(gold_idx, gold_total)
    pp = relative_position(pred_idx, pred_total)
    pos_sim = 1.0 - abs(gp - pp)
    pos_sim = clamp01(pos_sim)

    id_sim = event_proximity(
        safe_get(gold_event, "id", ""),
        safe_get(pred_event, "id", "")
    )

    score = (
        weights["text"] * text_sim
        + weights["position"] * pos_sim
        + weights["id"] * id_sim
    )

    return clamp01(score)


def longest_non_decreasing_subsequence_length(seq: List[int]) -> int:
    if not seq:
        return 0

    n = len(seq)
    dp = [1] * n

    for i in range(n):
        for j in range(i):
            if seq[j] <= seq[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def align_events(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    threshold: float = 0.55
) -> Dict[str, Any]:
    gold_events = gold_graph.get("events", []) or []
    pred_events = pred_graph.get("events", []) or []

    gold_count = len(gold_events)
    pred_count = len(pred_events)

    if gold_count == 0 and pred_count == 0:
        return {
            "sim_matrix": [],
            "matches": [],
            "gold_to_pred": {},
            "pred_to_gold": {},
            "gold_count": 0,
            "pred_count": 0,
            "matched_count": 0,
            "event_precision": 1.0,
            "event_recall": 1.0,
            "event_f1": 1.0,
            "gold_coverage": 1.0,
            "pred_coverage": 1.0,
            "order_consistency": 1.0,
            "event_score": 1.0,
            "pair_scores": {},
        }

    if gold_count == 0 or pred_count == 0:
        return {
            "sim_matrix": [],
            "matches": [],
            "gold_to_pred": {},
            "pred_to_gold": {},
            "gold_count": gold_count,
            "pred_count": pred_count,
            "matched_count": 0,
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "gold_coverage": 0.0,
            "pred_coverage": 0.0,
            "order_consistency": 0.0,
            "event_score": 0.0,
            "pair_scores": {},
        }

    sim_matrix = []

    for i, ge in enumerate(gold_events):
        row = []
        for j, pe in enumerate(pred_events):
            row.append(event_similarity(ge, pe, i, j, gold_count, pred_count))
        sim_matrix.append(row)

    matches = hungarian_match(sim_matrix, threshold)

    gold_to_pred = {}
    pred_to_gold = {}
    pair_scores = {}

    for gi, pi, score in matches:
        gid = gold_events[gi].get("id")
        pid = pred_events[pi].get("id")
        if gid is not None and pid is not None:
            gold_to_pred[gid] = pid
            pred_to_gold[pid] = gid
            pair_scores[(gid, pid)] = score

    event_f1 = f1_score(
        pred_count=pred_count,
        gold_count=gold_count,
        match_count=len(matches)
    )

    gold_best_scores = []
    gold_best_pred_indices = []

    for i in range(gold_count):
        row = sim_matrix[i]
        best_j = max(range(pred_count), key=lambda j: row[j])
        gold_best_scores.append(row[best_j])
        gold_best_pred_indices.append(best_j)

    pred_best_scores = []
    for j in range(pred_count):
        best_score = max(sim_matrix[i][j] for i in range(gold_count))
        pred_best_scores.append(best_score)

    gold_coverage = sum(gold_best_scores) / gold_count
    pred_coverage = sum(pred_best_scores) / pred_count


    lnds = longest_non_decreasing_subsequence_length(gold_best_pred_indices)
    order_consistency = lnds / max(1, gold_count)

    event_score = (
        0.50 * gold_coverage
        + 0.30 * pred_coverage
        + 0.20 * order_consistency
    )

    return {
        "sim_matrix": sim_matrix,
        "matches": matches,
        "gold_to_pred": gold_to_pred,
        "pred_to_gold": pred_to_gold,
        "gold_count": gold_count,
        "pred_count": pred_count,
        "matched_count": len(matches),
        "event_precision": event_f1["precision"],
        "event_recall": event_f1["recall"],
        "event_f1": event_f1["f1"],
        "gold_coverage": gold_coverage,
        "pred_coverage": pred_coverage,
        "order_consistency": order_consistency,
        "event_score": event_score,
        "pair_scores": pair_scores,
    }


def event_alignment_similarity(
    event_align: Optional[Dict[str, Any]],
    gold_event_id: str,
    pred_event_id: str,
    fallback: bool = True
) -> float:
    if not gold_event_id or not pred_event_id:
        return 0.0

    if event_align:
        gold_ids = event_align.get("_gold_event_id_to_idx", {})
        pred_ids = event_align.get("_pred_event_id_to_idx", {})
        sim_matrix = event_align.get("sim_matrix", [])

        if gold_event_id in gold_ids and pred_event_id in pred_ids:
            gi = gold_ids[gold_event_id]
            pi = pred_ids[pred_event_id]
            try:
                return clamp01(sim_matrix[gi][pi])
            except Exception:
                pass

    if fallback:
        return event_proximity(gold_event_id, pred_event_id)

    return 0.0


def add_event_index_maps_to_alignment(
    event_align: Dict[str, Any],
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any]
) -> Dict[str, Any]:
    gold_events = gold_graph.get("events", []) or []
    pred_events = pred_graph.get("events", []) or []

    event_align["_gold_event_id_to_idx"] = {
        e.get("id"): i for i, e in enumerate(gold_events) if e.get("id") is not None
    }
    event_align["_pred_event_id_to_idx"] = {
        e.get("id"): i for i, e in enumerate(pred_events) if e.get("id") is not None
    }

    return event_align


def attached_error_to_event_map(graph: Dict[str, Any]) -> Dict[str, str]:
    m = {}
    for ed in graph.get("edges", []) or []:
        if ed.get("type") == "attached_to":
            src = ed.get("source")
            tgt = ed.get("target")
            if src and tgt:
                m[src] = tgt
    return m


def resolve_error_event_id(
    error: Dict[str, Any],
    attached_map: Optional[Dict[str, str]] = None
) -> str:
    eid = safe_get(error, "event_id", "")
    if eid:
        return eid

    if attached_map:
        err_id = safe_get(error, "id", "")
        return attached_map.get(err_id, "")

    return ""


def error_similarity(
    gold_error: Dict[str, Any],
    pred_error: Dict[str, Any],
    event_align: Optional[Dict[str, Any]] = None,
    gold_attached_map: Optional[Dict[str, str]] = None,
    pred_attached_map: Optional[Dict[str, str]] = None,
    weights: Dict[str, float] = None
) -> float:
    if weights is None:
        weights = {
            "event": 0.25,
            "mechanism": 0.20,
            "role": 0.15,
            "sub_mechanism": 0.15,
            "description": 0.25,
        }

    gold_event_id = resolve_error_event_id(gold_error, gold_attached_map)
    pred_event_id = resolve_error_event_id(pred_error, pred_attached_map)

    event_sim = event_alignment_similarity(
        event_align,
        gold_event_id,
        pred_event_id
    )

    mechanism_sim = (
        1.0
        if safe_get(gold_error, "mechanism") == safe_get(pred_error, "mechanism")
        else 0.0
    )

    role_sim = (
        1.0
        if safe_get(gold_error, "role") == safe_get(pred_error, "role")
        else 0.0
    )

    sub_sim = text_similarity(
        safe_get(gold_error, "sub_mechanism", ""),
        safe_get(pred_error, "sub_mechanism", "")
    )

    desc_sim = text_similarity(
        safe_get(gold_error, "description", ""),
        safe_get(pred_error, "description", "")
    )

    score = (
        weights["event"] * event_sim
        + weights["mechanism"] * mechanism_sim
        + weights["role"] * role_sim
        + weights["sub_mechanism"] * sub_sim
        + weights["description"] * desc_sim
    )

    return clamp01(score)


def failure_similarity(
    gold_failure: Dict[str, Any],
    pred_failure: Dict[str, Any]
) -> float:
    type_sim = (
        1.0
        if safe_get(gold_failure, "type") == safe_get(pred_failure, "type")
        else 0.0
    )

    desc_sim = text_similarity(
        safe_get(gold_failure, "description", ""),
        safe_get(pred_failure, "description", "")
    )

    return 0.4 * type_sim + 0.6 * desc_sim


def event_set_similarity(
    gold_event_ids: Set[str],
    pred_event_ids: Set[str],
    event_align: Optional[Dict[str, Any]] = None
) -> float:
    if not gold_event_ids and not pred_event_ids:
        return 1.0
    if not gold_event_ids or not pred_event_ids:
        return 0.0

    if event_align:
        gold_scores = []
        for geid in gold_event_ids:
            best = max(
                event_alignment_similarity(event_align, geid, peid)
                for peid in pred_event_ids
            )
            gold_scores.append(best)

        pred_scores = []
        for peid in pred_event_ids:
            best = max(
                event_alignment_similarity(event_align, geid, peid)
                for geid in gold_event_ids
            )
            pred_scores.append(best)

        recall = sum(gold_scores) / len(gold_scores)
        precision = sum(pred_scores) / len(pred_scores)

        return harmonic_mean(precision, recall)

    return len(gold_event_ids & pred_event_ids) / len(gold_event_ids | pred_event_ids)


def anomaly_similarity(
    gold_anom: Dict[str, Any],
    pred_anom: Dict[str, Any],
    event_align: Optional[Dict[str, Any]] = None
) -> float:
    gold_events = set(safe_get(gold_anom, "event_ids", []) or [])
    pred_events = set(safe_get(pred_anom, "event_ids", []) or [])

    event_sim = event_set_similarity(gold_events, pred_events, event_align)

    text_sim = text_similarity(
        safe_get(gold_anom, "summary", "") + " " + safe_get(gold_anom, "description", ""),
        safe_get(pred_anom, "summary", "") + " " + safe_get(pred_anom, "description", "")
    )

    return 0.4 * event_sim + 0.6 * text_sim


def align_errors(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    event_align: Optional[Dict[str, Any]] = None,
    threshold: float = 0.65
) -> Dict[str, Any]:
    gold_errors = gold_graph.get("errors", []) or []
    pred_errors = pred_graph.get("errors", []) or []

    gold_attached = attached_error_to_event_map(gold_graph)
    pred_attached = attached_error_to_event_map(pred_graph)

    sim_matrix = []
    for ge in gold_errors:
        row = []
        for pe in pred_errors:
            row.append(
                error_similarity(
                    ge,
                    pe,
                    event_align=event_align,
                    gold_attached_map=gold_attached,
                    pred_attached_map=pred_attached
                )
            )
        sim_matrix.append(row)

    matches = hungarian_match(sim_matrix, threshold)

    gold_to_pred = {}
    pred_to_gold = {}
    pair_scores = {}

    for gi, pi, score in matches:
        gid = gold_errors[gi]["id"]
        pid = pred_errors[pi]["id"]
        gold_to_pred[gid] = pid
        pred_to_gold[pid] = gid
        pair_scores[(gid, pid)] = score

    return {
        "matches": matches,
        "gold_to_pred": gold_to_pred,
        "pred_to_gold": pred_to_gold,
        "pair_scores": pair_scores,
        "matched_count": len(matches),
        "gold_count": len(gold_errors),
        "pred_count": len(pred_errors),
    }


def align_failures(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    threshold: float = 0.60
) -> Dict[str, Any]:
    gold_failures = gold_graph.get("failures", []) or []
    pred_failures = pred_graph.get("failures", []) or []

    sim_matrix = []
    for gf in gold_failures:
        row = []
        for pf in pred_failures:
            row.append(failure_similarity(gf, pf))
        sim_matrix.append(row)

    matches = hungarian_match(sim_matrix, threshold)

    gold_to_pred = {}
    pred_to_gold = {}
    pair_scores = {}

    for gi, pi, score in matches:
        gid = gold_failures[gi]["id"]
        pid = pred_failures[pi]["id"]
        gold_to_pred[gid] = pid
        pred_to_gold[pid] = gid
        pair_scores[(gid, pid)] = score

    if not gold_failures and not pred_failures:
        avg_sim = 1.0
    elif not matches:
        avg_sim = 0.0
    else:
        avg_sim = sum(score for _, _, score in matches) / max(
            len(gold_failures),
            len(pred_failures)
        )

    return {
        "matches": matches,
        "gold_to_pred": gold_to_pred,
        "pred_to_gold": pred_to_gold,
        "pair_scores": pair_scores,
        "matched_count": len(matches),
        "gold_count": len(gold_failures),
        "pred_count": len(pred_failures),
        "failure_sim": avg_sim,
    }


def align_anomalies(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    event_align: Optional[Dict[str, Any]] = None,
    threshold: float = 0.60
) -> Dict[str, Any]:
    gold_anoms = gold_graph.get("anomaly", []) or []
    pred_anoms = pred_graph.get("anomaly", []) or []

    sim_matrix = []
    for ga in gold_anoms:
        row = []
        for pa in pred_anoms:
            row.append(anomaly_similarity(ga, pa, event_align=event_align))
        sim_matrix.append(row)

    matches = hungarian_match(sim_matrix, threshold)

    return {
        "matches": matches,
        "matched_count": len(matches),
        "gold_count": len(gold_anoms),
        "pred_count": len(pred_anoms),
    }


def edge_set(
    graph: Dict[str, Any],
    edge_types: Set[str]
) -> Set[Tuple[str, str, str]]:
    edges = graph.get("edges", []) or []
    return {
        (e.get("source"), e.get("target"), e.get("type"))
        for e in edges
        if e.get("type") in edge_types
    }


def mapped_pred_edge_set(
    pred_graph: Dict[str, Any],
    pred_error_to_gold: Dict[str, str],
    pred_failure_to_gold: Dict[str, str],
    edge_types: Set[str]
) -> Set[Tuple[str, str, str]]:
    mapped = set()

    for e in pred_graph.get("edges", []) or []:
        etype = e.get("type")

        if etype not in edge_types:
            continue

        src = e.get("source")
        tgt = e.get("target")

        if etype in {"causes", "amplifies"}:
            if src in pred_error_to_gold and tgt in pred_error_to_gold:
                mapped.add(
                    (
                        pred_error_to_gold[src],
                        pred_error_to_gold[tgt],
                        etype
                    )
                )

        elif etype == "contributes_to":
            if src in pred_error_to_gold and tgt in pred_failure_to_gold:
                mapped.add(
                    (
                        pred_error_to_gold[src],
                        pred_failure_to_gold[tgt],
                        etype
                    )
                )

    return mapped


def edge_f1(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    pred_error_to_gold: Dict[str, str],
    pred_failure_to_gold: Dict[str, str],
    edge_types: Set[str]
) -> Dict[str, float]:
    gold_edges = edge_set(gold_graph, edge_types)

    pred_edges_mapped = mapped_pred_edge_set(
        pred_graph,
        pred_error_to_gold,
        pred_failure_to_gold,
        edge_types
    )

    match_count = len(gold_edges & pred_edges_mapped)

    scores = f1_score(
        len(pred_edges_mapped),
        len(gold_edges),
        match_count
    )

    scores["matched"] = match_count
    scores["gold_edges"] = len(gold_edges)
    scores["pred_edges"] = len(pred_edges_mapped)

    return scores


def build_error_graph(graph: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    adj = defaultdict(list)

    for e in graph.get("edges", []) or []:
        if e.get("type") in {"causes", "amplifies"}:
            adj[e.get("source")].append(
                (
                    e.get("target"),
                    e.get("type")
                )
            )

    return adj


def get_errors_contributing_to_failure(graph: Dict[str, Any]) -> Set[str]:
    result = set()

    for e in graph.get("edges", []) or []:
        if e.get("type") == "contributes_to":
            result.add(e.get("source"))

    return result


def get_error_ids(graph: Dict[str, Any]) -> Set[str]:
    return {
        e["id"]
        for e in graph.get("errors", []) or []
        if "id" in e
    }


def get_root_errors(graph: Dict[str, Any]) -> List[str]:
    roots = []

    for e in graph.get("errors", []) or []:
        if e.get("role") == "root":
            roots.append(e.get("id"))

    if roots:
        return roots

    error_ids = get_error_ids(graph)

    incoming = set()
    for ed in graph.get("edges", []) or []:
        if ed.get("type") in {"causes", "amplifies"}:
            incoming.add(ed.get("target"))

    return list(error_ids - incoming)


def extract_main_chain(graph: Dict[str, Any]) -> List[str]:
    adj = build_error_graph(graph)
    roots = get_root_errors(graph)
    contributing = get_errors_contributing_to_failure(graph)

    best_path = []

    def dfs(node: str, path: List[str], visited: Set[str]):
        nonlocal best_path

        if node in contributing:
            if len(path) > len(best_path):
                best_path = path[:]

        for nxt, _etype in adj.get(node, []):
            if nxt in visited:
                continue

            visited.add(nxt)
            dfs(nxt, path + [nxt], visited)
            visited.remove(nxt)

    for r in roots:
        dfs(r, [r], {r})

    if best_path:
        return best_path

    def dfs_longest(node: str, path: List[str], visited: Set[str]):
        nonlocal best_path

        if len(path) > len(best_path):
            best_path = path[:]

        for nxt, _etype in adj.get(node, []):
            if nxt in visited:
                continue

            visited.add(nxt)
            dfs_longest(nxt, path + [nxt], visited)
            visited.remove(nxt)

    for r in roots:
        dfs_longest(r, [r], {r})

    return best_path


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(
                    dp[i][j + 1],
                    dp[i + 1][j]
                )

    return dp[n][m]


def chain_edges(path: List[str]) -> Set[Tuple[str, str]]:
    return set(zip(path[:-1], path[1:]))


def main_chain_similarity(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    pred_error_to_gold: Dict[str, str]
) -> float:
    gold_chain = extract_main_chain(gold_graph)
    pred_chain_raw = extract_main_chain(pred_graph)

    pred_chain = [
        pred_error_to_gold[x]
        for x in pred_chain_raw
        if x in pred_error_to_gold
    ]

    if not gold_chain and not pred_chain:
        return 1.0

    if not gold_chain or not pred_chain:
        return 0.0

    gold_nodes = set(gold_chain)
    pred_nodes = set(pred_chain)

    node_overlap = len(gold_nodes & pred_nodes) / max(1, len(gold_nodes))

    gold_edges = chain_edges(gold_chain)
    pred_edges = chain_edges(pred_chain)

    edge_overlap = len(gold_edges & pred_edges) / max(1, len(gold_edges))

    lcs = lcs_length(gold_chain, pred_chain)
    order_consistency = lcs / max(1, len(gold_chain))

    return (
        0.4 * node_overlap
        + 0.4 * edge_overlap
        + 0.2 * order_consistency
    )


def validate_graph_consistency(graph: Dict[str, Any]) -> Dict[str, float]:
    events = {
        x["id"]
        for x in graph.get("events", []) or []
        if isinstance(x, dict) and "id" in x
    }

    errors = {
        x["id"]
        for x in graph.get("errors", []) or []
        if isinstance(x, dict) and "id" in x
    }

    failures = {
        x["id"]
        for x in graph.get("failures", []) or []
        if isinstance(x, dict) and "id" in x
    }

    all_nodes = events | errors | failures

    allowed_mechanisms = {
        "representation",
        "planning",
        "execution",
        "evidence_integration",
        "control",
        "self_evaluation",
        "omission",
        "environment",
    }

    allowed_roles = {
        "root",
        "propagated",
        "amplification"
    }

    checks = []

    required = {
        "trace_id",
        "events",
        "errors",
        "failures",
        "edges",
        "anomaly"
    }

    checks.append(
        1.0
        if required.issubset(set(graph.keys()))
        else 0.0
    )

    checks.append(
        1.0
        if len(failures) <= 3
        else 0.0
    )

    attached_map = attached_error_to_event_map(graph)

    for e in graph.get("errors", []) or []:
        checks.append(
            1.0
            if e.get("mechanism") in allowed_mechanisms
            else 0.0
        )

        checks.append(
            1.0
            if e.get("role") in allowed_roles
            else 0.0
        )

        event_id = resolve_error_event_id(e, attached_map)
        checks.append(
            1.0
            if event_id in events
            else 0.0
        )

    for ed in graph.get("edges", []) or []:
        src = ed.get("source")
        tgt = ed.get("target")
        etype = ed.get("type")

        exists = src in all_nodes and tgt in all_nodes
        checks.append(1.0 if exists else 0.0)

        legal = False

        if etype == "event_next":
            legal = src in events and tgt in events
        elif etype == "attached_to":
            legal = src in errors and tgt in events
        elif etype in {"causes", "amplifies"}:
            legal = src in errors and tgt in errors
        elif etype == "contributes_to":
            legal = src in errors and tgt in failures

        checks.append(1.0 if legal else 0.0)

    attached_errors = set()
    for ed in graph.get("edges", []) or []:
        if ed.get("type") == "attached_to":
            attached_errors.add(ed.get("source"))

    for eid in errors:
        checks.append(
            1.0
            if eid in attached_errors
            else 0.0
        )

    incoming = defaultdict(int)
    for ed in graph.get("edges", []) or []:
        if ed.get("type") in {"causes", "amplifies"}:
            incoming[ed.get("target")] += 1

    for e in graph.get("errors", []) or []:
        if e.get("role") == "root":
            checks.append(
                1.0
                if incoming[e["id"]] == 0
                else 0.0
            )

    if not checks:
        return {"consistency": 1.0}

    return {
        "consistency": sum(checks) / len(checks)
    }



def compute_ceg_similarity(
    gold_graph: Dict[str, Any],
    pred_graph: Dict[str, Any],
    event_threshold: float = 0.55,
    error_threshold: float = 0.65,
    failure_threshold: float = 0.60,
    anomaly_threshold: float = 0.60,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    if weights is None:
        weights = {
            "failure": 0.10,
            "error_node": 0.20,
            "attribution": 0.10,
            "causal_edge": 0.20,
            "main_chain": 0.20,
            "anomaly": 0.10,
            "event": 0.05,
            "consistency": 0.05,
        }

    event_align = align_events(
        gold_graph,
        pred_graph,
        threshold=event_threshold
    )

    event_align = add_event_index_maps_to_alignment(
        event_align,
        gold_graph,
        pred_graph
    )

    error_align = align_errors(
        gold_graph,
        pred_graph,
        event_align=event_align,
        threshold=error_threshold
    )

    failure_align = align_failures(
        gold_graph,
        pred_graph,
        threshold=failure_threshold
    )

    anomaly_align = align_anomalies(
        gold_graph,
        pred_graph,
        event_align=event_align,
        threshold=anomaly_threshold
    )

    pred_error_to_gold = error_align["pred_to_gold"]
    pred_failure_to_gold = failure_align["pred_to_gold"]

    error_f1 = f1_score(
        pred_count=error_align["pred_count"],
        gold_count=error_align["gold_count"],
        match_count=error_align["matched_count"]
    )

    anomaly_f1 = f1_score(
        pred_count=anomaly_align["pred_count"],
        gold_count=anomaly_align["gold_count"],
        match_count=anomaly_align["matched_count"]
    )

    failure_sim = failure_align["failure_sim"]

    attribution_scores = edge_f1(
        gold_graph,
        pred_graph,
        pred_error_to_gold,
        pred_failure_to_gold,
        edge_types={"contributes_to"}
    )

    causal_edge_scores = edge_f1(
        gold_graph,
        pred_graph,
        pred_error_to_gold,
        pred_failure_to_gold,
        edge_types={"causes", "amplifies"}
    )

    causes_scores = edge_f1(
        gold_graph,
        pred_graph,
        pred_error_to_gold,
        pred_failure_to_gold,
        edge_types={"causes"}
    )

    amplifies_scores = edge_f1(
        gold_graph,
        pred_graph,
        pred_error_to_gold,
        pred_failure_to_gold,
        edge_types={"amplifies"}
    )

    chain_sim = main_chain_similarity(
        gold_graph,
        pred_graph,
        pred_error_to_gold
    )

    consistency = validate_graph_consistency(pred_graph)["consistency"]

    event_score = event_align["event_score"]

    overall = (
        weights["failure"] * failure_sim
        + weights["error_node"] * error_f1["f1"]
        + weights["attribution"] * attribution_scores["f1"]
        + weights["causal_edge"] * causal_edge_scores["f1"]
        + weights["main_chain"] * chain_sim
        + weights["anomaly"] * anomaly_f1["f1"]
        + weights["event"] * event_score
        + weights["consistency"] * consistency
    )

    return {
        "CEGSim": overall,
        "failure_similarity": failure_sim,
        "event_node": {
            "event_score": event_score,
            "event_precision": event_align["event_precision"],
            "event_recall": event_align["event_recall"],
            "event_f1": event_align["event_f1"],
            "gold_coverage": event_align["gold_coverage"],
            "pred_coverage": event_align["pred_coverage"],
            "order_consistency": event_align["order_consistency"],
            "matched": event_align["matched_count"],
            "gold_events": event_align["gold_count"],
            "pred_events": event_align["pred_count"],
        },
        "error_node": error_f1,
        "anomaly_node": anomaly_f1,
        "error_failure_attribution": attribution_scores,
        "causal_edge": causal_edge_scores,
        "causes_edge": causes_scores,
        "amplifies_edge": amplifies_scores,
        "main_chain_similarity": chain_sim,
        "graph_consistency": consistency,
        "alignment": {
            "event_gold_to_pred": event_align["gold_to_pred"],
            "event_pred_to_gold": event_align["pred_to_gold"],
            "error_gold_to_pred": error_align["gold_to_pred"],
            "error_pred_to_gold": error_align["pred_to_gold"],
            "failure_gold_to_pred": failure_align["gold_to_pred"],
            "failure_pred_to_gold": failure_align["pred_to_gold"],
        },
        "weights": weights,
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_diagnosis_file(case_dir: Path, file_name: str = "diagnoise.json") -> Optional[Path]:
    p = case_dir / file_name
    if p.exists() and p.is_file():
        return p

    alternatives = []

    if file_name != "diagnose.json":
        alternatives.append("diagnose.json")

    if file_name != "diagnoise.json":
        alternatives.append("diagnoise.json")

    for alt in alternatives:
        q = case_dir / alt
        if q.exists() and q.is_file():
            return q

    return None


def discover_case_files(root: Path, file_name: str = "diagnoise.json") -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    cases = {}

    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            case_name = p.stem
            cases[case_name] = p

        elif p.is_dir():
            diagnosis_file = find_diagnosis_file(p, file_name=file_name)
            if diagnosis_file is not None:
                case_name = p.name
                cases[case_name] = diagnosis_file

    return cases


def get_nested_value(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def aggregate_results(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_paths = [
        "CEGSim",
        "failure_similarity",

        "event_node.event_score",
        "event_node.event_precision",
        "event_node.event_recall",
        "event_node.event_f1",
        "event_node.gold_coverage",
        "event_node.pred_coverage",
        "event_node.order_consistency",

        "error_node.precision",
        "error_node.recall",
        "error_node.f1",

        "anomaly_node.precision",
        "anomaly_node.recall",
        "anomaly_node.f1",

        "error_failure_attribution.precision",
        "error_failure_attribution.recall",
        "error_failure_attribution.f1",

        "causal_edge.precision",
        "causal_edge.recall",
        "causal_edge.f1",

        "causes_edge.f1",
        "amplifies_edge.f1",

        "main_chain_similarity",
        "graph_consistency",
    ]

    successful = [
        x for x in per_case
        if x.get("status") == "ok" and isinstance(x.get("scores"), dict)
    ]

    averages = {}

    for path in metric_paths:
        values = []

        for item in successful:
            v = get_nested_value(item["scores"], path)
            if isinstance(v, (int, float)):
                values.append(float(v))

        if values:
            averages[path] = sum(values) / len(values)
        else:
            averages[path] = None

    return {
        "evaluated_cases": len(successful),
        "averages": averages,
    }


def evaluate_folders(
    gold_dir: str,
    pred_dir: str,
    file_name: str = "diagnoise.json",
    output: Optional[str] = None,
    event_threshold: float = 0.55,
    error_threshold: float = 0.65,
    failure_threshold: float = 0.60,
    anomaly_threshold: float = 0.60
) -> Dict[str, Any]:
    
    gold_root = Path(gold_dir)
    pred_root = Path(pred_dir)

    gold_cases = discover_case_files(gold_root, file_name=file_name)
    pred_cases = discover_case_files(pred_root, file_name=file_name)

    all_case_names = sorted(set(gold_cases.keys()) | set(pred_cases.keys()))

    per_case = []

    for case_name in all_case_names:
        item = {
            "case": case_name,
            "status": None,
        }

        if case_name not in gold_cases:
            item["status"] = "missing_gold_file"
            item["error"] = f"Gold json not found for case: {case_name}"
            per_case.append(item)
            continue

        if case_name not in pred_cases:
            item["status"] = "missing_pred_file"
            item["error"] = f"Pred json not found for case: {case_name}"
            per_case.append(item)
            continue

        gold_file = gold_cases[case_name]
        pred_file = pred_cases[case_name]

        item["gold_file"] = str(gold_file)
        item["pred_file"] = str(pred_file)

        try:
            gold_graph = load_json(gold_file)
            pred_graph = load_json(pred_file)

            scores = compute_ceg_similarity(
                gold_graph,
                pred_graph,
                event_threshold=event_threshold,
                error_threshold=error_threshold,
                failure_threshold=failure_threshold,
                anomaly_threshold=anomaly_threshold
            )

            item["status"] = "ok"
            item["scores"] = scores

        except Exception as e:
            item["status"] = "failed"
            item["error"] = repr(e)

        per_case.append(item)

    status_counts = defaultdict(int)
    for item in per_case:
        status_counts[item.get("status")] += 1

    summary = aggregate_results(per_case)
    summary["total_cases"] = len(per_case)
    summary["status_counts"] = dict(status_counts)

    result = {
        "summary": summary,
        "per_case": per_case,
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Causal Error Graph similarity with event-aware scoring."
    )

    parser.add_argument(
        "--gold_dir",
        default="./data/final/",
        help="Gold root folder. It should contain case subfolders."
    )

    parser.add_argument(
        "--pred_dir",
        default="./output/CEG-GLM/",
        help="Pred root folder. It should contain case subfolders."
    )

    parser.add_argument(
        "--file_name",
        default="diagnosis.json",
        help="Diagnosis json filename inside each case folder. Default: diagnosis.json"
    )

    parser.add_argument(
        "--output",
        default="tmp.json",
        help="Output JSON path. If not set, only print summary."
    )

    parser.add_argument(
        "--event_threshold",
        type=float,
        default=0.60,
        help="Threshold for strict one-to-one event matching. Default: 0.60"
    )

    parser.add_argument(
        "--error_threshold",
        type=float,
        default=0.60,
        help="Threshold for error node matching. Default: 0.60"
    )

    parser.add_argument(
        "--failure_threshold",
        type=float,
        default=0.60,
        help="Threshold for failure node matching. Default: 0.60"
    )

    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=0.60,
        help="Threshold for anomaly node matching. Default: 0.60"
    )

    parser.add_argument(
        "--text_sim",
        choices=["semantic"],
        default=os.environ.get("CEG_TEXT_SIM", "semantic"),
        help="semantic = BGE embedding cosine (recommended)."
    )

    parser.add_argument(
        "--embed_model",
        default=os.environ.get(
            "CEG_EMBED_MODEL",
            "/andrew/checkpoints/bge-m3"
        ),
        help="Local path to sentence embedding model when --text_sim=semantic."
    )

    args = parser.parse_args()

    set_text_sim_mode(args.text_sim, model_path=args.embed_model)

    result = evaluate_folders(
        gold_dir=args.gold_dir,
        pred_dir=args.pred_dir,
        file_name=args.file_name,
        output=args.output,
        event_threshold=args.event_threshold,
        error_threshold=args.error_threshold,
        failure_threshold=args.failure_threshold,
        anomaly_threshold=args.anomaly_threshold,
    )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

    if args.output:
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

