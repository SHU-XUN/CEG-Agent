from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

DIMENSIONS: List[str] = [
    "event",
    "event_next",
    "failure",
    "error",
    "attached_to",
    "causes",
    "contributes_to",
]

_CAUSAL_EDGE_TYPES = {"causes", "amplifies"}


def _f1(matched: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = matched / max(1, n_pred)
    r = matched / max(1, n_gold)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _edge_set(edges: List[dict], types: Set[str]) -> Set[Tuple[str, str, str]]:
    return {
        (e["source"], e["target"], e["type"])
        for e in edges
        if e.get("type") in types
        and "source" in e and "target" in e
    }


def _eval_event(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    g_ids = {e["id"] for e in gold.get("events", []) or []}
    p_ids = {e["id"] for e in pred.get("events", []) or []}
    matched = len(g_ids & p_ids)
    p, r, f1 = _f1(matched, len(p_ids), len(g_ids))
    return {
        "gold_count": len(g_ids),
        "pred_count": len(p_ids),
        "matched": matched,
        "missing_in_pred": sorted(g_ids - p_ids)[:20],
        "extra_in_pred": sorted(p_ids - g_ids)[:20],
        "precision": round(p, 3),
        "recall": round(r, 3),
        "score": round(f1, 3),
    }


def _eval_event_next(
    gold: Dict[str, Any], pred: Dict[str, Any], parent_score: float
) -> Dict[str, Any]:
    g_set = _edge_set(gold.get("edges", []) or [], {"event_next"})
    p_set = _edge_set(pred.get("edges", []) or [], {"event_next"})
    tp = len(g_set & p_set)
    p, r, raw_f1 = _f1(tp, len(p_set), len(g_set))
    return {
        "gold_count": len(g_set),
        "pred_count": len(p_set),
        "matched": tp,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "raw_f1": round(raw_f1, 3),
        "parent_score": round(parent_score, 3),
        "cascaded_score": round(raw_f1 * parent_score, 3),
        "score": round(raw_f1, 3),
    }


def _eval_failure(
    gold: Dict[str, Any], pred: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    gold_failures = gold.get("failures", []) or []
    pred_failures = pred.get("failures", []) or []
    g_types = Counter(f.get("type") for f in gold_failures)
    p_types = Counter(f.get("type") for f in pred_failures)
    tp = sum((g_types & p_types).values())
    p, r, f1 = _f1(tp, sum(p_types.values()), sum(g_types.values()))

    pairs: List[Tuple[str, str]] = []
    pred_used = set()
    for g in gold_failures:
        for i, pr in enumerate(pred_failures):
            if i in pred_used:
                continue
            if pr.get("type") == g.get("type"):
                pairs.append((g["id"], pr["id"]))
                pred_used.add(i)
                break

    return {
        "gold_count": len(gold_failures),
        "pred_count": len(pred_failures),
        "type_intersection": tp,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "score": round(f1, 3),
    }, pairs


def _eval_error(
    gold: Dict[str, Any], pred: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    gold_errors = gold.get("errors", []) or []
    pred_errors = pred.get("errors", []) or []

    pred_by_key: Dict[Tuple[Optional[str], Optional[str]], List[int]] = {}
    for i, p in enumerate(pred_errors):
        key = (p.get("event_id"), (p.get("mechanism") or "").lower() or None)
        pred_by_key.setdefault(key, []).append(i)

    pred_used: Set[int] = set()
    pairs: List[Tuple[str, str]] = []
    for g in gold_errors:
        gkey = (g.get("event_id"), (g.get("mechanism") or "").lower() or None)
        for pi in pred_by_key.get(gkey, []):
            if pi not in pred_used:
                pred_used.add(pi)
                pairs.append((g["id"], pred_errors[pi]["id"]))
                break

    G, P = len(gold_errors), len(pred_errors)
    matched = len(pairs)
    p, r, f1 = _f1(matched, P, G)

    if pairs:
        gold_by = {g["id"]: g for g in gold_errors}
        pred_by = {pr["id"]: pr for pr in pred_errors}
        role_match = sum(
            1 for gid, pid in pairs
            if (gold_by[gid].get("role") or "").lower()
               == (pred_by[pid].get("role") or "").lower()
        ) / len(pairs)
    else:
        role_match = 0.0

    score = round(0.7 * f1 + 0.3 * role_match, 3)
    return {
        "gold_count": G,
        "pred_count": P,
        "matched": matched,
        "matched_pairs": [{"gold_id": g, "pred_id": p_} for g, p_ in pairs],
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "role_acc_on_matched": round(role_match, 3),
        "score": score,
    }, pairs


def _translate_edge_set(
    edges: List[dict],
    types: Set[str],
    err_translate: Dict[str, str],
    fail_translate: Dict[str, str],
    *,
    translate_source: bool,
    translate_target: bool,
) -> Tuple[Set[Tuple[str, str, str]], int, int]:
    translated: Set[Tuple[str, str, str]] = set()
    n_total = 0
    n_dropped = 0
    for e in edges:
        ty = e.get("type")
        if ty not in types:
            continue
        n_total += 1
        s, t = e.get("source"), e.get("target")
        if translate_source:
            s2 = err_translate.get(s) or fail_translate.get(s)
            if s2 is None:
                n_dropped += 1
                continue
            s = s2
        if translate_target:
            t2 = err_translate.get(t) or fail_translate.get(t)
            if t2 is None:
                n_dropped += 1
                continue
            t = t2
        translated.add((s, t, ty))
    return translated, n_total, n_dropped


def _eval_edges(
    gold: Dict[str, Any],
    pred: Dict[str, Any],
    types: Set[str],
    error_pairs: List[Tuple[str, str]],
    failure_pairs: List[Tuple[str, str]],
    *,
    translate_source: bool,
    translate_target: bool,
    parent_score: float,
) -> Dict[str, Any]:
    err_translate = {pid: gid for gid, pid in error_pairs}
    fail_translate = {pid: gid for gid, pid in failure_pairs}
    g_set = _edge_set(gold.get("edges", []) or [], types)
    p_set, p_total, p_dropped = _translate_edge_set(
        pred.get("edges", []) or [], types,
        err_translate, fail_translate,
        translate_source=translate_source,
        translate_target=translate_target,
    )
    tp = len(g_set & p_set)
    fp = max(0, len(p_set) - tp) + p_dropped
    fn = max(0, len(g_set) - tp)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    raw_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    cascaded = round(raw_f1 * parent_score, 3)

    return {
        "gold_count": len(g_set),
        "pred_count_raw": p_total,
        "pred_count_translated": len(p_set),
        "pred_dropped_unresolvable_as_fp": p_dropped,
        "matched": tp,
        "false_positive": round(fp, 3),
        "false_negative": fn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "raw_f1": round(raw_f1, 3),
        "parent_score": round(parent_score, 3),
        "cascaded_score": cascaded,
        "score": round(raw_f1, 3),
    }


def score_one(gold_doc: Dict[str, Any], pred_doc: Dict[str, Any]) -> Dict[str, Any]:
    event_report   = _eval_event(gold_doc, pred_doc)
    failure_report, failure_pairs = _eval_failure(gold_doc, pred_doc)

    event_next_report = _eval_event_next(
        gold_doc, pred_doc, parent_score=event_report["score"],
    )
    error_report, error_pairs = _eval_error(gold_doc, pred_doc)
    error_raw = error_report.get("score", 0.0)
    error_report["raw_f1"] = error_raw
    error_report["parent_score"] = event_report["score"]
    error_report["cascaded_score"] = round(error_raw * event_report["score"], 3)
    error_report["score"] = round(error_raw, 3)

    err_score = error_report["score"]
    fail_score = failure_report["score"]
    parent_for_contrib = min(err_score, fail_score)

    attached_report = _eval_edges(
        gold_doc, pred_doc, {"attached_to"}, error_pairs, failure_pairs,
        translate_source=True, translate_target=False,
        parent_score=err_score,
    )
    causes_report = _eval_edges(
        gold_doc, pred_doc, _CAUSAL_EDGE_TYPES, error_pairs, failure_pairs,
        translate_source=True, translate_target=True,
        parent_score=err_score,
    )
    contrib_report = _eval_edges(
        gold_doc, pred_doc, {"contributes_to"}, error_pairs, failure_pairs,
        translate_source=True, translate_target=True,
        parent_score=parent_for_contrib,
    )

    sub = {
        "event":          event_report["score"],
        "event_next":     event_next_report["score"],
        "failure":        failure_report["score"],
        "error":          error_report["score"],
        "attached_to":    attached_report["score"],
        "causes":         causes_report["score"],
        "contributes_to": contrib_report["score"],
    }
    
    _overall_dims = ("failure", "error", "attached_to", "causes", "contributes_to")
    overall = round(sum(sub[d] for d in _overall_dims) / len(_overall_dims), 3)

    return {
        "trace_id": gold_doc.get("trace_id") or pred_doc.get("trace_id") or "?",
        "event":          event_report,
        "event_next":     event_next_report,
        "failure":        failure_report,
        "error":          error_report,
        "attached_to":    attached_report,
        "causes":         causes_report,
        "contributes_to": contrib_report,
        "scores":         {**sub, "overall": overall},
        "overall":        overall,
    }
