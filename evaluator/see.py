from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .scorer import DIMENSIONS, score_one


def _resolve_pred(pred_root: Path, stem: str) -> Optional[Path]:
    p1 = pred_root / stem / "diagnosis.json"
    if p1.is_file():
        return p1
    p2 = pred_root / f"{stem}.json"
    if p2.is_file():
        return p2
    return None


def _fmt_eta(s: float) -> str:
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s // 60:.0f}m{s % 60:.0f}s"
    return f"{s // 3600:.0f}h{(s % 3600) // 60:.0f}m"


def cmd_single(args: argparse.Namespace) -> int:
    gold = json.loads(Path(args.gold).read_text())
    pred = json.loads(Path(args.pred).read_text())
    t0 = time.time()
    report = score_one(gold, pred)
    elapsed = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"overall={report['overall']:.3f}  ({elapsed*1000:.0f}ms)  → {out}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    gold_root = Path(args.gold_dir)
    pred_root = Path(args.pred_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gold_files = sorted(gold_root.glob("*.json"))
    pairs: List[Tuple[str, Path, Path]] = []
    skipped: List[str] = []
    for gf in gold_files:
        pf = _resolve_pred(pred_root, gf.stem)
        if pf is None:
            skipped.append(gf.stem); continue
        pairs.append((gf.stem, gf, pf))
    total = len(pairs)
    print(f"evaluaterule batch: {total} pairs  (rules-only)  → {out_root}")

    reports: List[Dict[str, Any]] = []
    run_t0 = time.time()
    for idx, (stem, gf, pf) in enumerate(pairs, start=1):
        try:
            gold = json.loads(gf.read_text())
            pred = json.loads(pf.read_text())
        except Exception as e:
            print(f"[skip] {stem}: parse error {e}", file=sys.stderr)
            continue
        report = score_one(gold, pred)
        out = out_root / f"{stem}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        reports.append(report)
        cum = time.time() - run_t0
        eta = (cum / idx) * (total - idx)
        print(f"  [{idx:>3}/{total}] {stem}: overall={report['overall']:.3f}  ETA {_fmt_eta(eta)}")

    summary = _aggregate(reports)
    summary["skipped"] = skipped
    summary["wall_elapsed_s"] = round(time.time() - run_t0, 2)
    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(
        f"\n{len(reports)} scored, {len(skipped)} skipped. "
        f"overall_macro={summary['overall_macro']}  "
        f"({_fmt_eta(summary['wall_elapsed_s'])})  → {out_root}/_summary.json"
    )
    return 0


def _aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not reports:
        return {
            "overall_macro": 0.0,
            "per_dim_macro": {d: 0.0 for d in DIMENSIONS},
            "by_complexity": {},
        }
    n = len(reports)
    macro = {d: round(sum(r["scores"][d] for r in reports) / n, 3) for d in DIMENSIONS}
    overall = round(sum(r["overall"] for r in reports) / n, 3)

    sized = sorted(reports, key=lambda r: r["event"]["gold_count"])
    third = max(1, len(sized) // 3)
    buckets = {
        "small": sized[:third],
        "medium": sized[third:2 * third],
        "large": sized[2 * third:],
    }
    by_complexity: Dict[str, Any] = {}
    for name, group in buckets.items():
        if not group:
            continue
        gn = len(group)
        gold_counts = [r["event"]["gold_count"] for r in group]
        by_complexity[name] = {
            "n": gn,
            "gold_event_min": min(gold_counts),
            "gold_event_max": max(gold_counts),
            "gold_event_mean": round(sum(gold_counts) / gn, 2),
            "overall_macro": round(sum(r["overall"] for r in group) / gn, 3),
            "per_dim_macro": {
                d: round(sum(r["scores"][d] for r in group) / gn, 3)
                for d in DIMENSIONS
            },
        }

    return {
        "n_scored": n,
        "per_dim_macro": macro,
        "overall_macro": overall,
        "by_complexity": by_complexity,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser("evaluaterule", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("single", help="score one (gold, pred) pair")
    sp.add_argument("--gold", required=True)
    sp.add_argument("--pred", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_single)

    sp = sub.add_parser("batch", help="score every (gold, pred) pair under two dirs")
    sp.add_argument("--gold-dir", required=True)
    sp.add_argument("--pred-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_batch)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
