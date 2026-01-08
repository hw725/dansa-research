#!/usr/bin/env python3
"""Dump worst boundary-mismatch paragraphs for a given PA stage trace.

This script is meant for "확신형 오답" 디버깅:
- stage=src_matched_selected에서 per-paragraph boundary micro-F1을 계산
- F1이 낮은 문단(top K)을 뽑아
  - gold src segments
  - pred src_segments
  - 경계 mismatch(fp/fn)와 근접 이동(mean nearest distance)
  - top 후보(tag/score) 정보를 함께 덤프

Output is a single Markdown file for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _norm(s: str) -> str:
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


def _boundary_positions_normed(segments: List[str]) -> List[int]:
    """Return sorted boundary positions (normed cumulative lengths)."""
    positions: List[int] = []
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.append(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _read_tabular(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"GT 파일을 찾을 수 없습니다: {path}")
    if p.suffix.lower() == ".csv":
        try:
            return pd.read_csv(p, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="utf-8")
    return pd.read_excel(p)


def load_gt(gt_path: str) -> Dict[Tuple[str, int], List[str]]:
    """(book_name, pid) -> gold src segments (list[str])"""
    df = _read_tabular(gt_path)
    required = {"문단식별자", "원문"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"GT 파일에 필수 컬럼이 없습니다: {sorted(missing)}")
    if "book_name" not in df.columns:
        df["book_name"] = ""

    df = df.reset_index(drop=False).rename(columns={"index": "__row"})
    has_sid = "문장식별자" in df.columns
    if has_sid:
        df["문장식별자"] = pd.to_numeric(df["문장식별자"], errors="coerce")

    out: Dict[Tuple[str, int], List[str]] = {}
    for (book, pid), g in df.groupby(["book_name", "문단식별자"], sort=False):
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if has_sid:
            g = g.sort_values(["문장식별자", "__row"], kind="stable")
        else:
            g = g.sort_values(["__row"], kind="stable")
        out[(str(book or ""), pid_int)] = [str(x).strip() for x in g["원문"].fillna("").tolist()]
    return out


def load_stage_records(trace_jsonl: str, *, stage: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(trace_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("stage") != stage:
                continue
            pid = rec.get("paragraph_id")
            book = rec.get("book_name") or ""
            if pid is None:
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            out[(str(book), pid_int)] = rec
    return out


def _mean_nearest_distance(src: List[int], targets: List[int]) -> float | None:
    if not src or not targets:
        return None
    return float(sum(min(abs(x - y) for y in targets) for x in src) / len(src))


@dataclass
class Case:
    book_name: str
    paragraph_id: int
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    error_type: str
    margin: float | None
    best_tag: str | None
    best_score: float | None
    second_tag: str | None
    second_score: float | None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-xlsx", required=True)
    ap.add_argument("--trace-jsonl", required=True)
    ap.add_argument("--stage", default="src_matched_selected")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    gt = load_gt(args.gt_xlsx)
    recs = load_stage_records(args.trace_jsonl, stage=str(args.stage))

    cases: List[Case] = []
    for (book, pid), rec in recs.items():
        gold_src = gt.get((book, pid))
        if not gold_src:
            continue
        pred_src = rec.get("src_segments") or []
        if not isinstance(pred_src, list) or len(pred_src) == 0:
            continue

        pred_b = set(_boundary_positions_normed([str(x).strip() for x in pred_src]))
        gold_b = set(_boundary_positions_normed(gold_src))

        tp = len(pred_b & gold_b)
        fp = len(pred_b - gold_b)
        fn = len(gold_b - pred_b)
        p, r, f1 = _prf1(tp, fp, fn)

        if pred_b == gold_b:
            err = "exact"
        elif pred_b.issubset(gold_b):
            err = "under_split"
        elif gold_b.issubset(pred_b):
            err = "over_split"
        else:
            err = "mixed"

        meta = rec.get("meta") or {}
        margin = meta.get("best_margin_vs_second")
        margin_f = float(margin) if isinstance(margin, (int, float)) else None
        best_tag = meta.get("best_tag") if isinstance(meta.get("best_tag"), str) else None
        best_score = meta.get("best_score") if isinstance(meta.get("best_score"), (int, float)) else None

        top = meta.get("top_candidates") or []
        second_tag = None
        second_score = None
        if isinstance(top, list) and len(top) >= 2 and isinstance(top[1], dict):
            second_tag = top[1].get("tag") if isinstance(top[1].get("tag"), str) else None
            second_score = float(top[1].get("score")) if isinstance(top[1].get("score"), (int, float)) else None

        cases.append(
            Case(
                book_name=str(book),
                paragraph_id=int(pid),
                f1=float(f1),
                precision=float(p),
                recall=float(r),
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                error_type=err,
                margin=margin_f,
                best_tag=best_tag,
                best_score=float(best_score) if isinstance(best_score, (int, float)) else None,
                second_tag=second_tag,
                second_score=second_score,
            )
        )

    cases.sort(key=lambda c: (c.f1, -(c.fp + c.fn), c.book_name, c.paragraph_id))
    worst = cases[: int(args.top_k)]

    out_md = Path(args.out_md) if args.out_md else Path("test_results") / f"worst_{args.stage}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # Summary stats
    best_tag_wrong = Counter([c.best_tag for c in cases if c.f1 < 1.0 and c.best_tag])

    md: List[str] = []
    md.append(f"# Worst cases dump: stage={args.stage}\n")
    md.append(f"- trace: `{args.trace_jsonl}`\n")
    md.append(f"- gold: `{args.gt_xlsx}`\n")
    md.append(f"- total_cases: {len(cases)}\n")
    md.append(f"- top_k: {len(worst)}\n")
    md.append("\n## best_tag among wrong (top 15)\n")
    for tag, cnt in best_tag_wrong.most_common(15):
        md.append(f"- {tag}: {cnt}\n")

    md.append("\n---\n")

    for idx, c in enumerate(worst, start=1):
        key = (c.book_name, c.paragraph_id)
        rec = recs.get(key)
        gold_src = gt.get(key) or []
        pred_src = rec.get("src_segments") if rec else []
        pred_src = [str(x) for x in pred_src] if isinstance(pred_src, list) else []

        pred_pos = _boundary_positions_normed(pred_src)
        gold_pos = _boundary_positions_normed(gold_src)
        pred_set = set(pred_pos)
        gold_set = set(gold_pos)
        fp_pos = sorted(list(pred_set - gold_set))
        fn_pos = sorted(list(gold_set - pred_set))

        mean_shift_fn = _mean_nearest_distance(fn_pos, pred_pos)
        mean_shift_fp = _mean_nearest_distance(fp_pos, gold_pos)

        md.append(f"## {idx}. {c.book_name} / pid={c.paragraph_id}\n")
        md.append(
            f"- f1={c.f1:.4f} (p={c.precision:.4f}, r={c.recall:.4f}) tp={c.tp} fp={c.fp} fn={c.fn} type={c.error_type}\n"
        )
        md.append(f"- margin={c.margin} best={c.best_tag}:{c.best_score} second={c.second_tag}:{c.second_score}\n")
        md.append(f"- n_pred={len(pred_src)} n_gold={len(gold_src)}\n")
        md.append(f"- mean_shift(fn→pred)={mean_shift_fn} mean_shift(fp→gold)={mean_shift_fp}\n")
        md.append(f"- fp_positions(normed): {fp_pos}\n")
        md.append(f"- fn_positions(normed): {fn_pos}\n")

        md.append("\n### pred src_segments\n")
        for i, s in enumerate(pred_src, start=1):
            md.append(f"{i:02d}. {s}\n")

        md.append("\n### gold src (GT)\n")
        for i, s in enumerate(gold_src, start=1):
            md.append(f"{i:02d}. {s}\n")

        # Top candidates detail (if present)
        meta = (rec or {}).get("meta") or {}
        top = meta.get("top_candidates") or []
        if isinstance(top, list) and top:
            md.append("\n### top_candidates (trace)\n")
            for cand in top:
                if not isinstance(cand, dict):
                    continue
                md.append(
                    "- "
                    + ", ".join(
                        [
                            f"tag={cand.get('tag')}",
                            f"score={cand.get('score')}",
                            f"considered={cand.get('considered')}",
                            f"prior_bonus={cand.get('prior_bonus')}",
                            f"short_pairs={cand.get('short_pairs')}",
                            f"empty_src_pairs={cand.get('empty_src_pairs')}",
                            f"skip_reason={cand.get('skip_reason')}",
                        ]
                    )
                    + "\n"
                )

        md.append("\n---\n")

    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
