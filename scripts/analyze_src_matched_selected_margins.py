#!/usr/bin/env python3
"""Analyze low-margin cases in PA stage trace (default: src_matched_selected).

Goal
- Identify paragraphs where the winner vs runner-up score margin is small.
- For those paragraphs, quantify boundary mismatch patterns vs gold:
  - under-split / over-split / mixed
  - tp/fp/fn counts and per-paragraph F1

Inputs
- --gt-xlsx: gold in .xlsx or .csv (supports datasets/p2s/test_100_from_pd.csv)
- --trace-jsonl: pa stage trace JSONL produced by p2s/processor.py

Outputs
- CSV (one row per paragraph)
- Console summary for low-margin bucket
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _norm(s: str) -> str:
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


def _boundary_positions_normed(segments: List[str]) -> set[int]:
    positions: set[int] = set()
    cursor = 0
    for i, seg in enumerate(segments):
        seg_norm = _norm(seg)
        cursor += len(seg_norm)
        if i < len(segments) - 1:
            positions.add(cursor)
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


def load_gt(gt_path: str) -> Dict[Tuple[str, int], Dict[str, List[str]]]:
    df = _read_tabular(gt_path)

    required = {"문단식별자", "원문", "번역문"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"GT 파일에 필수 컬럼이 없습니다: {sorted(missing)}")

    if "book_name" not in df.columns:
        df["book_name"] = ""

    df = df.reset_index(drop=False).rename(columns={"index": "__row"})
    has_sid = "문장식별자" in df.columns
    if has_sid:
        df["문장식별자"] = pd.to_numeric(df["문장식별자"], errors="coerce")

    grouped: Dict[Tuple[str, int], Dict[str, List[str]]] = {}
    for (book, pid), g in df.groupby(["book_name", "문단식별자"], sort=False):
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if has_sid:
            g = g.sort_values(["문장식별자", "__row"], kind="stable")
        else:
            g = g.sort_values(["__row"], kind="stable")
        grouped[(str(book or ""), pid_int)] = {
            "src": [str(x).strip() for x in g["원문"].fillna("").tolist()],
            "tgt": [str(x).strip() for x in g["번역문"].fillna("").tolist()],
        }
    return grouped


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


def _quantile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * q))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return float(values_sorted[idx])


def _mean_nearest_distance(src: Iterable[int], targets: Iterable[int]) -> float | None:
    src_list = list(src)
    tgt_list = list(targets)
    if not src_list or not tgt_list:
        return None
    dists: List[int] = []
    for x in src_list:
        dists.append(min(abs(x - y) for y in tgt_list))
    return float(sum(dists) / len(dists)) if dists else None


@dataclass(frozen=True)
class Row:
    book_name: str
    paragraph_id: int
    margin: float | None
    best_tag: str | None
    best_score: float | None
    second_tag: str | None
    second_score: float | None
    n_pred: int
    n_gold: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    error_type: str
    mean_shift_fn_to_pred: float | None
    mean_shift_fp_to_gold: float | None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-xlsx", required=True)
    ap.add_argument("--trace-jsonl", required=True)
    ap.add_argument("--stage", default="src_matched_selected")
    ap.add_argument("--margin-quantile", type=float, default=0.10)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    gt = load_gt(args.gt_xlsx)
    stage_recs = load_stage_records(args.trace_jsonl, stage=str(args.stage))

    rows: List[Row] = []
    margins: List[float] = []

    for (book, pid), rec in stage_recs.items():
        pack = gt.get((book, pid))
        if not pack:
            continue
        pred_src = rec.get("src_segments") or []
        if not isinstance(pred_src, list) or len(pred_src) == 0:
            continue

        gold_src = pack["src"]
        pred_b = _boundary_positions_normed([str(x).strip() for x in pred_src])
        gold_b = _boundary_positions_normed(gold_src)

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
        if isinstance(margin, (int, float)):
            margin_f = float(margin)
            margins.append(margin_f)
        else:
            margin_f = None

        top = meta.get("top_candidates") or []
        best_tag = meta.get("best_tag") if isinstance(meta.get("best_tag"), str) else None
        best_score = meta.get("best_score") if isinstance(meta.get("best_score"), (int, float)) else None

        second_tag = None
        second_score = None
        if isinstance(top, list) and len(top) >= 2:
            c2 = top[1]
            if isinstance(c2, dict):
                if isinstance(c2.get("tag"), str):
                    second_tag = c2.get("tag")
                if isinstance(c2.get("score"), (int, float)):
                    second_score = float(c2.get("score"))

        mean_shift_fn_to_pred = _mean_nearest_distance(gold_b - pred_b, pred_b)
        mean_shift_fp_to_gold = _mean_nearest_distance(pred_b - gold_b, gold_b)

        rows.append(
            Row(
                book_name=str(book),
                paragraph_id=int(pid),
                margin=margin_f,
                best_tag=best_tag,
                best_score=float(best_score) if isinstance(best_score, (int, float)) else None,
                second_tag=second_tag,
                second_score=second_score,
                n_pred=len(pred_src),
                n_gold=len(gold_src),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=p,
                recall=r,
                f1=f1,
                error_type=err,
                mean_shift_fn_to_pred=mean_shift_fn_to_pred,
                mean_shift_fp_to_gold=mean_shift_fp_to_gold,
            )
        )

    margin_cut = _quantile(margins, float(args.margin_quantile))

    out_path = Path(args.out_csv) if args.out_csv else Path("test_results") / f"low_margin_{args.stage}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["is_low_margin"] = False
    if margin_cut is not None:
        df["is_low_margin"] = df["margin"].apply(lambda x: isinstance(x, (int, float)) and float(x) <= float(margin_cut))

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"stage={args.stage}")
    print(f"rows={len(df)} out={out_path}")
    print(f"margin_cut(q={args.margin_quantile})={margin_cut}")

    def _summ(sub: pd.DataFrame, label: str) -> None:
        if len(sub) == 0:
            print(f"[{label}] empty")
            return
        print(f"[{label}] n={len(sub)} f1_mean={sub['f1'].mean():.4f} f1_p50={sub['f1'].median():.4f}")
        ct = Counter(sub["error_type"].fillna("unknown").tolist())
        print(f"[{label}] error_type: {dict(ct)}")
        wrong = sub[sub["f1"] < 1.0]
        if len(wrong) > 0:
            tag_ct = Counter(wrong["best_tag"].fillna("(none)").tolist())
            print(f"[{label}] best_tag among wrong top10: {tag_ct.most_common(10)}")

    _summ(df, "all")
    _summ(df[df["is_low_margin"]], "low_margin")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
