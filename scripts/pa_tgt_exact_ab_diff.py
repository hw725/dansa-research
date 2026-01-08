#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tgt_exact subset에서 A/B 문단별 변화(ΔF1) 분석.

목적
- 숫자(예: threshold) 튜닝이 실제로 무엇을 바꾸는지 '문단 단위'로 확인
- 어떤 문단에서 개선/악화가 나는지 추출해, 다음 단계(로직/모델 개선)의 근거로 사용

정의
- subset 키: pred A의 번역문 리스트가 gold와 완전일치하는 (book_name, 문단식별자)
- 각 키에 대해 src boundary micro 집계(tp/fp/fn → P/R/F1)

예)
  python scripts/pa_tgt_exact_ab_diff.py \
    --pred-a test_results/.../pa_output_n100_seed4.csv \
    --pred-b test_results/.../pa_output_n100_seed4_thr072.csv \
    --gold   test_results/.../pa_gold_subset_n100_seed4.csv \
    --top 20 \
    --out-csv test_results/.../seed4_thr072_diff.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from integrity_report import _boundary_positions_normed, _norm, _prf1


KeyT = Tuple[str, int]  # (book_name, pid)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _group_keys(df: pd.DataFrame) -> Dict[KeyT, pd.DataFrame]:
    required = {"book_name", "문단식별자", "원문", "번역문"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"필수 컬럼이 없습니다: {missing}")

    out = df.copy()
    out["book_name"] = out["book_name"].fillna("").astype(str)
    out["문단식별자"] = out["문단식별자"].astype(int)
    return {k: g for k, g in out.groupby(["book_name", "문단식별자"], sort=False)}


def _tgt_exact(pred_g: pd.DataFrame, gold_g: pd.DataFrame) -> bool:
    pred_tgt = [str(x).strip() for x in pred_g["번역문"].fillna("").tolist()]
    gold_tgt = [str(x).strip() for x in gold_g["번역문"].fillna("").tolist()]
    return ([_norm(s) for s in pred_tgt] == [_norm(s) for s in gold_tgt])


def _counts(pred_g: pd.DataFrame, gold_g: pd.DataFrame) -> Tuple[int, int, int, int, int]:
    """(tp, fp, fn, n_src_sent, n_tgt_sent)."""
    pred_src = [str(x).strip() for x in pred_g["원문"].fillna("").tolist()]
    gold_src = [str(x).strip() for x in gold_g["원문"].fillna("").tolist()]
    pred_b = _boundary_positions_normed(pred_src)
    gold_b = _boundary_positions_normed(gold_src)
    tp = len(pred_b & gold_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)
    return tp, fp, fn, len(pred_src), len(pred_g["번역문"].fillna("").tolist())


def _f1(tp: int, fp: int, fn: int) -> float:
    _p, _r, f1 = _prf1(tp, fp, fn)
    return float(f1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-key ΔF1 analysis on tgt_exact subset")
    ap.add_argument("--pred-a", required=True, type=str)
    ap.add_argument("--pred-b", required=True, type=str)
    ap.add_argument("--gold", required=True, type=str)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    pa = Path(args.pred_a)
    pb = Path(args.pred_b)
    pg = Path(args.gold)

    a_df = _read(pa)
    b_df = _read(pb)
    g_df = _read(pg)

    a_map = _group_keys(a_df)
    b_map = _group_keys(b_df)
    g_map = _group_keys(g_df)

    common_keys = sorted(set(a_map) & set(b_map) & set(g_map))
    if not common_keys:
        raise SystemExit("공통 키가 없습니다. pred/gold의 book_name/문단식별자 정합을 확인하세요.")

    rows: List[dict] = []
    for k in common_keys:
        a_g = a_map[k]
        g_g = g_map[k]
        if not _tgt_exact(a_g, g_g):
            continue

        b_g = b_map[k]
        tp_a, fp_a, fn_a, n_src_a, n_tgt_a = _counts(a_g, g_g)
        tp_b, fp_b, fn_b, n_src_b, n_tgt_b = _counts(b_g, g_g)

        f1_a = _f1(tp_a, fp_a, fn_a)
        f1_b = _f1(tp_b, fp_b, fn_b)

        rows.append(
            {
                "book_name": k[0],
                "문단식별자": k[1],
                "tgt_exact_A": True,
                "n_src_sent_A": n_src_a,
                "n_src_sent_B": n_src_b,
                "n_tgt_sent_A": n_tgt_a,
                "n_tgt_sent_B": n_tgt_b,
                "tp_A": tp_a,
                "fp_A": fp_a,
                "fn_A": fn_a,
                "tp_B": tp_b,
                "fp_B": fp_b,
                "fn_B": fn_b,
                "f1_A": f1_a,
                "f1_B": f1_b,
                "delta_f1": f1_b - f1_a,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("tgt_exact(A) subset이 비어 있습니다. 입력 파일을 확인하세요.")

    overall_a = _f1(int(out_df["tp_A"].sum()), int(out_df["fp_A"].sum()), int(out_df["fn_A"].sum()))
    overall_b = _f1(int(out_df["tp_B"].sum()), int(out_df["fp_B"].sum()), int(out_df["fn_B"].sum()))
    print("=")
    print("Per-key ΔF1 on A tgt_exact subset")
    print("=")
    print(f"pred A: {pa}")
    print(f"pred B: {pb}")
    print(f"gold  : {pg}")
    print(f"keys used (A tgt_exact): {len(out_df)} / {len(common_keys)}")
    print(f"overall F1 A/B: {overall_a:.4f} / {overall_b:.4f} (Δ={overall_b - overall_a:+.4f})")

    top = int(args.top)
    gains = out_df.sort_values(["delta_f1", "book_name", "문단식별자"], ascending=[False, True, True]).head(top)
    losses = out_df.sort_values(["delta_f1", "book_name", "문단식별자"], ascending=[True, True, True]).head(top)

    print("\n-- Top gains --")
    for r in gains.itertuples(index=False):
        print(
            f"{r.book_name}#{r.문단식별자}: Δ={r.delta_f1:+.4f}  A={r.f1_A:.4f} B={r.f1_B:.4f}  "
            f"(tp/fp/fn A={r.tp_A}/{r.fp_A}/{r.fn_A} → B={r.tp_B}/{r.fp_B}/{r.fn_B})"
        )

    print("\n-- Top losses --")
    for r in losses.itertuples(index=False):
        print(
            f"{r.book_name}#{r.문단식별자}: Δ={r.delta_f1:+.4f}  A={r.f1_A:.4f} B={r.f1_B:.4f}  "
            f"(tp/fp/fn A={r.tp_A}/{r.fp_A}/{r.fn_A} → B={r.tp_B}/{r.fp_B}/{r.fn_B})"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
