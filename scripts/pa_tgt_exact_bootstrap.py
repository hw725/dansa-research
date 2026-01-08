#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tgt 완전일치 subset 기준 micro-F1 부트스트랩.

왜 필요?
- 표본이 작을 때(예: test_100) 절대 점수의 분산이 커질 수 있어,
  설정 변경(A/B)의 개선이 '진짜'인지 확인하려면 부트스트랩 CI가 유용하다.
- 이 스크립트는 integrity_report.py와 동일한 정의로
  (micro, tgt 완전일치 subset) P/R/F1을 계산한다.

사용 예)
  docker compose run --rm csp python scripts/pa_tgt_exact_bootstrap.py \
    --pred test_results/repro_det_thr070_len200_seed1_ml20.csv \
    --gold datasets/pa/test_100_from_pd.csv \
    --n 5000 --seed 1

  # 두 설정 비교(ΔF1)까지
  docker compose run --rm csp python scripts/pa_tgt_exact_bootstrap.py \
    --pred A.csv --pred-b B.csv --gold datasets/pa/test_100_from_pd.csv \
    --n 5000 --seed 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# scripts/ 아래에서 실행되면 sys.path[0]이 scripts로 잡혀 루트 모듈 import가 깨질 수 있어 보정
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from integrity_report import _boundary_positions_normed, _norm, _prf1


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


KeyT = Tuple[str, int]  # (book_name, pid)


def _group_keys(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[List[KeyT], Dict[KeyT, pd.DataFrame], Dict[KeyT, pd.DataFrame]]:
    if "book_name" not in pred_df.columns:
        raise SystemExit("pred에 book_name이 없습니다. tgt_exact subset 평가의 키 정합을 위해 book_name 포함 출력이 필요합니다.")

    required_pred = {"book_name", "문단식별자", "원문", "번역문"}
    required_gold = {"book_name", "문단식별자", "문장식별자", "원문", "번역문"}
    if not required_pred.issubset(set(pred_df.columns)):
        raise SystemExit(f"pred에 필수 컬럼이 없습니다: {sorted(required_pred - set(pred_df.columns))}")
    if not required_gold.issubset(set(gold_df.columns)):
        raise SystemExit(f"gold에 필수 컬럼이 없습니다: {sorted(required_gold - set(gold_df.columns))}")

    pred = pred_df.copy()
    gold = gold_df.copy()
    pred["book_name"] = pred["book_name"].fillna("").astype(str)
    gold["book_name"] = gold["book_name"].fillna("").astype(str)
    pred["문단식별자"] = pred["문단식별자"].astype(int)
    gold["문단식별자"] = gold["문단식별자"].astype(int)

    pred_groups = pred.groupby(["book_name", "문단식별자"], sort=False)
    gold_groups = gold.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable").groupby(
        ["book_name", "문단식별자"], sort=False
    )

    pred_keys = set(pred_groups.groups.keys())
    gold_keys = set(gold_groups.groups.keys())
    keys = sorted(pred_keys & gold_keys)

    pred_map = {k: pred_groups.get_group(k) for k in keys}
    gold_map = {k: gold_groups.get_group(k) for k in keys}
    return keys, pred_map, gold_map


def _per_key_counts(pred_g: pd.DataFrame, gold_g: pd.DataFrame) -> Tuple[bool, int, int, int]:
    pred_tgt = [str(x).strip() for x in pred_g["번역문"].fillna("").tolist()]
    gold_tgt = [str(x).strip() for x in gold_g["번역문"].fillna("").tolist()]
    pred_tgt_norm = [_norm(s) for s in pred_tgt]
    gold_tgt_norm = [_norm(s) for s in gold_tgt]
    tgt_match = (pred_tgt_norm == gold_tgt_norm)

    pred_src = [str(x).strip() for x in pred_g["원문"].fillna("").tolist()]
    gold_src = [str(x).strip() for x in gold_g["원문"].fillna("").tolist()]
    pred_b = _boundary_positions_normed(pred_src)
    gold_b = _boundary_positions_normed(gold_src)
    inter = pred_b & gold_b

    tp = len(inter)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)
    return tgt_match, tp, fp, fn


def _point_estimate(rows: List[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    tp = sum(x for x, _, _ in rows)
    fp = sum(x for _, x, _ in rows)
    fn = sum(x for _, _, x in rows)
    return _prf1(tp, fp, fn)


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def bootstrap_f1(rows: List[Tuple[int, int, int]], *, n: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    m = len(rows)
    if m == 0:
        return []
    out: List[float] = []
    for _ in range(n):
        tp = fp = fn = 0
        for _j in range(m):
            a, b, c = rows[rng.randrange(m)]
            tp += a
            fp += b
            fn += c
        _, _, f1 = _prf1(tp, fp, fn)
        out.append(float(f1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap micro-F1 on tgt-exact subset")
    ap.add_argument("--pred", required=True, type=str, help="PA output CSV/XLSX")
    ap.add_argument("--pred-b", default=None, type=str, help="비교 대상 PA output(옵션)")
    ap.add_argument("--gold", required=True, type=str, help="gold sentences CSV/XLSX")
    ap.add_argument("--n", type=int, default=5000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold)

    pred_df = _read(pred_path)
    gold_df = _read(gold_path)
    keys, pred_map, gold_map = _group_keys(pred_df, gold_df)

    subset_rows: List[Tuple[int, int, int]] = []
    subset_keys: List[KeyT] = []
    for k in keys:
        tgt_match, tp, fp, fn = _per_key_counts(pred_map[k], gold_map[k])
        if not tgt_match:
            continue
        subset_keys.append(k)
        subset_rows.append((tp, fp, fn))

    p0, r0, f0 = _point_estimate(subset_rows)
    boots = bootstrap_f1(subset_rows, n=int(args.n), seed=int(args.seed))
    boots_sorted = sorted(boots)
    lo = _percentile(boots_sorted, 0.025)
    mid = _percentile(boots_sorted, 0.5)
    hi = _percentile(boots_sorted, 0.975)

    print("=")
    print("Bootstrap: micro-F1 (tgt 완전일치 subset)")
    print("=")
    print(f"pred: {pred_path}")
    print(f"gold: {gold_path}")
    print(f"tgt_exact keys: {len(subset_keys)}/{len(keys)}")
    print(f"point micro P/R/F1: {p0:.4f} / {r0:.4f} / {f0:.4f}")
    print(f"bootstrap n={len(boots_sorted)} seed={args.seed}")
    print(f"F1 CI(2.5/50/97.5): {lo:.4f} / {mid:.4f} / {hi:.4f}")

    if args.pred_b:
        pred_b_path = Path(args.pred_b)
        pred_b_df = _read(pred_b_path)
        _keys2, pred_b_map, gold_map2 = _group_keys(pred_b_df, gold_df)
        if _keys2 != keys:
            # 키가 달라도 공통키만으로 재구성
            common = sorted(set(keys) & set(_keys2))
            keys_use = common
        else:
            keys_use = keys

        rows_a: List[Tuple[int, int, int]] = []
        rows_b: List[Tuple[int, int, int]] = []
        for k in keys_use:
            tgt_a, tp_a, fp_a, fn_a = _per_key_counts(pred_map[k], gold_map2[k])
            tgt_b, tp_b, fp_b, fn_b = _per_key_counts(pred_b_map[k], gold_map2[k])
            # 비교는 tgt_exact subset 기준(둘 다 tgt가 완전일치일 필요는 없음: A 기준/ B 기준?)
            # 여기서는 정의를 명확히: A의 tgt_exact subset에서 src boundary 개선을 보려면 tgt_a로 필터.
            if not tgt_a:
                continue
            rows_a.append((tp_a, fp_a, fn_a))
            rows_b.append((tp_b, fp_b, fn_b))

        _, _, f_a = _point_estimate(rows_a)
        _, _, f_b = _point_estimate(rows_b)
        delta_point = f_b - f_a

        rng = random.Random(int(args.seed))
        m = len(rows_a)
        deltas: List[float] = []
        for _ in range(int(args.n)):
            tp1 = fp1 = fn1 = 0
            tp2 = fp2 = fn2 = 0
            for _j in range(m):
                i = rng.randrange(m)
                a1, b1, c1 = rows_a[i]
                a2, b2, c2 = rows_b[i]
                tp1 += a1
                fp1 += b1
                fn1 += c1
                tp2 += a2
                fp2 += b2
                fn2 += c2
            _, _, f1 = _prf1(tp1, fp1, fn1)
            _, _, f2 = _prf1(tp2, fp2, fn2)
            deltas.append(float(f2 - f1))

        deltas_sorted = sorted(deltas)
        dlo = _percentile(deltas_sorted, 0.025)
        dmid = _percentile(deltas_sorted, 0.5)
        dhi = _percentile(deltas_sorted, 0.975)

        print("\n=")
        print("Paired bootstrap ΔF1 (B - A) on A's tgt_exact subset")
        print("=")
        print(f"pred A: {pred_path}")
        print(f"pred B: {pred_b_path}")
        print(f"keys used (A tgt_exact): {m}")
        print(f"point F1 A/B: {f_a:.4f} / {f_b:.4f} (Δ={delta_point:+.4f})")
        print(f"ΔF1 CI(2.5/50/97.5): {dlo:+.4f} / {dmid:+.4f} / {dhi:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
