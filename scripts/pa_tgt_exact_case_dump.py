#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tgt_exact subset에서 A/B 변화가 큰 케이스를 사람이 읽을 수 있게 덤프.

왜 필요?
- threshold 같은 수치 튜닝이 '어떤 문단'에서 '어떤 경계'를 바꿨는지 확인
- 개선/악화의 원인이 한두 문단(특정 패턴)에 집중되는지 빠르게 파악

정의
- subset 키: pred A의 번역문 리스트가 gold와 완전일치하는 (book_name, 문단식별자)
- boundary: integrity_report.py와 동일(_boundary_positions_normed)

예)
  python scripts/pa_tgt_exact_case_dump.py \
    --pred-a test_results/.../pa_output_n100_seed4.csv \
    --pred-b test_results/.../pa_output_n100_seed4_thr072.csv \
    --gold   test_results/.../pa_gold_subset_n100_seed4.csv \
    --diff-csv test_results/.../seed4_thr072_diff.csv \
    --top-gain 5 --top-loss 5 \
    --out-dir test_results/.../seed4_cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _group_pred(df: pd.DataFrame) -> Dict[KeyT, pd.DataFrame]:
    required = {"book_name", "문단식별자", "원문", "번역문"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"pred에 필수 컬럼이 없습니다: {missing}")
    out = df.copy()
    out["book_name"] = out["book_name"].fillna("").astype(str)
    out["문단식별자"] = out["문단식별자"].astype(int)
    return {k: g for k, g in out.groupby(["book_name", "문단식별자"], sort=False)}


def _group_gold(df: pd.DataFrame) -> Dict[KeyT, pd.DataFrame]:
    required = {"book_name", "문단식별자", "문장식별자", "원문", "번역문"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"gold에 필수 컬럼이 없습니다: {missing}")
    out = df.copy()
    out["book_name"] = out["book_name"].fillna("").astype(str)
    out["문단식별자"] = out["문단식별자"].astype(int)
    out["문장식별자"] = out["문장식별자"].astype(int)
    out = out.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable")
    return {k: g for k, g in out.groupby(["book_name", "문단식별자"], sort=False)}


def _tgt_exact(pred_g: pd.DataFrame, gold_g: pd.DataFrame) -> bool:
    pred_tgt = [str(x).strip() for x in pred_g["번역문"].fillna("").tolist()]
    gold_tgt = [str(x).strip() for x in gold_g["번역문"].fillna("").tolist()]
    return ([_norm(s) for s in pred_tgt] == [_norm(s) for s in gold_tgt])


def _counts(pred_g: pd.DataFrame, gold_g: pd.DataFrame) -> Tuple[int, int, int]:
    pred_src = [str(x).strip() for x in pred_g["원문"].fillna("").tolist()]
    gold_src = [str(x).strip() for x in gold_g["원문"].fillna("").tolist()]
    pred_b = _boundary_positions_normed(pred_src)
    gold_b = _boundary_positions_normed(gold_src)
    tp = len(pred_b & gold_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    _p, _r, f1 = _prf1(tp, fp, fn)
    return float(f1)


def _boundary_map(segments: List[str]) -> Tuple[List[int], Dict[int, int]]:
    """(positions list, pos->sent_idx). sent_idx는 1-based (경계는 sent_idx 뒤)."""
    cursor = 0
    positions: List[int] = []
    pos_to_i: Dict[int, int] = {}
    for i, seg in enumerate(segments, start=1):
        cursor += len(_norm(seg))
        if i < len(segments):
            positions.append(cursor)
            pos_to_i[cursor] = i
    return positions, pos_to_i


def _dump_case(
    *,
    out_path: Path,
    key: KeyT,
    pred_a_g: pd.DataFrame,
    pred_b_g: pd.DataFrame,
    gold_g: pd.DataFrame,
    label: str,
) -> None:
    book, pid = key
    a_src = [str(x) for x in pred_a_g["원문"].fillna("").tolist()]
    b_src = [str(x) for x in pred_b_g["원문"].fillna("").tolist()]
    g_src = [str(x) for x in gold_g["원문"].fillna("").tolist()]

    a_pos, a_map = _boundary_map(a_src)
    b_pos, b_map = _boundary_map(b_src)
    g_pos, g_map = _boundary_map(g_src)

    a_set = set(a_pos)
    b_set = set(b_pos)
    g_set = set(g_pos)
    union = sorted(a_set | b_set | g_set)

    tp_a, fp_a, fn_a = _counts(pred_a_g, gold_g)
    tp_b, fp_b, fn_b = _counts(pred_b_g, gold_g)
    f1_a = _f1(tp_a, fp_a, fn_a)
    f1_b = _f1(tp_b, fp_b, fn_b)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# {label}: {book}#{pid}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- key: ({book}, {pid})")
    lines.append(f"- F1 A/B: {f1_a:.4f} / {f1_b:.4f} (Δ={f1_b - f1_a:+.4f})")
    lines.append(f"- tp/fp/fn A: {tp_a}/{fp_a}/{fn_a}")
    lines.append(f"- tp/fp/fn B: {tp_b}/{fp_b}/{fn_b}")
    lines.append("")

    def _section(title: str, segs: List[str]) -> None:
        lines.append(f"## {title} ({len(segs)} sents)")
        cursor = 0
        for i, s in enumerate(segs, start=1):
            cursor += len(_norm(s))
            lines.append(f"{i:02d}. len_norm_cum={cursor} | {s}")
        lines.append("")

    _section("Gold src", g_src)
    _section("Pred A src", a_src)
    _section("Pred B src", b_src)

    lines.append("## Boundary positions (normed cumulative offsets)")
    lines.append("pos\tin_gold\tin_A\tin_B\tgold_after_sent\tA_after_sent\tB_after_sent")
    for pos in union:
        lines.append(
            "\t".join(
                [
                    str(pos),
                    "1" if pos in g_set else "0",
                    "1" if pos in a_set else "0",
                    "1" if pos in b_set else "0",
                    str(g_map.get(pos, "-")),
                    str(a_map.get(pos, "-")),
                    str(b_map.get(pos, "-")),
                ]
            )
        )
    lines.append("")

    def _pos_list(name: str, vals: Iterable[int]) -> None:
        vs = sorted(vals)
        lines.append(f"- {name}: {', '.join(str(x) for x in vs) if vs else '(none)'}")

    lines.append("## Sets")
    _pos_list("gold", g_set)
    _pos_list("A", a_set)
    _pos_list("B", b_set)
    _pos_list("A_only", a_set - g_set)
    _pos_list("B_only", b_set - g_set)
    _pos_list("gold_only", g_set - a_set)
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _select_from_diff_csv(path: Path, *, top_gain: int, top_loss: int) -> Tuple[List[KeyT], List[KeyT]]:
    df = pd.read_csv(path)
    required = {"book_name", "문단식별자", "delta_f1"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"diff-csv에 필수 컬럼이 없습니다: {missing}")
    df = df.copy()
    df["문단식별자"] = df["문단식별자"].astype(int)
    df["book_name"] = df["book_name"].fillna("").astype(str)

    gains = (
        df.sort_values(["delta_f1", "book_name", "문단식별자"], ascending=[False, True, True])
        .head(int(top_gain))
        .loc[:, ["book_name", "문단식별자"]]
    )
    losses = (
        df.sort_values(["delta_f1", "book_name", "문단식별자"], ascending=[True, True, True])
        .head(int(top_loss))
        .loc[:, ["book_name", "문단식별자"]]
    )
    return [(r[0], int(r[1])) for r in gains.itertuples(index=False)], [(r[0], int(r[1])) for r in losses.itertuples(index=False)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump top gain/loss cases with boundary details")
    ap.add_argument("--pred-a", required=True, type=str)
    ap.add_argument("--pred-b", required=True, type=str)
    ap.add_argument("--gold", required=True, type=str)
    ap.add_argument("--diff-csv", default=None, type=str)
    ap.add_argument("--top-gain", type=int, default=5)
    ap.add_argument("--top-loss", type=int, default=5)
    ap.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help=(
            "직접 덤프할 (book:pid) 목록. 예: --keys '당송팔대가문초구양수2:381' 'foo:12'. "
            "지정 시 diff-csv/top-gain/top-loss 설정과 무관하게 해당 키만 덤프합니다."
        ),
    )
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    pa = Path(args.pred_a)
    pb = Path(args.pred_b)
    pg = Path(args.gold)
    out_dir = Path(args.out_dir)

    a_df = _read(pa)
    b_df = _read(pb)
    g_df = _read(pg)
    a_map = _group_pred(a_df)
    b_map = _group_pred(b_df)
    g_map = _group_gold(g_df)

    common = sorted(set(a_map) & set(b_map) & set(g_map))
    if not common:
        raise SystemExit("공통 키가 없습니다. pred/gold의 book_name/문단식별자 정합을 확인하세요.")

    # 특정 키 직접 덤프
    if args.keys:
        out_dir.mkdir(parents=True, exist_ok=True)
        selected: List[KeyT] = []
        for raw in args.keys:
            s = str(raw).strip()
            if not s:
                continue
            if ":" not in s:
                raise SystemExit(f"--keys 형식 오류: {s} (book:pid)")
            book, pid_s = s.rsplit(":", 1)
            try:
                pid = int(pid_s)
            except Exception:
                raise SystemExit(f"--keys pid 파싱 실패: {s}")
            selected.append((book, pid))

        print("=")
        print("Case dump (explicit keys)")
        print("=")
        print(f"pred A: {pa}")
        print(f"pred B: {pb}")
        print(f"gold  : {pg}")
        print(f"out   : {out_dir}")

        dumped = 0
        for k in selected:
            if k not in a_map or k not in b_map or k not in g_map:
                print(f"- skip(not found in pred/gold): {k}")
                continue
            book, pid = k
            out_path = out_dir / f"key_{book}__{pid}.md"
            _dump_case(
                out_path=out_path,
                key=k,
                pred_a_g=a_map[k],
                pred_b_g=b_map[k],
                gold_g=g_map[k],
                label="key",
            )
            dumped += 1
            print(f"- wrote: {out_path}")

        if dumped == 0:
            raise SystemExit("덤프된 케이스가 없습니다. (--keys 값과 book_name/문단식별자 정합을 확인하세요)")
        return 0

    # 선택할 케이스 목록
    if args.diff_csv:
        gains, losses = _select_from_diff_csv(Path(args.diff_csv), top_gain=args.top_gain, top_loss=args.top_loss)
    else:
        # diff-csv가 없으면, common에서 tgt_exact(A)인 것만 ΔF1 계산 후 정렬
        rows: List[Tuple[KeyT, float]] = []
        for k in common:
            if not _tgt_exact(a_map[k], g_map[k]):
                continue
            tp_a, fp_a, fn_a = _counts(a_map[k], g_map[k])
            tp_b, fp_b, fn_b = _counts(b_map[k], g_map[k])
            rows.append((k, _f1(tp_b, fp_b, fn_b) - _f1(tp_a, fp_a, fn_a)))
        rows.sort(key=lambda x: (x[1], x[0][0], x[0][1]))
        losses = [k for k, _d in rows[: int(args.top_loss)]]
        gains = [k for k, _d in rows[::-1][: int(args.top_gain)]]

    # 덤프
    print("=")
    print("Case dump (A tgt_exact subset)")
    print("=")
    print(f"pred A: {pa}")
    print(f"pred B: {pb}")
    print(f"gold  : {pg}")
    print(f"out   : {out_dir}")

    dumped = 0
    for kind, ks in [("gain", gains), ("loss", losses)]:
        for k in ks:
            if k not in a_map or k not in b_map or k not in g_map:
                continue
            if not _tgt_exact(a_map[k], g_map[k]):
                continue
            book, pid = k
            out_path = out_dir / f"{kind}_{book}__{pid}.md"
            _dump_case(
                out_path=out_path,
                key=k,
                pred_a_g=a_map[k],
                pred_b_g=b_map[k],
                gold_g=g_map[k],
                label=kind,
            )
            dumped += 1
            print(f"- wrote: {out_path}")

    if dumped == 0:
        raise SystemExit("덤프된 케이스가 없습니다. (tgt_exact subset/키 정합을 확인하세요)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
