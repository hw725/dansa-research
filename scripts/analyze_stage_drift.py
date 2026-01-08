"""단계별 드리프트(원문 경계) 최초 발생 지점 추적.

입력:
- GT: 문단식별자 단위로 정답 alignments가 나열된 엑셀(예: ...grouped_by_paragraphid.xlsx)
    컬럼: 문단식별자, 원문, 번역문, (선택) book_name
- Trace(JSONL): pa/processor.py에서 기록한 단계별 src/tgt segments

출력:
- stage별 원문 경계 Precision/Recall/F1 (micro)
    정의는 integrity_report.py의 _boundary_positions_normed/_prf1과 동일
    (문장 경계 위치 집합 비교: {len(A), len(A)+len(B), ...})

비고:
- tgt_split 처럼 src_segments가 비어있는 stage는 boundary 평가에서 제외합니다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_STAGE_ORDER: List[str] = [
    "tgt_split",
    "src_matched_selected",
    "src_adjacent_refined",
    "src_safe_source_split",
    "alignment_built",
    "after_model_alignment",
    "after_atomic_brackets",
    "after_particle_enhance",
    "after_restore_integrity",
    "after_final_cleanup",
    "after_quote_merge",
    "final",
]


def _norm(s: str) -> str:
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


def _boundary_positions_normed(segments: List[str]) -> set[int]:
    """정규화 문자열 기준 문장 경계 위치(누적 길이) 집합.

    예: [A,B,C]면 {len(A), len(A)+len(B)}.
    """

    positions: set[int] = set()
    cursor = 0
    for i, seg in enumerate(segments):
        seg_norm = _norm(seg)
        cursor += len(seg_norm)
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # integrity_report.py와 동일한 정의
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


@dataclass
class StageAgg:
    paragraphs: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        p, _r, _f1 = _prf1(self.tp, self.fp, self.fn)
        return p

    def recall(self) -> float:
        _p, r, _f1 = _prf1(self.tp, self.fp, self.fn)
        return r

    def f1(self) -> float:
        _p, _r, f1 = _prf1(self.tp, self.fp, self.fn)
        return f1


def load_trace(trace_jsonl: str) -> Dict[Tuple[str, int, str], Dict]:
    """(book_name, paragraph_id, stage) -> record (last wins)"""
    out: Dict[Tuple[str, int, str], Dict] = {}
    with open(trace_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("paragraph_id")
            stage = rec.get("stage")
            book = rec.get("book_name")
            if pid is None or stage is None:
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            out[(str(book or ""), pid_int, str(stage))] = rec
    return out


def _read_tabular(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"GT 파일을 찾을 수 없습니다: {path}")
    if p.suffix.lower() == ".csv":
        # repo의 csv는 대개 utf-8-sig
        try:
            return pd.read_csv(p, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="utf-8")
    return pd.read_excel(p)


def load_gt(gt_path: str) -> Dict[Tuple[str, int], Dict[str, List[str]]]:
    """Gold를 (book_name, 문단식별자) -> {src: [...], tgt: [...]}로 로드한다.

    지원 포맷:
    - grouped_by_paragraphid.xlsx: (문단식별자, book_name, 원문, 번역문) 행이 문장 단위로 이미 나열
    - datasets/pa/test_100_from_pd.csv: (문단식별자, 문장식별자, book_name, 원문, 번역문)

    핵심은 '문단 내 문장 순서'를 안정적으로 복원하는 것.
    """

    df = _read_tabular(gt_path)

    required = {"문단식별자", "원문", "번역문"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"GT 엑셀에 필수 컬럼이 없습니다: {sorted(missing)}")

    if "book_name" not in df.columns:
        df["book_name"] = ""

    # 그룹 내 순서를 안정적으로 유지하기 위한 정렬키
    # - 문장식별자가 있으면 그걸 우선
    # - 없으면 원본 파일 행순서(__row)
    df = df.reset_index(drop=False).rename(columns={"index": "__row"})
    has_sid = "문장식별자" in df.columns
    if has_sid:
        # numeric sort가 되도록 강제
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


def _tgt_exact_match(pred_tgt: List[str], gold_tgt: List[str]) -> bool:
    return [_norm(s) for s in pred_tgt] == [_norm(s) for s in gold_tgt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt-xlsx",
        required=True,
        help=(
            "GT 경로(.xlsx 또는 .csv). "
            "grouped_by_paragraphid.xlsx 또는 datasets/pa/test_100_from_pd.csv 같은 문장단위 gold를 지원합니다."
        ),
    )
    ap.add_argument("--trace-jsonl", required=True, help="PA 단계 트레이스 JSONL 경로")
    ap.add_argument("--out-csv", default=None, help="요약 CSV 출력 경로(옵션)")
    ap.add_argument(
        "--stages",
        default=None,
        help="콤마로 stage를 제한합니다(옵션). 기본은 trace에 있는 stage 전체.",
    )
    args = ap.parse_args()

    gt = load_gt(args.gt_xlsx)
    trace = load_trace(args.trace_jsonl)

    # tgt_exact subset이 stage별로 바뀌면(예: restore_integrity에서 tgt를 GT로 교정)
    # '경계 품질이 떨어진 것처럼' 보이는 착시가 생길 수 있다.
    # 이를 분리하기 위해 최종 stage 기준으로 고정된 tgt_exact_final subset을 만든다.
    tgt_exact_final_keys: set[Tuple[str, int]] = set()
    final_stage_name = "final"
    for (book, pid), gt_pack in gt.items():
        rec_final = trace.get((book, pid, final_stage_name))
        if not rec_final:
            continue
        pred_tgt_final = rec_final.get("tgt_segments") or []
        if not isinstance(pred_tgt_final, list):
            continue
        if _tgt_exact_match([str(x).strip() for x in pred_tgt_final], gt_pack["tgt"]):
            tgt_exact_final_keys.add((book, pid))

    stages: List[str]
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    else:
        stages = sorted({stage for (_book, _pid, stage) in trace.keys()})
        # 사람이 보기 좋은 순서로 재정렬
        order_index = {s: i for i, s in enumerate(DEFAULT_STAGE_ORDER)}
        stages.sort(key=lambda s: (order_index.get(s, 10_000), s))

    aggs: Dict[str, StageAgg] = {s: StageAgg() for s in stages}

    # tgt 완전일치 subset
    aggs_ok: Dict[str, StageAgg] = {s: StageAgg() for s in stages}

    # tgt 완전일치 subset (final 기준 고정)
    aggs_ok_final: Dict[str, StageAgg] = {s: StageAgg() for s in stages}

    for stage in stages:
        agg = aggs[stage]
        agg_ok = aggs_ok[stage]
        agg_ok_final = aggs_ok_final[stage]
        for (book, pid), gt_pack in gt.items():
            rec = trace.get((book, pid, stage))
            if not rec:
                continue
            pred_src = rec.get("src_segments") or []
            pred_tgt = rec.get("tgt_segments") or []
            if not isinstance(pred_src, list):
                continue
            if not isinstance(pred_tgt, list):
                continue

            # src가 아직 없는 stage는 boundary 평가에서 제외
            if len(pred_src) == 0:
                continue

            gt_src = gt_pack["src"]
            gt_tgt = gt_pack["tgt"]

            pred_b = _boundary_positions_normed([str(x).strip() for x in pred_src])
            gold_b = _boundary_positions_normed(gt_src)

            tp_i = len(pred_b & gold_b)
            fp_i = len(pred_b - gold_b)
            fn_i = len(gold_b - pred_b)

            agg.paragraphs += 1
            agg.tp += tp_i
            agg.fp += fp_i
            agg.fn += fn_i

            if _tgt_exact_match([str(x).strip() for x in pred_tgt], gt_tgt):
                agg_ok.paragraphs += 1
                agg_ok.tp += tp_i
                agg_ok.fp += fp_i
                agg_ok.fn += fn_i

            if (book, pid) in tgt_exact_final_keys:
                agg_ok_final.paragraphs += 1
                agg_ok_final.tp += tp_i
                agg_ok_final.fp += fp_i
                agg_ok_final.fn += fn_i

    rows = []
    for subset, agg_map in [("all", aggs), ("tgt_exact", aggs_ok), ("tgt_exact_final", aggs_ok_final)]:
        prev_f1 = None
        for stage in stages:
            a = agg_map[stage]
            f1 = a.f1()
            row = {
                "subset": subset,
                "stage": stage,
                "paragraphs": a.paragraphs,
                "tp": a.tp,
                "fp": a.fp,
                "fn": a.fn,
                "precision": round(a.precision(), 6),
                "recall": round(a.recall(), 6),
                "f1": round(f1, 6),
            }
            if prev_f1 is not None:
                row["delta_f1_vs_prev"] = round(f1 - prev_f1, 6)
            prev_f1 = f1
            rows.append(row)

    out_df = pd.DataFrame(rows)

    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # 콘솔 요약
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
        print(out_df)


if __name__ == "__main__":
    main()
