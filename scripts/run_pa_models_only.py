#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PA models-only 파이프라인 (임베더/파서/토크나이저 없이)

목적:
- 현재 PA의 --use-boundary-model(strict)은 bge 임베더 + (SuPar-Kanbun/Stanza) 파서를 강제한다.
- "학습모델만"으로도 성능이 어디까지 나오는지 분리 진단하기 위해,
  char-based 학습 모델(dual_encoder_boundary_aware_pa.pt)만으로 문단→문장 병렬을 생성한다.

구성:
- tgt 문장 분할: pa.sentence_splitter.split_target_sentences_advanced(가능하면)
- src 문장 경계 추정: 후보 경계(구두점 + 균등 grid)를 두고, tgt 문장 순서에 맞춰
  단조(monotonic) greedy로 경계 위치를 선택한다.
- 출력 포맷: integrity_report.py의 run_pa_output_vs_gold_report에 바로 넣을 수 있는
  (문단식별자, book_name, 원문, 번역문) CSV

주의:
- 이 스크립트는 "최고 성능"이 목적이 아니라 병목 분해(임베더/파서 영향 제거)가 목적이다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"


def _read_csv_or_xlsx(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _split_tgt_sentences(tgt_paragraph: str, *, allow_fallback: bool) -> List[str]:
    text = "" if tgt_paragraph is None else str(tgt_paragraph)
    text = text.strip()
    if not text:
        return [""]

    # 번역문 문장 분할은 PA 정책상 불변(=동일 splitter를 강제)이어야 한다.
    try:
        from pa.sentence_splitter import split_target_sentences_advanced

        sents = split_target_sentences_advanced(text)
        sents = [str(s).strip() for s in sents if str(s).strip()]
        return sents if sents else [text]
    except Exception:
        if not allow_fallback:
            raise RuntimeError(
                "번역문 문장 분할기(pa.sentence_splitter.split_target_sentences_advanced) 로드/실행에 실패했습니다. "
                "models-only 진단에서도 번역문 문장 분할은 동일 로직을 강제합니다. "
                "(환경 문제로 splitter가 실패한다면, --allow-tgt-split-fallback 옵션으로만 폴백을 허용할 수 있습니다.)"
            )

        # 진단 편의용 폴백(기본 OFF): 줄바꿈 단위
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        return parts if parts else [text]


def _candidate_boundaries(src_text: str, *, grid: int) -> List[int]:
    s = "" if src_text is None else str(src_text)
    n = len(s)
    if n == 0:
        return [0]

    boundaries = {0, n}

    punct = set("。！？!?；;：:．.\n\r\t")
    for i, ch in enumerate(s):
        if ch in punct:
            # 구두점 직후를 경계 후보로
            j = i + 1
            if 0 < j < n:
                boundaries.add(j)

    if grid and grid > 0:
        for j in range(grid, n, grid):
            boundaries.add(j)

    out = sorted(boundaries)
    # 0..n 범위 보장
    out = [b for b in out if 0 <= b <= n]
    # 0은 항상 포함
    if out[0] != 0:
        out = [0] + out
    return out


def _score(matcher, src_seg: str, tgt_seg: str, *, boundary_weight: float) -> float:
    src_seg = "" if src_seg is None else str(src_seg)
    tgt_seg = "" if tgt_seg is None else str(tgt_seg)
    if not src_seg and not tgt_seg:
        return 0.0
    return float(matcher.compute_combined_score(src_seg, tgt_seg, boundary_weight=boundary_weight))


def _greedy_align(
    matcher,
    src_text: str,
    tgt_sents: List[str],
    *,
    grid: int,
    boundary_weight: float,
    max_lookahead: int,
    max_len_factor: float,
    min_src_len: int,
) -> List[str]:
    s = "" if src_text is None else str(src_text)
    n = len(s)
    if not tgt_sents:
        return [s]

    if len(tgt_sents) == 1:
        return [s]

    bounds = _candidate_boundaries(s, grid=grid)
    # bounds는 0..n 포함, 오름차순

    # 각 문장마다 src segment를 하나씩 만든다.
    segs: List[str] = []
    start = 0
    start_idx = 0

    # 미리 boundary index로 빠르게 범위를 찾기 위해 start_idx를 유지
    for sent_i in range(len(tgt_sents) - 1):
        tgt = tgt_sents[sent_i]

        # 남은 src 길이 / 남은 문장 수로 대략 목표 길이를 잡는다.
        remaining_src = n - start
        remaining_sents = len(tgt_sents) - sent_i
        target_len = max(min_src_len, remaining_src // max(1, remaining_sents))
        max_len = int(max(min_src_len, target_len * max_len_factor))

        # 다음 후보 end를 만든다.
        # - 최소 길이 보장
        # - 마지막 문장에 최소 1 char 이상 남기기
        end_candidates: List[int] = []
        # start_idx는 bounds에서 start 이상인 첫 idx를 가리키도록 보정
        while start_idx + 1 < len(bounds) and bounds[start_idx] < start:
            start_idx += 1

        # 후보는 bounds 중 start보다 큰 것들
        for b in bounds[start_idx + 1 :]:
            if b <= start:
                continue
            if (b - start) < min_src_len:
                continue
            if (b - start) > max_len:
                break
            # 남은 문장 수 - 1개를 위해 적어도 min_src_len 남겨두기
            if (n - b) < min_src_len:
                continue
            end_candidates.append(b)
            if len(end_candidates) >= max_lookahead:
                break

        # 후보가 너무 없으면 grid를 무시하고 최소 길이로 하나는 만든다.
        if not end_candidates:
            fallback_end = min(n - min_src_len, start + max(min_src_len, target_len))
            fallback_end = max(start + min_src_len, fallback_end)
            end_candidates = [fallback_end]

        best_end = end_candidates[0]
        best_score = -1e9
        for end in end_candidates:
            sc = _score(matcher, s[start:end], tgt, boundary_weight=boundary_weight)
            if sc > best_score:
                best_score = sc
                best_end = end

        segs.append(s[start:best_end])
        start = best_end

    # 마지막 문장은 remainder
    segs.append(s[start:])
    # 안전장치: 문장 수 일치
    if len(segs) != len(tgt_sents):
        # 불일치 시 단순 보정
        while len(segs) < len(tgt_sents):
            segs.append("")
        segs = segs[: len(tgt_sents)]
    return segs


def main() -> int:
    p = argparse.ArgumentParser(description="Run PA using only learned models (no embedder/parsers)")
    p.add_argument("input_pd", type=str, help="PD 형식 입력 CSV/XLSX (문단 단위)")
    p.add_argument("out_csv", type=str, help="PA output CSV (문장 병렬)")
    p.add_argument("--gold", type=str, default="datasets/pa/test_100_from_pd.csv", help="gold sentences CSV/XLSX")
    p.add_argument("--alignment", type=str, default=str(MODELS_ROOT / "dual_encoder_boundary_aware_pa.pt"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--boundary-weight", type=float, default=0.3)
    p.add_argument("--grid", type=int, default=10, help="src 후보 경계 grid(문자 단위)")
    p.add_argument("--max-lookahead", type=int, default=80, help="각 tgt 문장마다 평가할 end 후보 최대 개수")
    p.add_argument("--max-len-factor", type=float, default=3.0, help="(대략 목표 길이)*factor 만큼만 탐색")
    p.add_argument("--min-src-len", type=int, default=6, help="src segment 최소 길이")
    p.add_argument(
        "--allow-tgt-split-fallback",
        action="store_true",
        help="(기본 OFF) 번역문 splitter 로드 실패 시 폴백을 허용(진단용)",
    )
    args = p.parse_args()

    input_path = Path(args.input_pd)
    out_path = Path(args.out_csv)
    gold_path = Path(args.gold)

    from common.boundary_aware_alignment_loader import BoundaryAwareAlignmentMatcher

    matcher = BoundaryAwareAlignmentMatcher(
        model_path=Path(args.alignment),
        device=args.device,
        boundary_weight=float(args.boundary_weight),
    )

    df = _read_csv_or_xlsx(input_path).copy()
    required = {"문단식별자", "원문", "번역문"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"입력에 필수 컬럼이 없습니다: {sorted(required - set(df.columns))}")
    if "book_name" not in df.columns:
        df["book_name"] = ""
    df["book_name"] = df["book_name"].fillna("").astype(str)

    out_rows = []
    for _, row in df.iterrows():
        pid = int(row["문단식별자"])
        book = str(row.get("book_name", "")).strip()
        src = "" if pd.isna(row["원문"]) else str(row["원문"])  # 원문 문단
        tgt = "" if pd.isna(row["번역문"]) else str(row["번역문"])  # 번역 문단

        tgt_sents = _split_tgt_sentences(tgt, allow_fallback=bool(args.allow_tgt_split_fallback))
        src_segs = _greedy_align(
            matcher,
            src,
            tgt_sents,
            grid=int(args.grid),
            boundary_weight=float(args.boundary_weight),
            max_lookahead=int(args.max_lookahead),
            max_len_factor=float(args.max_len_factor),
            min_src_len=int(args.min_src_len),
        )

        # sentence index는 integrity_report에 필수는 아니지만, 정렬을 위해 넣어둔다.
        for i, (src_seg, tgt_sent) in enumerate(zip(src_segs, tgt_sents), start=1):
            out_rows.append(
                {
                    "book_name": book,
                    "문단식별자": pid,
                    "문장식별자": i,
                    "원문": src_seg,
                    "번역문": tgt_sent,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ models-only PA output 저장: {out_path} (rows={len(out_df)})")

    # gold가 있으면 동일 리포트로 점수 출력
    if gold_path.exists():
        from integrity_report import run_pa_output_vs_gold_report

        return int(run_pa_output_vs_gold_report(out_path, gold_path))

    print(f"⚠️ gold 파일이 없어 평가를 건너뜁니다: {gold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
