#!/usr/bin/env python3
"""SA 경계 태거 모델 평가

입력(p2s/test.csv) → 태거로 분할 → Gold(s2p/test.csv)와 비교
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import difflib
import pandas as pd
from typing import List, Set
import warnings
warnings.filterwarnings("ignore")


def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))


def _boundary_positions(segments: List[str]) -> Set[int]:
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _similarity(a: str, b: str) -> float:
    a_n, b_n = _norm(a), _norm(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    # SA 경계 태거 로드
    from common.s2p_boundary_tagger_loader import get_sa_boundary_tagger
    tagger = get_sa_boundary_tagger()
    
    # 데이터 로드
    input_df = pd.read_csv("datasets/p2s/test.csv")
    gold_df = pd.read_csv("datasets/s2p/test.csv")
    
    sent_ids = list(input_df['문장식별자'].unique()[:args.sample_size])
    
    print("=" * 60)
    print("🎯 SA 경계 태거 평가 (입력→분할→Gold비교)")
    print("=" * 60)
    print(f"📂 샘플: {len(sent_ids)}문장, threshold={args.threshold}")
    
    # 평가
    tp = fp = fn = 0
    src_exact_count = 0
    seg_sims = []
    
    total_count = 0
    
    for sent_id in sent_ids:
        input_row = input_df[input_df['문장식별자'] == sent_id]
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
        
        if input_row.empty or gold_rows.empty:
            continue
        
        total_count += 1
        
        # 입력 텍스트
        src_text = str(input_row['원문'].iloc[0])
        tgt_text = str(input_row['번역문'].iloc[0])
        
        # Gold 세그먼트
        gold_src_segs = [str(r) for r in gold_rows['원문']]
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        # 원문 분할 및 경계 비교 (통계용)
        pred_src_segs = src_text.split()
        gold_src_bounds = _boundary_positions(gold_src_segs)
        pred_src_bounds = _boundary_positions(pred_src_segs)
        
        if gold_src_bounds == pred_src_bounds:
            src_exact_count += 1
        
        # 번역문 분할 (태거 사용) - 모든 문장에 대해 평가
        pred_tgt_segs = tagger.segment_text(tgt_text, threshold=args.threshold)
        
        gold_tgt_bounds = _boundary_positions(gold_tgt_segs)
        pred_tgt_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_tgt_bounds & pred_tgt_bounds)
        fp += len(pred_tgt_bounds - gold_tgt_bounds)
        fn += len(gold_tgt_bounds - pred_tgt_bounds)
        
        # 세그먼트별 유사도 (개수 일치 시만)
        if len(gold_tgt_segs) == len(pred_tgt_segs):
            for g, p in zip(gold_tgt_segs, pred_tgt_segs):
                seg_sims.append(_similarity(g, p))
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    avg_sim = sum(seg_sims) / len(seg_sims) if seg_sims else 0
    
    print(f"\n📍 원문 경계 일치: {src_exact_count}/{total_count} ({src_exact_count/total_count:.1%})" if total_count > 0 else "\n📍 평가 대상 없음")
    print("\n" + "=" * 60)
    print("📈 번역문 경계 F1 (전체 문장 대상)")
    print("=" * 60)
    print(f"  평가 문장 수:     {total_count}")
    print(f"  번역문 경계 F1:   {f1:.4f}")
    print(f"  번역문 Precision: {precision:.4f}")
    print(f"  번역문 Recall:    {recall:.4f}")
    print(f"  세그먼트 유사도:  {avg_sim:.4f} ({len(seg_sims)}쌍)")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    
    return 0


if __name__ == "__main__":
    exit(main())
