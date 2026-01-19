#!/usr/bin/env python3
"""기존 SA 파이프라인 (DP alignment) vs 경계 모델 비교 평가

- baseline: 기존 SA 파이프라인 (DP alignment)
- boundary: 경계 모델만 사용
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import pandas as pd
from typing import List, Set

def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))

def _boundary_positions(segments: List[str]) -> Set[int]:
    """세그먼트 끝 위치에서 경계 계산"""
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

def evaluate_baseline(sample_size: int = 50):
    """기존 SA 파이프라인 (DP alignment) 평가"""
    from s2p.s2p_aligner import process_single_row
    
    input_df = pd.read_csv("datasets/p2s/test.csv")
    gold_df = pd.read_csv("datasets/s2p/test.csv")
    
    sent_ids = list(input_df['문장식별자'].unique()[:sample_size])
    
    tp = fp = fn = 0
    count = 0
    
    for sent_id in sent_ids:
        input_row = input_df[input_df['문장식별자'] == sent_id]
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
        
        if input_row.empty or gold_rows.empty:
            continue
        
        count += 1
        
        # 입력 데이터 구성
        row_data = {
            '문장식별자': sent_id,
            '원문': str(input_row['원문'].iloc[0]),
            '번역문': str(input_row['번역문'].iloc[0]),
        }
        
        # 기존 SA 파이프라인 실행
        try:
            result_rows = process_single_row(row_data, use_boundary_model=False)
            pred_tgt_segs = [r['번역문'] for r in result_rows]
        except Exception as e:
            print(f"Error processing {sent_id}: {e}")
            pred_tgt_segs = [row_data['번역문']]  # 폴백: 단일 세그먼트
        
        # Gold 세그먼트
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        # 경계 비교
        gold_bounds = _boundary_positions(gold_tgt_segs)
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    return {
        "method": "baseline (DP alignment)",
        "count": count,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("SA Pipeline Comparison Evaluation")
    print("=" * 60)
    
    # Baseline 평가
    print("\n[1] Evaluating baseline (DP alignment)...")
    result = evaluate_baseline(args.sample_size)
    
    print(f"\n--- {result['method']} ---")
    print(f"Samples: {result['count']}")
    print(f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("Done")

if __name__ == "__main__":
    main()
