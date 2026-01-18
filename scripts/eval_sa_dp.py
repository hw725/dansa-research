#!/usr/bin/env python3
"""기존 SA 파이프라인 (DP alignment) 평가 - SA Gold 데이터 사용

SA Gold의 구들을 연결한 문장을 입력으로 DP alignment 실행 후 비교
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    
    from sa.sa_aligner import process_single_row
    
    # SA Gold 데이터만 사용
    gold_df = pd.read_csv("datasets/sa/test.csv")
    sent_ids = list(gold_df['문장식별자'].unique()[:args.sample_size])
    
    print("=" * 60)
    print("🎯 SA DP Alignment 평가 (SA Gold 데이터)")
    print("=" * 60)
    
    tp = fp = fn = 0
    count = 0
    errors = 0
    seg_similarities = []
    seg_count_matches = 0
    
    for sent_id in sent_ids:
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id].sort_values('구식별자')
        
        if gold_rows.empty:
            continue
        
        count += 1
        
        # Gold 세그먼트
        gold_src_segs = [str(r) for r in gold_rows['원문']]
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        # 입력: 구들을 연결한 전체 문장
        src_text = ' '.join(gold_src_segs)
        tgt_text = ' '.join(gold_tgt_segs)
        
        row_data = {
            '문장식별자': sent_id,
            '원문': src_text,
            '번역문': tgt_text,
        }
        
        try:
            result_rows = process_single_row(row_data, use_boundary_model=False)
            pred_tgt_segs = [r['번역문'] for r in result_rows]
        except Exception as e:
            errors += 1
            pred_tgt_segs = [tgt_text]
        
        # 세그먼트 수 일치
        if len(pred_tgt_segs) == len(gold_tgt_segs):
            seg_count_matches += 1
            
            # 유사도 계산
            for gold_seg, pred_seg in zip(gold_tgt_segs, pred_tgt_segs):
                gold_norm = _norm(gold_seg)
                pred_norm = _norm(pred_seg)
                if gold_norm == pred_norm:
                    seg_similarities.append(1.0)
                elif gold_norm and pred_norm:
                    sim = difflib.SequenceMatcher(None, gold_norm, pred_norm).ratio()
                    seg_similarities.append(sim)
                else:
                    seg_similarities.append(0.0)
        
        # 경계 비교
        gold_bounds = _boundary_positions(gold_tgt_segs)
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    avg_sim = sum(seg_similarities) / len(seg_similarities) if seg_similarities else 0
    exact_match = sum(1 for s in seg_similarities if s == 1.0)
    
    print(f"\n평가 문장 수:      {count}")
    print(f"오류 수:          {errors}")
    print(f"세그먼트 수 일치:  {seg_count_matches}/{count} ({seg_count_matches/count*100:.1f}%)")
    print(f"번역문 경계 F1:    {f1:.4f}")
    print(f"번역문 Precision:  {precision:.4f}")
    print(f"번역문 Recall:     {recall:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    if seg_similarities:
        print(f"\n📊 세그먼트 유사도:")
        print(f"   평균 유사도:    {avg_sim:.4f}")
        print(f"   완전 일치:      {exact_match}/{len(seg_similarities)} ({exact_match/len(seg_similarities)*100:.1f}%)")
    
    result = {
        "method": "dp_alignment",
        "sample_size": count,
        "errors": errors,
        "seg_match_rate": seg_count_matches / count if count > 0 else 0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "avg_similarity": avg_sim,
        "exact_match_rate": exact_match / len(seg_similarities) if seg_similarities else 0,
    }
    
    with open("sa_dp_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n💾 Saved: sa_dp_eval_result.json")


if __name__ == "__main__":
    main()
