#!/usr/bin/env python3
"""기존 SA 파이프라인 (DP alignment) 평가 - JSON 결과"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
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
    sample_size = 30
    
    from sa.sa_aligner import process_single_row
    
    input_df = pd.read_csv("datasets/pa/test.csv")
    gold_df = pd.read_csv("datasets/sa/test.csv")
    
    sent_ids = list(input_df['문장식별자'].unique()[:sample_size])
    
    tp = fp = fn = 0
    count = 0
    errors = 0
    
    for sent_id in sent_ids:
        input_row = input_df[input_df['문장식별자'] == sent_id]
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
        
        if input_row.empty or gold_rows.empty:
            continue
        
        count += 1
        
        row_data = {
            '문장식별자': sent_id,
            '원문': str(input_row['원문'].iloc[0]),
            '번역문': str(input_row['번역문'].iloc[0]),
        }
        
        try:
            result_rows = process_single_row(row_data, use_boundary_model=False)
            pred_tgt_segs = [r['번역문'] for r in result_rows]
        except Exception as e:
            errors += 1
            pred_tgt_segs = [row_data['번역문']]
        
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        gold_bounds = _boundary_positions(gold_tgt_segs)
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    
    result = {
        "method": "baseline_dp_alignment",
        "sample_size": sample_size,
        "count": count,
        "errors": errors,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    
    with open("sa_baseline_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Saved to sa_baseline_result.json")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
