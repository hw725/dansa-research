#!/usr/bin/env python3
"""Debug: 원문 경계 비교 확인"""
import pandas as pd
import re

def _norm(s):
    return re.sub(r'[\s\u3000]', '', str(s))

def _boundary_positions(segments):
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions

input_df = pd.read_csv("datasets/p2s/test.csv")
gold_df = pd.read_csv("datasets/s2p/test.csv")

sent_ids = list(input_df['문장식별자'].unique()[:5])

for sent_id in sent_ids:
    input_row = input_df[input_df['문장식별자'] == sent_id]
    gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
    
    if input_row.empty or gold_rows.empty:
        continue
    
    src_text = str(input_row['원문'].iloc[0])
    gold_src_segs = [str(r) for r in gold_rows['원문']]
    
    pred_src_segs = src_text.split()
    
    gold_bounds = _boundary_positions(gold_src_segs)
    pred_bounds = _boundary_positions(pred_src_segs)
    
    print(f"\n=== Sent {sent_id} ===")
    print(f"Input src (split): {pred_src_segs[:3]}...")
    print(f"Gold src segs: {gold_src_segs[:3]}...")
    print(f"Gold bounds: {sorted(gold_bounds)[:5]}")
    print(f"Pred bounds: {sorted(pred_bounds)[:5]}")
    print(f"Match: {gold_bounds == pred_bounds}")
