#!/usr/bin/env python3
"""SA 경계 태거 디버깅 스크립트 - JSON 출력"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import re
from common.s2p_boundary_tagger_loader import get_sa_boundary_tagger

def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))

def _boundary_positions(segments: list) -> set:
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions

def main():
    tagger = get_sa_boundary_tagger()
    
    # 데이터 로드
    input_df = pd.read_csv("datasets/p2s/test.csv")
    gold_df = pd.read_csv("datasets/s2p/test.csv")
    
    results = {
        "input_rows": len(input_df),
        "gold_rows": len(gold_df),
        "sentences": []
    }
    
    sent_ids = list(input_df['문장식별자'].unique()[:5])
    
    for sent_id in sent_ids:
        input_row = input_df[input_df['문장식별자'] == sent_id]
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
        
        if input_row.empty or gold_rows.empty:
            continue
        
        tgt_text = str(input_row['번역문'].iloc[0])
        gold_src_segs = [str(r) for r in gold_rows['원문']]
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        pred_src_segs = str(input_row['원문'].iloc[0]).split()
        pred_tgt_segs = tagger.segment_text(tgt_text, threshold=0.5)
        
        gold_tgt_bounds = sorted(_boundary_positions(gold_tgt_segs))
        pred_tgt_bounds = sorted(_boundary_positions(pred_tgt_segs))
        
        probs = tagger.predict_boundary_probs(tgt_text)
        high_prob_count = len([p for p in probs if p > 0.5])
        
        results["sentences"].append({
            "sent_id": int(sent_id),
            "gold_src_count": len(gold_src_segs),
            "pred_src_count": len(pred_src_segs),
            "gold_tgt_count": len(gold_tgt_segs),
            "pred_tgt_count": len(pred_tgt_segs),
            "gold_tgt_bounds": gold_tgt_bounds[:5],
            "pred_tgt_bounds": pred_tgt_bounds[:5],
            "high_prob_positions": high_prob_count,
            "max_prob": max(probs) if probs else 0,
            "mean_prob": sum(probs)/len(probs) if probs else 0,
        })
    
    # JSON 파일로 저장
    with open("debug_tagger_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Saved to debug_tagger_output.json")

if __name__ == "__main__":
    main()
