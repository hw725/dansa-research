#!/usr/bin/env python3
"""경계 위치 불일치 디버깅"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import pandas as pd
from common.s2p_boundary_tagger_loader import get_sa_boundary_tagger

def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))

def get_gold_boundary_positions_normalized(segments: list) -> set:
    """평가 스크립트 방식: 정규화 후 구간 끝 위치"""
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions

def get_tagger_boundary_positions_raw(tagger, text: str, threshold: float) -> set:
    """태거 예측: 원래 텍스트 상의 경계 위치"""
    probs = tagger.predict_boundary_probs(text)
    positions = set()
    for i, prob in enumerate(probs):
        if prob >= threshold and i > 0:  # 첫 문자 제외
            positions.add(i)
    return positions

def get_tagger_boundary_positions_normalized(tagger, text: str, threshold: float) -> set:
    """태거 예측: 경계 위치를 정규화된 인덱스로 변환"""
    probs = tagger.predict_boundary_probs(text)
    positions = set()
    
    norm_idx = 0
    for i, (ch, prob) in enumerate(zip(text, probs)):
        is_space = ch in ' \u3000\t\n\r'
        if not is_space:
            if prob >= threshold and norm_idx > 0:
                positions.add(norm_idx)
            norm_idx += 1
    return positions

def main():
    tagger = get_sa_boundary_tagger()
    
    gold_df = pd.read_csv("datasets/s2p/test.csv")
    input_df = pd.read_csv("datasets/p2s/test.csv")
    
    sent_id = 76  # 첫 번째 문장
    
    input_row = input_df[input_df['문장식별자'] == sent_id]
    gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
    
    tgt_text = str(input_row['번역문'].iloc[0])
    gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
    
    print(f"Text length (raw): {len(tgt_text)}")
    print(f"Text length (normalized): {len(_norm(tgt_text))}")
    print(f"Gold segments: {len(gold_tgt_segs)}")
    
    # Gold 경계
    gold_bounds = get_gold_boundary_positions_normalized(gold_tgt_segs)
    print(f"\nGold boundaries (normalized, first 10): {sorted(gold_bounds)[:10]}")
    
    # 태거 예측 (원래 텍스트)
    pred_bounds_raw = get_tagger_boundary_positions_raw(tagger, tgt_text, 0.5)
    print(f"Pred boundaries (raw, first 10): {sorted(pred_bounds_raw)[:10]}")
    
    # 태거 예측 (정규화)
    pred_bounds_norm = get_tagger_boundary_positions_normalized(tagger, tgt_text, 0.5)
    print(f"Pred boundaries (normalized, first 10): {sorted(pred_bounds_norm)[:10]}")
    
    # 일치 비교
    intersection_raw = gold_bounds & pred_bounds_raw
    intersection_norm = gold_bounds & pred_bounds_norm
    
    print(f"\nIntersection (gold & raw): {len(intersection_raw)}")
    print(f"Intersection (gold & normalized): {len(intersection_norm)}")
    
    # 전체 확률 분포 확인
    probs = tagger.predict_boundary_probs(tgt_text)
    high_prob_chars = [(i, tgt_text[i], p) for i, p in enumerate(probs) if p > 0.3][:10]
    print(f"\nHigh prob positions (>0.3, first 10): {high_prob_chars}")

if __name__ == "__main__":
    main()
