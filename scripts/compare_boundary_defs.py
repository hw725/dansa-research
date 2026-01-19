#!/usr/bin/env python3
"""SA 학습 데이터 vs 평가 경계 정의 비교"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import pandas as pd
from collections import defaultdict

def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))

def _boundary_positions_end(segments: list) -> set:
    """현재 평가 스크립트 방식: 세그먼트 끝에서 경계 계산"""
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions

def _boundary_positions_start(segments: list) -> set:
    """구의 시작점 기준 경계 (학습 데이터 방식)"""
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        if i > 0:  # 첫 번째가 아니면 경계
            positions.add(cursor)
        cursor += len(_norm(seg))
    return positions

def load_gold_boundary_labels(csv_path: Path, text_col: str = "번역문"):
    """학습 스크립트와 동일한 방식으로 B 레이블 위치 수집"""
    df = pd.read_csv(csv_path)
    
    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row['문장식별자']
        phrase_id = row['구식별자']
        text = str(row[text_col])
        sent_groups[sent_id].append((phrase_id, text))
    
    results = []
    for sent_id, phrases in list(sent_groups.items())[:5]:
        phrases.sort(key=lambda x: x[0])
        
        full_text = ""
        labels = ""
        b_positions = []
        
        for i, (_, phrase_text) in enumerate(phrases):
            phrase_text = phrase_text.strip()
            if not phrase_text:
                continue
            
            if i > 0 and full_text:
                full_text += " "
                labels += "O"
            
            for j, char in enumerate(phrase_text):
                if j == 0:
                    b_positions.append(len(full_text))
                    labels += "B"
                else:
                    labels += "O"
                full_text += char
        
        results.append({
            "sent_id": int(sent_id),
            "num_phrases": len(phrases),
            "text_len": len(full_text),
            "b_positions": b_positions,
            "first_5_labels": labels[:50]
        })
    
    return results

def main():
    # Gold 데이터 로드
    gold_df = pd.read_csv("datasets/s2p/test.csv")
    
    # 첫 몇 개 문장의 경계 비교
    sent_ids = list(gold_df['문장식별자'].unique()[:5])
    
    results = {"comparison": []}
    
    for sent_id in sent_ids:
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id]
        gold_tgt_segs = [str(r) for r in gold_rows['번역문']]
        
        bounds_end = sorted(_boundary_positions_end(gold_tgt_segs))
        bounds_start = sorted(_boundary_positions_start(gold_tgt_segs))
        
        results["comparison"].append({
            "sent_id": int(sent_id),
            "num_segs": len(gold_tgt_segs),
            "bounds_end_method": bounds_end[:5],
            "bounds_start_method": bounds_start[:5],
            "match": bounds_end == bounds_start
        })
    
    # 학습 데이터 방식 확인
    results["train_labels"] = load_gold_boundary_labels(Path("datasets/s2p/test.csv"))
    
    with open("boundary_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Saved to boundary_comparison.json")

if __name__ == "__main__":
    main()
