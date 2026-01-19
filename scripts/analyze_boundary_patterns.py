#!/usr/bin/env python3
"""PA boundary 학습 데이터에서 종결/연결 패턴 통계 추출"""

import json
from collections import Counter
from pathlib import Path

def analyze_boundary_patterns(train_jsonl: str, top_n: int = 30):
    """경계 직전 N-gram 패턴 통계"""
    
    terminal_bigrams = Counter()  # 종결형 경계 직전 2-gram
    terminal_trigrams = Counter()  # 종결형 경계 직전 3-gram
    
    total_boundaries = 0
    total_samples = 0
    
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('task') != 'pa':
                continue
            
            text = obj['text']
            labels = obj['labels']
            
            total_samples += 1
            
            # 경계 위치 (label=1) 찾기
            for i, label in enumerate(labels):
                if label == 1:
                    total_boundaries += 1
                    
                    # 경계 직전 2-gram
                    if i >= 1:
                        bigram = ''.join(text[i-1:i+1])
                        terminal_bigrams[bigram] += 1
                    
                    # 경계 직전 3-gram
                    if i >= 2:
                        trigram = ''.join(text[i-2:i+1])
                        terminal_trigrams[trigram] += 1
    
    print(f"=== PA Boundary Pattern 통계 ===")
    print(f"Total samples: {total_samples:,}")
    print(f"Total boundaries: {total_boundaries:,}")
    print(f"\n종결형 경계 직전 2-gram Top {top_n}:")
    for gram, count in terminal_bigrams.most_common(top_n):
        print(f"  {gram:10s}: {count:6d} ({count/total_boundaries*100:.2f}%)")
    
    print(f"\n종결형 경계 직전 3-gram Top {top_n}:")
    for gram, count in terminal_trigrams.most_common(top_n):
        print(f"  {gram:15s}: {count:6d} ({count/total_boundaries*100:.2f}%)")
    
    return terminal_bigrams, terminal_trigrams

if __name__ == '__main__':
    bigrams, trigrams = analyze_boundary_patterns('datasets/sentence_boundary/train.jsonl')
