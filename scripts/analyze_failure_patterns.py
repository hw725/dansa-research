#!/usr/bin/env python3
"""
실패 케이스에서 구체적 패턴 추출

목적:
1. Gold vs Pred 원문 차이 분석
2. 병합/분할이 발생한 지점의 어미/시작 패턴 추출
3. 통계적으로 유의미한 패턴 발견
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path


def extract_sentence_endings(text):
    """문장 끝 어미 추출 (마지막 2-3글자)"""
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    endings = []
    for sent in sentences:
        if len(sent) >= 3:
            endings.append(sent[-3:])
        elif len(sent) >= 2:
            endings.append(sent[-2:])
    return endings


def extract_sentence_beginnings(text):
    """문장 시작 패턴 추출 (처음 2-3글자)"""
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    beginnings = []
    for sent in sentences:
        if len(sent) >= 3:
            beginnings.append(sent[:3])
        elif len(sent) >= 2:
            beginnings.append(sent[:2])
    return beginnings


def find_merge_points(gold_src, pred_src):
    """
    과소분할: Gold가 더 많이 나뉨 → Pred에서 병합된 지점 찾기

    Returns:
        list: 병합된 위치의 Gold 문장 끝 어미들
    """
    gold_sents = [s.strip() for s in gold_src.split('\n') if s.strip()]
    pred_sents = [s.strip() for s in pred_src.split('\n') if s.strip()]

    merge_endings = []

    # Simple heuristic: Gold 문장이 Pred에서 연결되어 있으면 병합으로 간주
    for i in range(len(gold_sents) - 1):
        sent1 = gold_sents[i]
        sent2 = gold_sents[i + 1]
        combined = sent1 + ' ' + sent2

        # Pred에서 이 조합이 하나의 문장으로 나타나는지 확인
        for pred_sent in pred_sents:
            if sent1 in pred_sent and sent2 in pred_sent:
                # 병합 발생! sent1의 끝 어미 기록
                if len(sent1) >= 2:
                    merge_endings.append(sent1[-2:])
                if len(sent1) >= 3:
                    merge_endings.append(sent1[-3:])
                break

    return merge_endings


def find_split_points(gold_src, pred_src):
    """
    과분할: Pred가 더 많이 나뉨 → Pred에서 과도하게 분할된 지점 찾기

    Returns:
        list: 과분할된 위치의 Pred 문장 끝 어미들
    """
    gold_sents = [s.strip() for s in gold_src.split('\n') if s.strip()]
    pred_sents = [s.strip() for s in pred_src.split('\n') if s.strip()]

    split_endings = []

    # Simple heuristic: Pred 문장 2개가 Gold 1개에 포함되면 과분할로 간주
    for i in range(len(pred_sents) - 1):
        sent1 = pred_sents[i]
        sent2 = pred_sents[i + 1]

        for gold_sent in gold_sents:
            if sent1 in gold_sent and sent2 in gold_sent:
                # 과분할 발생! sent1의 끝 어미 기록
                if len(sent1) >= 2:
                    split_endings.append(sent1[-2:])
                if len(sent1) >= 3:
                    split_endings.append(sent1[-3:])
                break

    return split_endings


def main():
    analysis_file = Path("test_results/failure_analysis/analysis_cases_detail.csv")
    output_dir = Path("test_results/failure_analysis")

    df = pd.read_csv(analysis_file)

    print("=" * 80)
    print("실패 케이스 구체적 패턴 분석")
    print("=" * 80)

    # Analyze under-split cases (most common)
    print("\n[1] 과소분할 케이스 분석 (104개)")
    under_split = df[df['src_count_diff'] < 0]

    all_merge_endings = []
    for idx, row in under_split.iterrows():
        endings = find_merge_points(row['gold_src'], row['pred_src'])
        all_merge_endings.extend(endings)

    print(f"   병합 지점 어미 총 {len(all_merge_endings)}개 발견")
    print("\n   [병합 발생한 문장 끝 어미 Top 20]")
    merge_counter = Counter(all_merge_endings)
    for ending, count in merge_counter.most_common(20):
        print(f"      '{ending}': {count}회")

    # Analyze over-split cases
    print("\n[2] 과분할 케이스 분석 (79개)")
    over_split = df[df['src_count_diff'] > 0]

    all_split_endings = []
    for idx, row in over_split.iterrows():
        endings = find_split_points(row['gold_src'], row['pred_src'])
        all_split_endings.extend(endings)

    print(f"   과분할 지점 어미 총 {len(all_split_endings)}개 발견")
    print("\n   [과분할 발생한 문장 끝 어미 Top 20]")
    split_counter = Counter(all_split_endings)
    for ending, count in split_counter.most_common(20):
        print(f"      '{ending}': {count}회")

    # Compare: which endings are problematic?
    print("\n[3] 문제 어미 분석")
    print("\n   [병합되기 쉬운 어미 (should split but didn't)]")
    for ending, count in merge_counter.most_common(10):
        print(f"      '{ending}': {count}회 병합됨")

    print("\n   [과분할되기 쉬운 어미 (should not split but did)]")
    for ending, count in split_counter.most_common(10):
        print(f"      '{ending}': {count}회 과분할됨")

    # Save results
    with open(output_dir / "merge_endings.txt", 'w', encoding='utf-8') as f:
        for ending, count in merge_counter.most_common(50):
            f.write(f"{ending}\t{count}\n")

    with open(output_dir / "split_endings.txt", 'w', encoding='utf-8') as f:
        for ending, count in split_counter.most_common(50):
            f.write(f"{ending}\t{count}\n")

    print(f"\n결과 저장:")
    print(f"   - merge_endings.txt: 병합 어미 Top 50")
    print(f"   - split_endings.txt: 과분할 어미 Top 50")


if __name__ == "__main__":
    main()
