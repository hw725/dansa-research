#!/usr/bin/env python3
"""
Test 실패 케이스 분석 및 분할

목적:
1. Test 데이터의 실패 케이스 502개 추출
2. 250개 분석용 / 252개 검증용으로 랜덤 분할
3. 분석용 케이스에서 공통 패턴 추출
"""

import pandas as pd
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

def extract_failure_cases(eval_log_path):
    """
    eval_log에서 실패 케이스 추출

    Returns:
        list: ['book_name:pid', ...]
    """
    with open(eval_log_path, encoding='utf-8') as f:
        content = f.read()

    # [FAIL] 번역문 불일치 문단: [...] 파싱
    fail_marker = "[FAIL] 번역문 불일치 문단: ["
    start_idx = content.find(fail_marker)
    if start_idx == -1:
        raise ValueError("실패 케이스 리스트를 찾을 수 없습니다")

    start_idx += len(fail_marker)
    end_idx = content.find("]", start_idx)

    fail_list_str = content[start_idx:end_idx]

    # 파싱: '당송팔대가문초구양수1:203', ... → ['당송팔대가문초구양수1:203', ...]
    failures = []
    for item in fail_list_str.split("', '"):
        item = item.strip("'")
        if ':' in item:
            failures.append(item)

    return failures


def split_failures(failures, analysis_size=250, seed=42):
    """
    실패 케이스를 분석용/검증용으로 분할

    Args:
        failures: 실패 케이스 리스트
        analysis_size: 분석용 크기
        seed: 랜덤 시드

    Returns:
        tuple: (analysis_cases, validation_cases)
    """
    random.seed(seed)
    shuffled = failures.copy()
    random.shuffle(shuffled)

    analysis = shuffled[:analysis_size]
    validation = shuffled[analysis_size:]

    return analysis, validation


def load_failure_details(failures, test_output_path, gold_subset_path):
    """
    실패 케이스의 상세 정보 로드

    Returns:
        pd.DataFrame: book_name, paragraph_id, src, tgt, pred_src, error_type 등
    """
    # Load PA output and gold
    pa_output = pd.read_excel(test_output_path)
    gold_df = pd.read_csv(gold_subset_path)

    # Parse failure cases
    failure_data = []
    for fail_case in failures:
        book_name, pid_str = fail_case.split(':')
        pid = int(pid_str)

        # Get gold data
        gold_rows = gold_df[(gold_df['book_name'] == book_name) &
                            (gold_df['문단식별자'] == pid)]

        # Get PA output data
        pa_rows = pa_output[(pa_output['book_name'] == book_name) &
                            (pa_output['문단식별자'] == pid)]

        if len(gold_rows) == 0 or len(pa_rows) == 0:
            continue

        # Extract info
        failure_data.append({
            'book_name': book_name,
            'paragraph_id': pid,
            'gold_src_count': len(gold_rows),
            'pred_src_count': len(pa_rows),
            'src_count_diff': len(pa_rows) - len(gold_rows),
            'gold_src': '\n'.join(gold_rows['원문'].tolist()),
            'gold_tgt': '\n'.join(gold_rows['번역문'].tolist()),
            'pred_src': '\n'.join(pa_rows['원문'].tolist()),
            'pred_tgt': '\n'.join(pa_rows['번역문'].tolist()),
        })

    return pd.DataFrame(failure_data)


def analyze_patterns(failures_df):
    """
    실패 케이스에서 패턴 분석

    Returns:
        dict: 패턴 분석 결과
    """
    patterns = {
        'by_book': Counter(failures_df['book_name']),
        'by_src_count_diff': Counter(failures_df['src_count_diff']),
        'over_split': len(failures_df[failures_df['src_count_diff'] > 0]),  # 과분할
        'under_split': len(failures_df[failures_df['src_count_diff'] < 0]),  # 과소분할
        'exact_split': len(failures_df[failures_df['src_count_diff'] == 0]),  # 개수는 맞지만 경계 틀림
    }

    return patterns


def main():
    # Paths
    eval_log = Path("test_results/grid_search_full_refine/pb0.40_lp0.50/seed1/eval_log_seed1.txt")
    test_output = Path("test_results/grid_search_full_refine/pb0.40_lp0.50/seed1/pa_test_output_seed1.xlsx")
    gold_subset = Path("test_results/grid_search_full_refine/pb0.40_lp0.50/seed1/pa_gold_subset_seed1.csv")

    output_dir = Path("test_results/failure_analysis")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PA Test 실패 케이스 분석")
    print("=" * 80)

    # Step 1: Extract failures
    print("\n[1단계] 실패 케이스 추출 중...")
    failures = extract_failure_cases(eval_log)
    print(f"   총 실패 케이스: {len(failures)}개")

    # Step 2: Split
    print("\n[2단계] 분석용/검증용 분할 중...")
    analysis_cases, validation_cases = split_failures(failures, analysis_size=250, seed=42)
    print(f"   분석용: {len(analysis_cases)}개")
    print(f"   검증용: {len(validation_cases)}개")

    # Save split
    with open(output_dir / "analysis_cases.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_cases, f, ensure_ascii=False, indent=2)

    with open(output_dir / "validation_cases.json", 'w', encoding='utf-8') as f:
        json.dump(validation_cases, f, ensure_ascii=False, indent=2)

    print(f"\n   분할 결과 저장: {output_dir}/")

    # Step 3: Load details for analysis set
    print("\n[3단계] 분석용 케이스 상세 정보 로드 중...")
    analysis_df = load_failure_details(analysis_cases, test_output, gold_subset)
    print(f"   로드 완료: {len(analysis_df)}개")

    # Step 4: Analyze patterns
    print("\n[4단계] 패턴 분석 중...")
    patterns = analyze_patterns(analysis_df)

    print("\n" + "=" * 80)
    print("[PATTERN ANALYSIS] 분석용 250개 케이스 패턴 분석 결과")
    print("=" * 80)

    print("\n[도서별 실패 분포 Top 10]")
    for book, count in patterns['by_book'].most_common(10):
        print(f"   {book}: {count}개")

    print("\n[분할 오류 유형]")
    print(f"   과분할 (예측 > 정답): {patterns['over_split']}개 ({patterns['over_split']/len(analysis_df)*100:.1f}%)")
    print(f"   과소분할 (예측 < 정답): {patterns['under_split']}개 ({patterns['under_split']/len(analysis_df)*100:.1f}%)")
    print(f"   경계만 틀림 (개수 일치): {patterns['exact_split']}개 ({patterns['exact_split']/len(analysis_df)*100:.1f}%)")

    print("\n[원문 개수 차이 분포]")
    for diff, count in sorted(patterns['by_src_count_diff'].items()):
        if diff > 0:
            print(f"   +{diff}개: {count}번 (과분할)")
        elif diff < 0:
            print(f"   {diff}개: {count}번 (과소분할)")
        else:
            print(f"   {diff}개: {count}번 (경계만 틀림)")

    # Save detailed analysis
    analysis_df.to_csv(output_dir / "analysis_cases_detail.csv", index=False, encoding='utf-8-sig')

    with open(output_dir / "pattern_analysis.json", 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON
        json_patterns = {
            'by_book': dict(patterns['by_book']),
            'by_src_count_diff': {str(k): v for k, v in patterns['by_src_count_diff'].items()},
            'over_split': patterns['over_split'],
            'under_split': patterns['under_split'],
            'exact_split': patterns['exact_split'],
        }
        json.dump(json_patterns, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] 분석 완료. 결과 저장: {output_dir}/")
    print(f"   - analysis_cases.json: 분석용 250개 케이스 ID")
    print(f"   - validation_cases.json: 검증용 252개 케이스 ID")
    print(f"   - analysis_cases_detail.csv: 분석용 케이스 상세 정보")
    print(f"   - pattern_analysis.json: 패턴 분석 결과")

    print("\n" + "=" * 80)
    print("다음 단계: analysis_cases_detail.csv를 열어 공통 패턴을 수동으로 확인하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()
