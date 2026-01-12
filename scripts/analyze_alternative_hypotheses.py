#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""대립 가설 검증: 예상 밖 용법 검출

목표:
1. 의외성 지수 (Unexpectedness Index): 예상 밖 용례 비율
2. 맥락 다양성 검증 (Context Diversity): 용법이 얼마나 다양한가
3. 대립 가설 테스트 (Alternative Hypothesis): 사서 외 다른 텍스트가 중심이라면?

사용 예:
    python scripts/analyze_alternative_hypotheses.py \
        --csv hyeonto/datasets/pa_train_full.csv \
        --cluster-csv hyeonto/reports/recluster_k16_child/reclustered.csv \
        --out-dir hyeonto/reports/bias_validation/alternative_hypotheses \
        --expected-contexts configs/expected_contexts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats


def compute_unexpectedness_index(
    marker: str,
    df: pd.DataFrame,
    expected_contexts: Dict[str, List[str]],
    context_col: str = 'syntactic_function'
) -> Dict[str, float]:
    """의외성 지수 계산: 예상 밖 용례가 얼마나 많은가

    Args:
        marker: 분석할 마커
        df: 전체 데이터프레임
        expected_contexts: {marker: [expected_context1, expected_context2, ...]}
        context_col: 문맥/용법을 나타내는 컬럼명

    Returns:
        {
            'total_count': int,
            'expected_count': int,
            'unexpected_count': int,
            'unexpectedness_index': float,  # 0~1
        }
    """
    marker_df = df[df['marker'] == marker]
    total_count = len(marker_df)

    if total_count == 0:
        return {
            'total_count': 0,
            'expected_count': 0,
            'unexpected_count': 0,
            'unexpectedness_index': 0.0,
        }

    # 예상된 문맥에 해당하는 용례 수
    expected_list = expected_contexts.get(marker, [])

    if not expected_list:
        # 예상 문맥이 없으면 의외성 1.0 (모두 의외)
        return {
            'total_count': total_count,
            'expected_count': 0,
            'unexpected_count': total_count,
            'unexpectedness_index': 1.0,
        }

    # context_col에 예상 문맥이 포함되는지 확인
    expected_count = 0
    if context_col in marker_df.columns:
        for expected_ctx in expected_list:
            expected_count += marker_df[context_col].astype(str).str.contains(
                expected_ctx, case=False, regex=False
            ).sum()

    unexpected_count = total_count - expected_count
    unexpectedness_index = unexpected_count / total_count if total_count > 0 else 0.0

    return {
        'total_count': total_count,
        'expected_count': expected_count,
        'unexpected_count': unexpected_count,
        'unexpectedness_index': unexpectedness_index,
    }


def compute_context_diversity(
    marker: str,
    df: pd.DataFrame,
    context_col: str = 'syntactic_function'
) -> Dict[str, float]:
    """맥락 다양성 계산: 하나의 마커가 얼마나 다양한 문맥에서 사용되는가

    Returns:
        {
            'unique_contexts': int,
            'entropy': float,  # Shannon entropy
            'dominant_context': str,
            'dominant_ratio': float,
        }
    """
    marker_df = df[df['marker'] == marker]

    if len(marker_df) == 0:
        return {
            'unique_contexts': 0,
            'entropy': 0.0,
            'dominant_context': 'N/A',
            'dominant_ratio': 0.0,
        }

    if context_col not in marker_df.columns:
        # syntactic_function이 없으면 book으로 대체
        context_col = 'book'

    context_counts = marker_df[context_col].value_counts()
    unique_contexts = len(context_counts)

    # Shannon entropy 계산
    probs = context_counts / context_counts.sum()
    entropy = -sum(probs * np.log2(probs + 1e-10))  # 0 방지

    # 최빈 문맥
    dominant_context = context_counts.index[0] if len(context_counts) > 0 else 'N/A'
    dominant_ratio = context_counts.iloc[0] / len(marker_df) if len(context_counts) > 0 else 0.0

    return {
        'unique_contexts': unique_contexts,
        'entropy': entropy,
        'dominant_context': dominant_context,
        'dominant_ratio': dominant_ratio,
    }


def compute_canonicity(df: pd.DataFrame, target_books: List[str], book_col: str = 'book') -> float:
    """특정 서적군의 비중 계산

    Args:
        df: 데이터프레임
        target_books: 타겟 서적 리스트
        book_col: 서적명 컬럼

    Returns:
        canonicity: 0~100 (퍼센트)
    """
    if len(df) == 0:
        return 0.0

    target_count = sum(df[book_col].isin(target_books))
    return target_count / len(df) * 100


def test_alternative_hypothesis(
    df: pd.DataFrame,
    alternative_books: List[str],
    weight: float = 5.0,
    book_col: str = 'book'
) -> Dict[str, float]:
    """대립 가설 테스트: 다른 텍스트 집단이 중심이라면?

    Args:
        df: 전체 데이터프레임 (parent_cluster_id 컬럼 필요)
        alternative_books: 대안 가설의 중심 서적들
        weight: 대안 가중치
        book_col: 서적명 컬럼

    Returns:
        {
            'alternative_canonicity': float,
            'saseo_canonicity': float,
            'delta': float,
            'effect_size': float,  # Cohen's d
        }
    """
    saseo_books = ['논어집주', '맹자집주', '대학장구', '중용장구']

    # 사서 중심성 계산 (p6 클러스터 기준)
    if 'parent_cluster_id' in df.columns:
        # parent_cluster_id가 숫자인 경우와 문자열인 경우 모두 처리
        p6_df = df[(df['parent_cluster_id'] == 6) | (df['parent_cluster_id'] == 'p6')]
        saseo_canonicity = compute_canonicity(p6_df, saseo_books, book_col)

        # 대안 중심성 계산
        alternative_canonicity = compute_canonicity(p6_df, alternative_books, book_col)
    else:
        # parent_cluster_id가 없으면 전체 데이터 기준
        saseo_canonicity = compute_canonicity(df, saseo_books, book_col)
        alternative_canonicity = compute_canonicity(df, alternative_books, book_col)

    delta = abs(saseo_canonicity - alternative_canonicity)

    # 효과 크기 계산 (Cohen's d 근사)
    # 표준편차 추정: 비율의 표준편차 = sqrt(p*(1-p)/n)
    n = len(df)
    p_saseo = saseo_canonicity / 100
    p_alt = alternative_canonicity / 100

    std_saseo = np.sqrt(p_saseo * (1 - p_saseo) / n) * 100
    std_alt = np.sqrt(p_alt * (1 - p_alt) / n) * 100
    pooled_std = np.sqrt((std_saseo**2 + std_alt**2) / 2)

    effect_size = delta / pooled_std if pooled_std > 0 else 0.0

    return {
        'alternative_canonicity': alternative_canonicity,
        'saseo_canonicity': saseo_canonicity,
        'delta': delta,
        'effect_size': effect_size,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="대립 가설 검증")
    p.add_argument("--csv", type=Path, required=True, help="원본 CSV (pa_train_merged.csv)")
    p.add_argument("--cluster-csv", type=Path, required=True, help="클러스터 결과 CSV")
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/bias_validation/alternative_hypotheses"))
    p.add_argument("--expected-contexts", type=Path, help="예상 문맥 JSON 파일 (선택)")
    p.add_argument("--min-count", type=int, default=50, help="분석 대상 최소 출현 횟수")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    if not args.cluster_csv.exists():
        print(f"❌ 파일 없음: {args.cluster_csv}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 CSV 로드...")
    df_cluster = pd.read_csv(args.cluster_csv)

    print(f"📊 클러스터 데이터: {len(df_cluster):,}행")

    # src_left에서 현토 마커 추출 (한글 패턴: \p{Hangul}+)
    import re

    def extract_hyeonto_markers(text):
        """src 텍스트에서 한글 현토 마커 추출"""
        if pd.isna(text):
            return ''
        # 한글만 추출 (한자 뒤에 붙은 토씨)
        matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
        return ''.join(matches) if matches else ''

    df_cluster['marker'] = df_cluster['src_left'].apply(extract_hyeonto_markers)

    # book 컬럼 생성 (book_name에서)
    df_cluster['book'] = df_cluster['book_name']

    # 빈 마커 제거
    df_cluster = df_cluster[df_cluster['marker'] != '']

    print(f"📊 마커 추출 완료: {len(df_cluster):,}행, {df_cluster['marker'].nunique():,}개 고유 마커")

    # 예상 문맥 로드 (있으면)
    expected_contexts = {}
    if args.expected_contexts and args.expected_contexts.exists():
        with open(args.expected_contexts, 'r', encoding='utf-8') as f:
            expected_contexts = json.load(f)
        print(f"✅ 예상 문맥 로드: {len(expected_contexts)}개 마커")
    else:
        print("⚠️ 예상 문맥 JSON 없음 → 기본 분석만 수행")
        # 기본 예상 문맥 (예시)
        expected_contexts = {
            '니': ['의문', '확인'],
            '라': ['서술', '명령'],
            '되': ['피동'],
            '이': ['주격'],
            '를': ['목적격'],
        }

    # 1. 의외성 지수 계산
    print(f"\n{'='*60}")
    print("1️⃣ 의외성 지수 (Unexpectedness Index) 계산...")
    print(f"{'='*60}")

    marker_counts = df_cluster['marker'].value_counts()
    frequent_markers = marker_counts[marker_counts >= args.min_count].index.tolist()

    print(f"📊 분석 대상: {len(frequent_markers)}개 마커 (출현 ≥ {args.min_count})")

    unexpectedness_results = []

    for marker in frequent_markers:
        result = compute_unexpectedness_index(
            marker, df_cluster, expected_contexts, context_col='syntactic_function'
        )
        result['marker'] = marker
        unexpectedness_results.append(result)

    df_unexpectedness = pd.DataFrame(unexpectedness_results)
    df_unexpectedness = df_unexpectedness.sort_values('unexpectedness_index', ascending=False)

    # 통계
    mean_unexpectedness = df_unexpectedness['unexpectedness_index'].mean()
    high_unexpectedness = df_unexpectedness[df_unexpectedness['unexpectedness_index'] >= 0.5]

    print(f"\n📈 의외성 지수 통계:")
    print(f"  - 평균: {mean_unexpectedness:.3f}")
    print(f"  - 위험 마커 (≥0.5): {len(high_unexpectedness)}개")

    if len(high_unexpectedness) > 0:
        print(f"\n⚠️ 고위험 마커 (예상 밖 용례 50% 이상):")
        for _, row in high_unexpectedness.head(10).iterrows():
            print(f"  • {row['marker']:6s}: {row['unexpectedness_index']:.2f} "
                  f"({row['unexpected_count']:,}/{row['total_count']:,})")

    # 2. 맥락 다양성 검증
    print(f"\n{'='*60}")
    print("2️⃣ 맥락 다양성 (Context Diversity) 검증...")
    print(f"{'='*60}")

    diversity_results = []

    for marker in frequent_markers:
        result = compute_context_diversity(
            marker, df_cluster, context_col='syntactic_function'
        )
        result['marker'] = marker
        diversity_results.append(result)

    df_diversity = pd.DataFrame(diversity_results)
    df_diversity = df_diversity.sort_values('entropy', ascending=False)

    high_diversity = df_diversity[df_diversity['entropy'] >= 2.5]

    print(f"\n📈 맥락 다양성 통계:")
    print(f"  - 평균 Entropy: {df_diversity['entropy'].mean():.3f}")
    print(f"  - 고다양성 마커 (entropy ≥ 2.5): {len(high_diversity)}개")

    if len(high_diversity) > 0:
        print(f"\n🌈 고다양성 마커:")
        for _, row in high_diversity.head(10).iterrows():
            print(f"  • {row['marker']:6s}: entropy {row['entropy']:.2f}, "
                  f"{row['unique_contexts']}개 문맥, "
                  f"최빈 {row['dominant_ratio']*100:.1f}%")

    # 3. 대립 가설 테스트
    print(f"\n{'='*60}")
    print("3️⃣ 대립 가설 테스트 (Alternative Hypothesis)")
    print(f"{'='*60}")

    alternative_tests = [
        {
            'name': '삼경 중심성',
            'books': ['시경', '서경', '역경'],
        },
        {
            'name': '문집 중심성',
            'books': ['동문선', '열녀전'],
        },
        {
            'name': '기타 중심성',
            'books': ['소학', '근사록', '심경부주', '가례'],
        },
    ]

    alternative_results = []

    for test in alternative_tests:
        print(f"\n📊 {test['name']} 검증...")
        result = test_alternative_hypothesis(
            df_cluster, test['books'], weight=5.0, book_col='book'
        )
        result['test_name'] = test['name']
        result['books'] = ', '.join(test['books'])
        alternative_results.append(result)

        print(f"  - {test['name']}: {result['alternative_canonicity']:.2f}%")
        print(f"  - 사서 Canonicity: {result['saseo_canonicity']:.2f}%")
        print(f"  - Δ = {result['delta']:.2f}%, Effect Size = {result['effect_size']:.3f}")

        if result['effect_size'] > 0.8:
            print(f"  ✅ 사서가 유의하게 더 중심적 (큰 효과)")
        elif result['effect_size'] > 0.5:
            print(f"  ⚠️ 중간 정도 차이")
        else:
            print(f"  ❌ 차이 미미 (편향 위험)")

    df_alternative = pd.DataFrame(alternative_results)

    # 4. 종합 판정
    print(f"\n{'='*60}")
    print("🎯 종합 판정")
    print(f"{'='*60}")

    # 편향 점수 계산 (0~1, 낮을수록 좋음)
    bias_score = 0.0

    # 4.1 의외성 지수 기여 (30%)
    unexpectedness_penalty = mean_unexpectedness * 0.3
    bias_score += unexpectedness_penalty

    # 4.2 맥락 다양성 기여 (20%)
    # 평균 entropy가 높으면 다양성이 높음 → 예상 밖 용례 많음
    mean_entropy = df_diversity['entropy'].mean()
    diversity_penalty = min(mean_entropy / 5.0, 1.0) * 0.2  # 정규화
    bias_score += diversity_penalty

    # 4.3 대립 가설 기여 (50%)
    # 평균 effect size가 낮으면 사서가 특별하지 않음 → 편향
    mean_effect_size = df_alternative['effect_size'].mean()
    alternative_penalty = max(0, 1.0 - mean_effect_size / 2.0) * 0.5
    bias_score += alternative_penalty

    print(f"\n📊 편향 점수 (Bias Score): {bias_score:.3f}")
    print(f"  - 의외성 기여: {unexpectedness_penalty:.3f} (30%)")
    print(f"  - 다양성 기여: {diversity_penalty:.3f} (20%)")
    print(f"  - 대립가설 기여: {alternative_penalty:.3f} (50%)")

    print(f"\n🔍 해석:")
    if bias_score < 0.3:
        print("  ✅ 편향 가능성 낮음 (Bias Score < 0.3)")
        print("     → 예상 밖 용례가 적고, 사서 중심성이 강함")
        print("     → 통계가 연구자 직관이 아닌 실제 데이터를 반영")
    elif bias_score < 0.5:
        print("  ⚠️ 중간 수준 편향 (0.3 ≤ Bias Score < 0.5)")
        print("     → 추가 검증 권장")
        print("     → 일부 마커에서 예상 밖 용례 존재")
    else:
        print("  ❌ 편향 가능성 높음 (Bias Score ≥ 0.5)")
        print("     → 예상 밖 용례가 많거나 사서 중심성 약함")
        print("     → 가중치/정규화 재검토 필요")

    # 5. 결과 저장
    print(f"\n💾 결과 저장...")

    # CSV 저장
    unexpectedness_csv = args.out_dir / "unexpectedness_index.csv"
    df_unexpectedness.to_csv(unexpectedness_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {unexpectedness_csv}")

    diversity_csv = args.out_dir / "context_diversity.csv"
    df_diversity.to_csv(diversity_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {diversity_csv}")

    alternative_csv = args.out_dir / "alternative_hypothesis_results.csv"
    df_alternative.to_csv(alternative_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ {alternative_csv}")

    # 마크다운 보고서
    report_md = args.out_dir / "alternative_hypothesis_report.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# 대립 가설 검증 보고서\n\n")
        f.write(f"**분석일**: 2026-01-10\n")
        f.write(f"**데이터**: {args.csv.name}, {args.cluster_csv.name}\n")
        f.write(f"**분석 마커 수**: {len(frequent_markers)}\n\n")

        f.write("## 1. 테스트 목적\n\n")
        f.write("**핵심 질문**: 내가 예상한 용법만 검증하는 편향된 통계가 아닌가?\n\n")
        f.write("**검증 방법**:\n")
        f.write("1. **의외성 지수**: 예상 밖 용례가 얼마나 많은가?\n")
        f.write("2. **맥락 다양성**: 하나의 마커가 얼마나 다양한 문맥에서 쓰이는가?\n")
        f.write("3. **대립 가설**: 사서 외 다른 텍스트가 중심이라면 어떻게 되는가?\n\n")

        f.write("## 2. 결과\n\n")
        f.write("### 2.1 의외성 지수 (Unexpectedness Index)\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 평균 의외성 지수 | {mean_unexpectedness:.3f} |\n")
        f.write(f"| 위험 마커 (≥0.5) | {len(high_unexpectedness)}개 |\n")
        f.write(f"| 분석 마커 총 개수 | {len(frequent_markers)}개 |\n\n")

        if len(high_unexpectedness) > 0:
            f.write("**고위험 마커 (Top 10)**:\n\n")
            f.write("| 마커 | 의외성 지수 | 예상 밖 용례 / 전체 |\n")
            f.write("|------|------------|--------------------|\n")
            for _, row in high_unexpectedness.head(10).iterrows():
                f.write(f"| {row['marker']} | {row['unexpectedness_index']:.3f} | "
                       f"{row['unexpected_count']:,} / {row['total_count']:,} |\n")
            f.write("\n")

        f.write("### 2.2 맥락 다양성 (Context Diversity)\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 평균 Entropy | {df_diversity['entropy'].mean():.3f} |\n")
        f.write(f"| 고다양성 마커 (≥2.5) | {len(high_diversity)}개 |\n\n")

        if len(high_diversity) > 0:
            f.write("**고다양성 마커 (Top 10)**:\n\n")
            f.write("| 마커 | Entropy | 고유 문맥 수 | 최빈 문맥 비율 |\n")
            f.write("|------|---------|-------------|---------------|\n")
            for _, row in high_diversity.head(10).iterrows():
                f.write(f"| {row['marker']} | {row['entropy']:.3f} | "
                       f"{row['unique_contexts']} | {row['dominant_ratio']*100:.1f}% |\n")
            f.write("\n")

        f.write("### 2.3 대립 가설 테스트\n\n")
        f.write("| 테스트 | 대안 Canonicity | 사서 Canonicity | Δ | Effect Size |\n")
        f.write("|--------|----------------|----------------|----|--------------|\n")
        for _, row in df_alternative.iterrows():
            f.write(f"| {row['test_name']} | {row['alternative_canonicity']:.2f}% | "
                   f"{row['saseo_canonicity']:.2f}% | {row['delta']:.2f}% | "
                   f"{row['effect_size']:.3f} |\n")
        f.write("\n")

        f.write("## 3. 종합 판정\n\n")
        f.write(f"**편향 점수 (Bias Score)**: {bias_score:.3f}\n\n")
        f.write("| 구성 요소 | 기여도 | 값 |\n")
        f.write("|-----------|--------|-----|\n")
        f.write(f"| 의외성 지수 | 30% | {unexpectedness_penalty:.3f} |\n")
        f.write(f"| 맥락 다양성 | 20% | {diversity_penalty:.3f} |\n")
        f.write(f"| 대립 가설 | 50% | {alternative_penalty:.3f} |\n\n")

        f.write("### 해석\n\n")
        if bias_score < 0.3:
            f.write("✅ **편향 가능성 낮음** (Bias Score < 0.3)\n\n")
            f.write("- 예상 밖 용례가 적고, 사서 중심성이 강함\n")
            f.write("- 통계가 연구자 직관이 아닌 실제 데이터를 반영\n")
            f.write("- **결론**: 현재 가중치와 정규화 방법은 적절함\n")
        elif bias_score < 0.5:
            f.write("⚠️ **중간 수준 편향** (0.3 ≤ Bias Score < 0.5)\n\n")
            f.write("- 추가 검증 권장\n")
            f.write("- 일부 마커에서 예상 밖 용례 존재\n")
            f.write("- **결론**: 고위험 마커에 대한 정성적 검토 필요\n")
        else:
            f.write("❌ **편향 가능성 높음** (Bias Score ≥ 0.5)\n\n")
            f.write("- 예상 밖 용례가 많거나 사서 중심성 약함\n")
            f.write("- 가중치가 소수 용례만 과대 반영했을 가능성\n")
            f.write("- **결론**: 가중치/정규화 재검토 필요\n")

        f.write("\n## 4. 권장 사항\n\n")

        if len(high_unexpectedness) > 0:
            f.write("### 고위험 마커 정성 검토\n\n")
            f.write("다음 마커들은 예상 밖 용례가 50% 이상입니다:\n\n")
            for _, row in high_unexpectedness.head(5).iterrows():
                f.write(f"- **{row['marker']}**: 의외성 지수 {row['unexpectedness_index']:.2f}\n")
                f.write(f"  - 예상 밖 용례: {row['unexpected_count']:,} / {row['total_count']:,}\n")
                f.write(f"  - **권장**: 실제 용례를 샘플링하여 정성적으로 검토\n\n")

        if mean_effect_size < 0.8:
            f.write("### 대립 가설 추가 검증\n\n")
            f.write(f"평균 Effect Size가 {mean_effect_size:.2f}로 중간 수준입니다.\n\n")
            f.write("- 삼경, 문집 등 다른 텍스트도 유사한 중심성을 보일 가능성\n")
            f.write("- **권장**: 클러스터별 세부 분석으로 사서만의 고유성 확인\n\n")

        f.write("\n## 5. 시각화 코드\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("import matplotlib.pyplot as plt\n")
        f.write("import seaborn as sns\n\n")
        f.write("# 의외성 지수 히스토그램\n")
        f.write(f"df_unexp = pd.read_csv('{unexpectedness_csv.name}')\n")
        f.write("plt.figure(figsize=(10, 6))\n")
        f.write("plt.hist(df_unexp['unexpectedness_index'], bins=20, alpha=0.7, edgecolor='black')\n")
        f.write("plt.axvline(x=0.5, color='r', linestyle='--', label='High Risk Threshold')\n")
        f.write("plt.xlabel('Unexpectedness Index')\n")
        f.write("plt.ylabel('Frequency')\n")
        f.write("plt.title('Distribution of Unexpectedness Index')\n")
        f.write("plt.legend()\n")
        f.write("plt.show()\n\n")
        f.write("# 맥락 다양성 vs 의외성\n")
        f.write(f"df_div = pd.read_csv('{diversity_csv.name}')\n")
        f.write("df_merged = df_unexp.merge(df_div, on='marker')\n")
        f.write("plt.figure(figsize=(10, 6))\n")
        f.write("plt.scatter(df_merged['entropy'], df_merged['unexpectedness_index'], alpha=0.6)\n")
        f.write("plt.xlabel('Context Diversity (Entropy)')\n")
        f.write("plt.ylabel('Unexpectedness Index')\n")
        f.write("plt.title('Context Diversity vs Unexpectedness')\n")
        f.write("plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)\n")
        f.write("plt.axvline(x=2.5, color='r', linestyle='--', alpha=0.5)\n")
        f.write("plt.show()\n")
        f.write("```\n")

    print(f"  ✅ {report_md}")

    # JSON 결과 저장
    summary_json = args.out_dir / "summary.json"
    summary = {
        'bias_score': float(bias_score),
        'unexpectedness_penalty': float(unexpectedness_penalty),
        'diversity_penalty': float(diversity_penalty),
        'alternative_penalty': float(alternative_penalty),
        'mean_unexpectedness': float(mean_unexpectedness),
        'mean_entropy': float(mean_entropy),
        'mean_effect_size': float(mean_effect_size),
        'high_risk_markers': len(high_unexpectedness),
        'high_diversity_markers': len(high_diversity),
        'judgment': 'low_bias' if bias_score < 0.3 else ('medium_bias' if bias_score < 0.5 else 'high_bias'),
    }

    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  ✅ {summary_json}")

    print(f"\n✅ 대립 가설 검증 완료!")
    print(f"\n📋 다음 단계:")
    print(f"  1. {report_md.name} 검토")
    print(f"  2. 고위험 마커 정성 분석")
    print(f"  3. Bias Score를 BIAS_VALIDATION.md에 통합")

    return 0


if __name__ == "__main__":
    exit(main())
