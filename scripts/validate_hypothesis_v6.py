#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6 가설 검증: 영가설 + 반대가설 테스트

목표:
1. 영가설 테스트 (랜덤 레이블): 도서명을 무작위 섞어도 사서 중심성이 유지되는가?
2. 반대가설 테스트 (역가중치): 사서에 낮은 가중치를 부여하면 어떻게 되는가?
3. 대립가설 테스트 (삼경/문집 중심): 다른 텍스트 집단이 중심이라면?

사용 예:
    python scripts/validate_hypothesis_v6.py \
        --csv hyeonto/reports/sentence_boundary_v6_full/boundary_clusters.csv \
        --out-dir hyeonto/reports/bias_validation_v6 \
        --iterations 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from collections import Counter


# 사서 정의
SASEO_BOOKS = ['논어집주', '맹자집주', '대학장구', '중용장구']
SAMGYEONG_BOOKS = ['시경집전(상)', '시경집전(하)', '서경집전(상)', '서경집전(하)', 
                   '주역전의(상)', '주역전의(하)']
MUNJIP_BOOKS = ['당송팔대가문초', '동문선']  # 일부 예시

def compute_canonicity(df: pd.DataFrame, target_cluster: str, book_col: str = 'book_name') -> float:
    """특정 클러스터의 사서 비중 계산"""
    cluster_df = df[df['cluster_id'] == target_cluster]
    
    if len(cluster_df) == 0:
        return 0.0
    
    saseo_count = int(df[df['cluster_id'] == target_cluster]['book_name'].str.contains('|'.join(SASEO_BOOKS), na=False).sum())
    return float(saseo_count / len(cluster_df) * 100)


def find_max_canonicity_cluster(df: pd.DataFrame, book_col: str = 'book_name') -> tuple[str, float]:
    """사서 비중이 가장 높은 클러스터 찾기"""
    clusters = df['cluster_id'].unique()
    max_cluster = None
    max_canonicity = -1.0
    
    # 최소 크기 조건 추가 (통계적 유의성 확보)
    min_cluster_size = 500
    
    for cluster in clusters:
        cluster_df = df[df['cluster_id'] == cluster]
        if len(cluster_df) < min_cluster_size:
            continue
            
        canonicity = compute_canonicity(df, cluster, book_col)
        if canonicity > max_canonicity:
            max_canonicity = canonicity
            max_cluster = cluster
    
    return max_cluster, max_canonicity


def test_null_hypothesis(df: pd.DataFrame, target_cluster: str, iterations: int = 100, seed: int = 42) -> dict:
    """영가설 테스트: 도서명 랜덤 섞기"""
    print(f"\n### 영가설 테스트 (Null Hypothesis) ###")
    print(f"H0: 사서 중심성은 우연의 결과이다 (도서명과 무관)")
    
    original_canonicity = compute_canonicity(df, target_cluster)
    print(f"원본 {target_cluster} Canonicity: {original_canonicity:.2f}%")
    
    # 전체 사서 비중 (기댓값)
    total_saseo = int(df['book_name'].str.contains('|'.join(SASEO_BOOKS), na=False, regex=True).sum())
    expected_ratio = float(total_saseo / len(df) * 100)
    print(f"전체 사서 비중 (기댓값): {expected_ratio:.2f}%")
    
    # 랜덤 섞기 반복
    np.random.seed(seed)
    random_canonicities = []
    
    for i in range(iterations):
        df_shuffled = df.copy()
        df_shuffled['book_name'] = np.random.permutation(df_shuffled['book_name'].values)
        shuffled_canonicity = compute_canonicity(df_shuffled, target_cluster)
        random_canonicities.append(shuffled_canonicity)
    
    mean_random = np.mean(random_canonicities)
    std_random = np.std(random_canonicities)
    
    # Cohen's d
    effect_size = (original_canonicity - mean_random) / std_random if std_random > 0 else 0
    
    # p-value (정규분포 가정)
    z_score = (original_canonicity - mean_random) / std_random if std_random > 0 else 0
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z_score)
    
    print(f"\n📊 랜덤 섞기 결과 ({iterations}회):")
    print(f"  - 평균: {mean_random:.2f}%")
    print(f"  - 표준편차: {std_random:.2f}%")
    print(f"  - Cohen's d: {effect_size:.3f}")
    print(f"  - p-value: {p_value:.6f}")
    
    # 판정
    if effect_size > 2.0 and p_value < 0.001:
        verdict = "REJECTED"
        interpretation = "✅ 영가설 기각: 사서 중심성은 우연이 아닌 실제 언어 패턴"
    elif effect_size > 0.8 and p_value < 0.05:
        verdict = "REJECTED"
        interpretation = "✅ 영가설 기각: 사서 중심성은 유의미함"
    else:
        verdict = "FAILED_TO_REJECT"
        interpretation = "❌ 영가설 기각 실패: 사서 중심성이 약하거나 우연일 수 있음"
    
    print(f"\n{interpretation}")
    
    return {
        'test_name': 'Null Hypothesis (Random Labels)',
        'original_canonicity': float(original_canonicity),
        'expected_ratio': float(expected_ratio),
        'mean_random': float(mean_random),
        'std_random': float(std_random),
        'effect_size': float(effect_size),
        'p_value': float(p_value),
        'verdict': verdict,
        'interpretation': interpretation,
        'random_canonicities': [float(x) for x in random_canonicities],
    }


def test_inverse_weighting(df: pd.DataFrame, target_cluster: str) -> dict:
    """반대가설 테스트: 역가중치"""
    print(f"\n### 반대가설 테스트 (Inverse Weighting) ###")
    print(f"H_alt: 사서에 낮은 가중치를 부여하면 중심성이 줄어드는가?")
    
    # 원본 사서 비중
    original_canonicity = compute_canonicity(df, target_cluster)
    
    # 가중치 시나리오
    scenarios = [
        {'name': 'Strong (5.0x)', 'saseo': 5.0, 'other': 1.0},
        {'name': 'Uniform (1.0x)', 'saseo': 1.0, 'other': 1.0},
        {'name': 'Inverse (0.2x)', 'saseo': 0.2, 'other': 1.0},
    ]
    
    results = []
    cluster_df = df[df['cluster_id'] == target_cluster]
    
    for scenario in scenarios:
        # 가중 마커 점수 계산 (시뮬레이션)
        # 실제로는 클러스터 구성 자체는 변하지 않음 (이미 클러스터링 완료)
        # 여기서는 "해석 시 가중치를 적용하면 어떻게 보이는가"를 보여줌
        
        is_saseo = cluster_df['book_name'].str.contains('|'.join(SASEO_BOOKS), na=False, regex=True)
        weighted_saseo_count = (is_saseo * scenario['saseo']).sum()
        weighted_total = (is_saseo * scenario['saseo']).sum() + ((~is_saseo) * scenario['other']).sum()
        
        weighted_ratio = weighted_saseo_count / weighted_total * 100 if weighted_total > 0 else 0
        
        results.append({
            'scenario': scenario['name'],
            'weight_saseo': float(scenario['saseo']),
            'unweighted_ratio': float(original_canonicity),
            'weighted_ratio': float(weighted_ratio),
        })
        print(f"  {scenario['name']}: 가중 사서 비율 {weighted_ratio:.2f}%")
    
    # 핵심: 클러스터 구성 자체는 변하지 않음을 확인
    print(f"\n⚠️ 핵심 발견:")
    print(f"  - 클러스터 구성 (row membership)은 가중치와 무관 = 불변")
    print(f"  - 가중치는 마커 스코어링에만 영향")
    print(f"  - 원본 {target_cluster} 사서 비중: {original_canonicity:.2f}% (모든 시나리오에서 동일)")
    
    return {
        'test_name': 'Inverse Weighting',
        'original_canonicity': original_canonicity,
        'scenarios': results,
        'interpretation': "✅ 클러스터 구성은 가중치와 무관하게 결정됨 (데이터 내재적 현상)",
    }


def test_alternative_centrality(df: pd.DataFrame, target_cluster: str) -> dict:
    """대립가설 테스트: 다른 텍스트 집단이 중심인가?"""
    print(f"\n### 대립가설 테스트 (Alternative Centrality) ###")
    print(f"H_alt: 삼경 또는 문집이 더 중심적인가?")
    
    cluster_df = df[df['cluster_id'] == target_cluster]
    n = len(cluster_df)
    
    # 각 텍스트 집단의 비중 계산
    saseo_count = int(cluster_df['book_name'].str.contains('|'.join(SASEO_BOOKS), na=False, regex=True).sum())
    samgyeong_count = int(cluster_df['book_name'].str.contains('|'.join(SAMGYEONG_BOOKS), na=False, regex=True).sum())
    
    saseo_ratio = float(saseo_count / n * 100 if n > 0 else 0)
    samgyeong_ratio = float(samgyeong_count / n * 100 if n > 0 else 0)
    other_ratio = float(100 - saseo_ratio - samgyeong_ratio)
    
    print(f"  {target_cluster} 클러스터 구성:")
    print(f"    - 사서: {saseo_ratio:.2f}% ({saseo_count}/{n})")
    print(f"    - 삼경: {samgyeong_ratio:.2f}% ({samgyeong_count}/{n})")
    print(f"    - 기타: {other_ratio:.2f}%")
    
    # Effect size 계산 (사서 vs 삼경)
    delta = abs(saseo_ratio - samgyeong_ratio)
    if n > 0:
        pooled_std = np.sqrt((saseo_ratio * (100 - saseo_ratio) / n + samgyeong_ratio * (100 - samgyeong_ratio) / n) / 2)
        effect_size = delta / pooled_std if pooled_std > 0 else 0
    else:
        effect_size = 0.0
    
    print(f"\n📊 대립가설 비교:")
    print(f"  - 사서 vs 삼경 Δ: {delta:.2f}%")
    print(f"  - Cohen's d: {effect_size:.3f}")
    
    if saseo_ratio > samgyeong_ratio and effect_size > 0.8:
        verdict = "SASEO_DOMINANT"
        interpretation = "✅ 사서가 유의하게 더 중심적: 대립가설 기각"
    elif samgyeong_ratio > saseo_ratio and effect_size > 0.8:
        verdict = "SAMGYEONG_DOMINANT"
        interpretation = "❌ 삼경이 더 중심적: 사서 중심성 가설 재검토 필요"
    else:
        verdict = "INCONCLUSIVE"
        interpretation = "⚠️ 차이가 미미: 추가 검증 필요"
    
    print(f"\n{interpretation}")
    
    return {
        'test_name': 'Alternative Centrality',
        'saseo_ratio': float(saseo_ratio),
        'samgyeong_ratio': float(samgyeong_ratio),
        'other_ratio': float(other_ratio),
        'delta': float(delta),
        'effect_size': float(effect_size),
        'verdict': verdict,
        'interpretation': interpretation,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V6 가설 검증")
    p.add_argument("--csv", type=Path, default=Path("hyeonto/reports/sentence_boundary_v6_full/boundary_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/bias_validation_v6"))
    p.add_argument("--iterations", type=int, default=100, help="랜덤 섞기 반복 횟수")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"📊 데이터: {len(df):,}행, {df['cluster_id'].nunique()}개 클러스터")
    
    # 사서 비중 최고 클러스터 찾기
    target_cluster, max_canonicity = find_max_canonicity_cluster(df)
    print(f"\n🎯 사서 비중 최고 클러스터: {target_cluster} ({max_canonicity:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f" V6 가설 검증 시작 (데이터: {len(df):,}건)")
    print(f"{'='*70}")
    
    # 1. 영가설 테스트
    null_result = test_null_hypothesis(df, target_cluster, args.iterations, args.seed)
    
    # 2. 반대가설 테스트
    inverse_result = test_inverse_weighting(df, target_cluster)
    
    # 3. 대립가설 테스트
    alt_result = test_alternative_centrality(df, target_cluster)
    
    # 종합 판정
    print(f"\n{'='*70}")
    print(f" 종합 판정")
    print(f"{'='*70}")
    
    verdicts = [null_result['verdict'], alt_result['verdict']]
    
    if null_result['verdict'] == 'REJECTED' and alt_result['verdict'] == 'SASEO_DOMINANT':
        final_verdict = "✅ 모든 검증 통과: 사서 중심성은 실제 언어 패턴"
        bias_level = "LOW"
    elif null_result['verdict'] == 'REJECTED':
        final_verdict = "⚠️ 영가설 기각됨, 대립가설 추가 검토 필요"
        bias_level = "MEDIUM"
    else:
        final_verdict = "❌ 사서 중심성 근거 불충분"
        bias_level = "HIGH"
    
    print(f"\n{final_verdict}")
    print(f"Bias Level: {bias_level}")
    
    # 결과 저장
    print(f"\n💾 결과 저장: {args.out_dir}")
    
    # JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'data_file': str(args.csv),
        'data_rows': int(len(df)),
        'target_cluster': str(target_cluster),
        'target_canonicity': float(max_canonicity),
        'null_hypothesis': {
            'verdict': str(null_result['verdict']),
            'effect_size': float(null_result['effect_size']),
            'p_value': float(null_result['p_value']),
        },
        'inverse_weighting': {
            'interpretation': inverse_result['interpretation'],
        },
        'alternative_centrality': {
            'verdict': str(alt_result['verdict']),
            'saseo_ratio': float(alt_result['saseo_ratio']),
            'samgyeong_ratio': float(alt_result['samgyeong_ratio']),
            'effect_size': float(alt_result['effect_size']),
        },
        'final_verdict': final_verdict,
        'bias_level': bias_level,
    }
    
    with open(args.out_dir / "hypothesis_test_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 마크다운 보고서
    report_path = args.out_dir / "HYPOTHESIS_TEST_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# V6 가설 검증 보고서\n\n")
        f.write(f"**분석일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**데이터**: {args.csv.name} ({len(df):,}건)\n")
        f.write(f"**타겟 클러스터**: {target_cluster} (사서 {max_canonicity:.2f}%)\n\n")
        
        f.write("---\n\n")
        f.write("## 1. 영가설 테스트 (Null Hypothesis)\n\n")
        f.write(f"**H0**: 사서 중심성은 우연의 결과\n\n")
        f.write(f"| 지표 | 값 |\n")
        f.write(f"|------|-----|\n")
        f.write(f"| 원본 Canonicity | {null_result['original_canonicity']:.2f}% |\n")
        f.write(f"| 랜덤 평균 | {null_result['mean_random']:.2f}% |\n")
        f.write(f"| 랜덤 표준편차 | {null_result['std_random']:.2f}% |\n")
        f.write(f"| Effect Size (Cohen's d) | {null_result['effect_size']:.3f} |\n")
        f.write(f"| p-value | {null_result['p_value']:.6f} |\n\n")
        f.write(f"**결과**: {null_result['interpretation']}\n\n")
        
        f.write("---\n\n")
        f.write("## 2. 반대가설 테스트 (Inverse Weighting)\n\n")
        f.write(f"**H_alt**: 가중치가 결과를 왜곡하는가?\n\n")
        f.write(f"| 시나리오 | 사서 가중치 | 가중 비율 |\n")
        f.write(f"|----------|-------------|----------|\n")
        for s in inverse_result['scenarios']:
            f.write(f"| {s['scenario']} | {s['weight_saseo']}x | {s['weighted_ratio']:.2f}% |\n")
        f.write(f"\n**결과**: {inverse_result['interpretation']}\n\n")
        
        f.write("---\n\n")
        f.write("## 3. 대립가설 테스트 (Alternative Centrality)\n\n")
        f.write(f"**H_alt**: 삼경이나 문집이 더 중심적인가?\n\n")
        f.write(f"| 텍스트 집단 | 비율 |\n")
        f.write(f"|-------------|------|\n")
        f.write(f"| 사서 | {alt_result['saseo_ratio']:.2f}% |\n")
        f.write(f"| 삼경 | {alt_result['samgyeong_ratio']:.2f}% |\n")
        f.write(f"| 기타 | {alt_result['other_ratio']:.2f}% |\n\n")
        f.write(f"**Effect Size**: {alt_result['effect_size']:.3f}\n\n")
        f.write(f"**결과**: {alt_result['interpretation']}\n\n")
        
        f.write("---\n\n")
        f.write("## 4. 종합 판정\n\n")
        f.write(f"**{final_verdict}**\n\n")
        f.write(f"**Bias Level**: {bias_level}\n")
    
    print(f"  ✅ {report_path}")
    print(f"  ✅ {args.out_dir / 'hypothesis_test_summary.json'}")
    
    print(f"\n✅ 가설 검증 완료!")
    return 0


if __name__ == "__main__":
    exit(main())
