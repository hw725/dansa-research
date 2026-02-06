# -*- coding: utf-8 -*-
"""
단사(斷辭) 전수조사 재현 분석 스크립트
=====================================

이 스크립트는 익명화된 LLM 판정 결과 파일을 기반으로
통계 분석을 재현합니다.

입력 파일:
  - results/dansa_level1_judgments_anon.csv
  - results/dansa_level2_judgments_anon.csv

출력:
  - 콘솔에 통계 분석 결과 출력
  - results/dansa_reproduced_stats.json (재현된 통계)

사용법:
  python scripts/reproduce_analysis.py
"""

import pandas as pd
import json
from pathlib import Path
from scipy.stats import chi2_contingency

def analyze_level1(df: pd.DataFrame) -> dict:
    """Level 1: ~로다 vs ~라(대조군) 분석"""
    roda = df[df['marker_type'] == '로다']
    ctrl = df[df['marker_type'] == '라(대조군)']
    
    roda_pos = roda['llm_judgment'].sum()
    ctrl_pos = ctrl['llm_judgment'].sum()
    n_roda = len(roda)
    n_ctrl = len(ctrl)
    
    # 카이제곱 검정
    contingency = [
        [roda_pos, n_roda - roda_pos],
        [ctrl_pos, n_ctrl - ctrl_pos]
    ]
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        'level': 'Level 1',
        'description': '~로다 감탄/여운 전수조사',
        'n_roda': int(n_roda),
        'n_control': int(n_ctrl),
        'roda_positive': int(roda_pos),
        'control_positive': int(ctrl_pos),
        'roda_rate': float(roda_pos / n_roda * 100),
        'control_rate': float(ctrl_pos / n_ctrl * 100),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'reject_h0': bool(p_value < 0.05)
    }

def analyze_level2(df: pd.DataFrame) -> dict:
    """Level 2: ~니라 vs ~라 분석"""
    nira = df[df['marker_type'] == '니라']
    ra = df[df['marker_type'] == '라']
    
    nira_pos = nira['llm_judgment'].sum()
    ra_pos = ra['llm_judgment'].sum()
    n_nira = len(nira)
    n_ra = len(ra)
    
    # 카이제곱 검정
    contingency = [
        [nira_pos, n_nira - nira_pos],
        [ra_pos, n_ra - ra_pos]
    ]
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        'level': 'Level 2',
        'description': '~니라 vs ~라 단호한 종결 전수조사',
        'n_nira': int(n_nira),
        'n_ra': int(n_ra),
        'nira_positive': int(nira_pos),
        'ra_positive': int(ra_pos),
        'nira_rate': float(nira_pos / n_nira * 100),
        'ra_rate': float(ra_pos / n_ra * 100),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'reject_h0': bool(p_value < 0.05)
    }

def main():
    print("=" * 60)
    print("단사(斷辭) 전수조사 재현 분석")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent / 'results'
    
    # Level 1 분석
    print("\n[Level 1] 로딩 중...")
    l1_path = base_path / 'dansa_level1_judgments_anon.csv'
    if not l1_path.exists():
        print(f"  ❌ 파일 없음: {l1_path}")
        return
    
    l1_df = pd.read_csv(l1_path)
    l1_stats = analyze_level1(l1_df)
    
    print(f"\n{'='*60}")
    print(f"Level 1: {l1_stats['description']}")
    print(f"{'='*60}")
    print(f"  로다: {l1_stats['roda_positive']:,}/{l1_stats['n_roda']:,} ({l1_stats['roda_rate']:.1f}%)")
    print(f"  대조군: {l1_stats['control_positive']:,}/{l1_stats['n_control']:,} ({l1_stats['control_rate']:.1f}%)")
    print(f"  χ² = {l1_stats['chi2']:.2f}, p = {l1_stats['p_value']:.2e}")
    print(f"  귀무가설 기각: {'예' if l1_stats['reject_h0'] else '아니오'}")
    
    # Level 2 분석
    print("\n[Level 2] 로딩 중...")
    l2_path = base_path / 'dansa_level2_judgments_anon.csv'
    if not l2_path.exists():
        print(f"  ❌ 파일 없음: {l2_path}")
        return
    
    l2_df = pd.read_csv(l2_path)
    l2_stats = analyze_level2(l2_df)
    
    print(f"\n{'='*60}")
    print(f"Level 2: {l2_stats['description']}")
    print(f"{'='*60}")
    print(f"  니라: {l2_stats['nira_positive']:,}/{l2_stats['n_nira']:,} ({l2_stats['nira_rate']:.1f}%)")
    print(f"  라: {l2_stats['ra_positive']:,}/{l2_stats['n_ra']:,} ({l2_stats['ra_rate']:.1f}%)")
    print(f"  χ² = {l2_stats['chi2']:.2f}, p = {l2_stats['p_value']:.2e}")
    print(f"  귀무가설 기각: {'예' if l2_stats['reject_h0'] else '아니오'}")
    
    # 결과 저장
    output_path = base_path / 'dansa_reproduced_stats.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([l1_stats, l2_stats], f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 재현 통계 저장: {output_path}")
    print("\n" + "=" * 60)
    print("재현 분석 완료")
    print("=" * 60)

if __name__ == '__main__':
    main()
