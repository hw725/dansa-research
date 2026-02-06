"""
'하다' 마커 장르별 분석 (기사지단 검증)
=========================================
Level 3: 기사지단(記史之斷) - 역사서 특유의 공적 기록 종결어

실행: python scripts/analyze_hada_by_genre.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

# 경로 설정 (루트 기준 상대경로)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'


def classify_genre(book_name: str) -> str:
    """Classifies books into genres based on content keywords."""
    if pd.isna(book_name):
        return '기타'

    book_name = str(book_name)

    if '자치통감' in book_name or '춘추좌씨전' in book_name:
        return '역사서'
    elif '당송팔대가' in book_name:
        return '문집'
    elif any(k in book_name for k in ['예기', '논어', '대학장구', '맹자', '중용']):
        return '경전'
    elif any(k in book_name for k in ['시경', '당시삼백수']):
        return '시'
    else:
        return '기타'


def analyze_hada_by_genre():
    """장르별 '하다' 마커 분포 분석"""
    print("=" * 60)
    print("Level 3: 기사지단(記史之斷) - '하다' 장르별 분석")
    print("=" * 60)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / 'sentence_normalized.csv', encoding='utf-8')
    print(f"총 데이터: {len(df):,}건")

    # 장르 분류
    df['genre'] = df['book'].apply(classify_genre)

    # '하다' 마커 추출
    hada_mask = df['marker_normalized'].str.endswith('하다', na=False)
    df['is_hada'] = hada_mask
    
    print(f"\n'하다' 마커 전체: {hada_mask.sum():,}건")
    
    # 장르별 집계
    print("\n" + "-" * 60)
    print("장르별 '하다' 분포")
    print("-" * 60)
    
    results = []
    for genre in ['역사서', '문집', '경전', '시', '기타']:
        genre_df = df[df['genre'] == genre]
        total = len(genre_df)
        hada_count = genre_df['is_hada'].sum()
        ratio = (hada_count / total * 100) if total > 0 else 0
        
        results.append({
            'genre': genre,
            'total': int(total),
            'hada_count': int(hada_count),
            'ratio': float(ratio)
        })
        
        print(f"{genre:8s}: {total:>8,}건 중 {hada_count:>6,}건 ({ratio:>5.2f}%)")
    
    # Chi-squared 검정 (역사서 vs 비역사서)
    history = df[df['genre'] == '역사서']
    non_history = df[df['genre'] != '역사서']
    
    table = np.array([
        [history['is_hada'].sum(), len(history) - history['is_hada'].sum()],
        [non_history['is_hada'].sum(), len(non_history) - non_history['is_hada'].sum()]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    print("\n" + "-" * 60)
    print("통계 검정: 역사서 vs 비역사서")
    print("-" * 60)
    print(f"χ² = {chi2:.2f}, p = {p_value:.2e}")
    print(f"결론: {'H₀ 기각 ✅ (역사서에서 유의미하게 높음)' if p_value < 0.05 else 'H₀ 기각 실패'}")
    
    # 결과 저장
    output = {
        'analysis': 'Level 3: 기사지단 (하다 장르별 분석)',
        'total_records': len(df),
        'total_hada': int(hada_mask.sum()),
        'by_genre': results,
        'chi2_test': {
            'history_vs_non_history': {
                'chi2': float(chi2),
                'p_value': float(p_value),
                'reject_h0': bool(p_value < 0.05)
            }
        }
    }
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    with open(RESULTS_DIR / 'hada_sentence_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {RESULTS_DIR / 'hada_sentence_analysis.json'}")
    
    return output


if __name__ == "__main__":
    analyze_hada_by_genre()