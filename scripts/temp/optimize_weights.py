# -*- coding: utf-8 -*-
"""
최적 가중치 산출 (빠른 버전 - 샘플링 적용)
Docker 환경에서 실행
"""
import pandas as pd
import numpy as np
from itertools import product
import json

# 데이터 로드
df = pd.read_csv('/workspace/hyeonto/reports/recluster_k16_child/reclustered.csv')
print(f"데이터 로드 완료: {len(df)} rows")

# 장르 분류
def classify_genre(book):
    book = str(book)
    if any(x in book for x in ['논어', '맹자', '대학', '중용']):
        return '經_사서'
    elif any(x in book for x in ['시경', '서경', '주역']):
        return '經_삼경'
    elif '춘추좌씨전' in book:
        return '經_춘추'
    elif '예기' in book:
        return '經_예기'
    elif '자치통감' in book:
        return '史'
    elif '당송팔대가' in book:
        return '集_산문'
    elif '당시삼백수' in book:
        return '集_시가'
    return '기타'

df['genre'] = df['book_name'].apply(classify_genre)
df['parent_cluster_id'] = 'p' + df['parent_cluster_id'].astype(str)

# 마커 추출 (벡터화)
DEFINITIVE_MARKERS = ['라', '는', '요', '니', '니라']
NARRATIVE_MARKERS = ['하니', '하야', '하여', '하고', '어늘']

# 마커 존재 여부 사전 계산 (속도 최적화)
df['text'] = df['src_left'].fillna('') + df['src_right'].fillna('')

for marker in DEFINITIVE_MARKERS:
    df[f'has_def_{marker}'] = df['text'].str.contains(marker, regex=False).astype(int)

for marker in NARRATIVE_MARKERS:
    df[f'has_nar_{marker}'] = df['text'].str.contains(marker, regex=False).astype(int)

# 총 정의형/서사형 마커 수
def_cols = [f'has_def_{m}' for m in DEFINITIVE_MARKERS]
nar_cols = [f'has_nar_{m}' for m in NARRATIVE_MARKERS]
df['def_count'] = df[def_cols].sum(axis=1)
df['nar_count'] = df[nar_cols].sum(axis=1)

print("마커 사전 계산 완료")

def compute_score_fast(weights, df_data):
    """빠른 점수 계산 (벡터화)"""
    w_saseo, w_samgyeong, w_other = weights
    
    # 가중치 매핑
    weight_map = {
        '經_사서': w_saseo,
        '經_삼경': w_samgyeong,
        '經_춘추': w_other,
        '經_예기': w_other,
        '史': 1.0,
        '集_산문': 1.0,
        '集_시가': 1.0,
        '기타': 1.0,
    }
    df_data = df_data.copy()
    df_data['weight'] = df_data['genre'].map(weight_map)
    
    # 1. 사서 클러스터 순수도
    p6_data = df_data[df_data['parent_cluster_id'] == 'p6']
    saseo_purity = (p6_data['genre'] == '經_사서').mean() if len(p6_data) > 0 else 0
    
    # 2. 클러스터별 가중 정의형 비율
    df_data['w_def'] = df_data['def_count'] * df_data['weight']
    df_data['w_nar'] = df_data['nar_count'] * df_data['weight']
    
    cluster_stats = df_data.groupby('parent_cluster_id').agg({
        'w_def': 'sum',
        'w_nar': 'sum',
    })
    cluster_stats['total'] = cluster_stats['w_def'] + cluster_stats['w_nar']
    cluster_stats['def_ratio'] = cluster_stats['w_def'] / cluster_stats['total'].replace(0, 1)
    
    # 정의-서사 분리도 (표준편차)
    separation = cluster_stats['def_ratio'].std()
    
    # 3. p6의 정의형 우세도
    p6_def_ratio = cluster_stats.loc['p6', 'def_ratio'] if 'p6' in cluster_stats.index else 0.5
    
    # 4. 사서 vs 비사서 마커 차별화
    saseo_def = df_data[df_data['genre'] == '經_사서']['def_count'].mean()
    non_saseo_def = df_data[df_data['genre'] != '經_사서']['def_count'].mean()
    marker_diff = (saseo_def - non_saseo_def + 1) / 2 if saseo_def and non_saseo_def else 0.5
    
    # 복합 점수
    score = (
        0.30 * saseo_purity +
        0.25 * separation +
        0.20 * marker_diff +
        0.25 * p6_def_ratio
    )
    
    return score, {
        'saseo_purity': saseo_purity,
        'separation': separation,
        'marker_diff': marker_diff,
        'p6_def_ratio': p6_def_ratio,
    }

# ===== 그리드 서치 =====
print('\n' + '=' * 60)
print('최적 가중치 탐색 (Grid Search - 빠른 버전)')
print('=' * 60)

# 더 세밀한 탐색 범위
saseo_range = np.arange(1.0, 10.1, 0.5)
samgyeong_range = np.arange(1.0, 5.1, 0.5)
other_range = np.arange(1.0, 3.1, 0.25)

results = []
best_score = 0
best_weights = None

for w_s in saseo_range:
    for w_g in samgyeong_range:
        for w_o in other_range:
            if w_s < w_g or w_g < w_o:
                continue
            
            score, details = compute_score_fast((w_s, w_g, w_o), df)
            results.append({
                'saseo': w_s,
                'samgyeong': w_g,
                'other': w_o,
                'composite_score': score,
                **details
            })
            
            if score > best_score:
                best_score = score
                best_weights = (w_s, w_g, w_o)

print(f"총 {len(results)}개 조합 탐색 완료")

# 결과 정렬
results_df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
print("\n=== 상위 10개 조합 ===")
print(results_df.head(10).to_string(index=False))

# 최적 가중치 상세 출력
print("\n" + "=" * 60)
print("🏆 최적 가중치 발견")
print("=" * 60)
print(f"  사서: {best_weights[0]:.1f}x")
print(f"  삼경: {best_weights[1]:.1f}x")
print(f"  기타경전: {best_weights[2]:.2f}x")
print(f"  복합 점수: {best_score:.4f}")

_, best_details = compute_score_fast(best_weights, df)
print(f"  - 사서 클러스터 순수도: {best_details['saseo_purity']:.4f}")
print(f"  - 정의-서사 분리도: {best_details['separation']:.4f}")
print(f"  - 마커 차별화 지수: {best_details['marker_diff']:.4f}")
print(f"  - p6 정의형 우세도: {best_details['p6_def_ratio']:.4f}")

# 저장
results_df.to_csv('/workspace/hyeonto/reports/weight_sensitivity/optimization_results.csv', 
                  index=False, encoding='utf-8')

optimal = {
    'optimal_weights': {
        'saseo': float(best_weights[0]),
        'samgyeong': float(best_weights[1]),
        'other_gyeong': float(best_weights[2]),
    },
    'composite_score': float(best_score),
    'details': {k: float(v) for k, v in best_details.items()},
    'analysis_date': '2026-01-08',
    'optimization_criteria': [
        'saseo_purity (30%): 사서 클러스터(p6) 내 사서 비중',
        'separation (25%): 클러스터 간 정의-서사 분리도',
        'marker_diff (20%): 사서/비사서 마커 차별화',
        'p6_def_ratio (25%): p6 클러스터의 정의형 마커 우세도',
    ]
}
with open('/workspace/hyeonto/reports/weight_sensitivity/optimal_weights.json', 'w', encoding='utf-8') as f:
    json.dump(optimal, f, indent=2, ensure_ascii=False)

# 기존 시나리오와 비교
print("\n" + "=" * 60)
print("📊 기존 시나리오 vs 최적 가중치 비교")
print("=" * 60)

scenarios = [
    ('uniform', (1.0, 1.0, 1.0)),
    ('weak', (2.0, 1.5, 1.2)),
    ('moderate', (3.0, 2.0, 1.5)),
    ('strong (현재)', (5.0, 3.0, 2.0)),
    ('OPTIMAL', best_weights),
]

comparison = []
for name, w in scenarios:
    score, details = compute_score_fast(w, df)
    comparison.append({
        'scenario': name,
        'saseo': w[0],
        'samgyeong': w[1],
        'other': w[2],
        'score': round(score, 4),
        **{k: round(v, 4) for k, v in details.items()}
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
comp_df.to_csv('/workspace/hyeonto/reports/weight_sensitivity/scenario_comparison.csv',
               index=False, encoding='utf-8')

print("\n✅ 저장 완료")
