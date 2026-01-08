# -*- coding: utf-8 -*-
"""
클러스터별 장르 분포 및 엔트로피 분석
Docker 환경에서 실행
"""
import pandas as pd
from scipy.stats import entropy
import numpy as np

# 기존 클러스터 데이터 로드
df = pd.read_csv('/workspace/hyeonto/reports/recluster_k16_child/reclustered.csv')

# 장르 분류
def classify_genre_detail(book):
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

df['genre'] = df['book_name'].apply(classify_genre_detail)

print('=' * 60)
print('클러스터별 장르 분포 및 엔트로피 분석')
print('=' * 60)

results = []
for pid in sorted(df['parent_cluster_id'].unique(), key=lambda x: int(str(x).replace('p',''))):
    subset = df[df['parent_cluster_id'] == pid]
    genre_counts = subset['genre'].value_counts()
    probs = genre_counts.values / genre_counts.sum()
    ent = entropy(probs)
    dominant = genre_counts.index[0]
    dominant_pct = (genre_counts.iloc[0] / len(subset)) * 100
    
    results.append({
        'cluster': pid,
        'entropy': ent,
        'dominant_genre': dominant,
        'dominant_pct': dominant_pct,
        'size': len(subset)
    })
    
    print(f'{pid}: 엔트로피={ent:.3f}, 주류={dominant}({dominant_pct:.1f}%), n={len(subset)}')

print()
print('=' * 60)
print('핵심 통계')
print('=' * 60)
avg_entropy = np.mean([r['entropy'] for r in results])
print(f'평균 장르 엔트로피: {avg_entropy:.4f}')
print(f'최고 엔트로피 클러스터: {max(results, key=lambda x: x["entropy"])["cluster"]} ({max(r["entropy"] for r in results):.3f})')
print(f'최저 엔트로피 클러스터: {min(results, key=lambda x: x["entropy"])["cluster"]} ({min(r["entropy"] for r in results):.3f})')

print()
print('=' * 60)
print('장르별 클러스터 분포')
print('=' * 60)
genre_cluster = df.groupby('genre')['parent_cluster_id'].value_counts().unstack(fill_value=0)
print(genre_cluster.to_string())

# 결과 CSV 저장
result_df = pd.DataFrame(results)
result_df.to_csv('/workspace/hyeonto/reports/weight_sensitivity/cluster_entropy_detail.csv', index=False, encoding='utf-8')
print()
print('결과 저장: /workspace/hyeonto/reports/weight_sensitivity/cluster_entropy_detail.csv')
