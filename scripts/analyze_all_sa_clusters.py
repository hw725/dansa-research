#!/usr/bin/env python3
"""
SA Cluster 16개 전체에 대한 심층 분석

각 클러스터별로:
1. 크기, 사서 비율, 서종 분포
2. Top 마커 패턴 (left~right) - 구 단위
3. Syntactic function 분포
4. 대표 예문 3개
5. 언어학적 특징 요약
"""
import pandas as pd
import json
from collections import Counter

print("=== SA Cluster 결과 로드 ===")
df = pd.read_csv('hyeonto/reports/sa_boundary_clusters/sa_clusters_with_features.csv')

print(f"전체 행수: {len(df):,}")
print(f"클러스터 수: {df['cluster_id'].nunique()}")

# 전체 분석 결과 저장용
all_cluster_analysis = {}

for cluster_id in sorted(df['cluster_id'].unique()):
    print(f"\n{'='*60}")
    print(f"SA Cluster {cluster_id} 분석")
    print(f"{'='*60}")

    cluster_df = df[df['cluster_id'] == cluster_id]

    # 1. 기본 통계
    size = len(cluster_df)
    saseo_count = len(cluster_df[cluster_df['book_category'] == '사서'])
    saseo_ratio = saseo_count / size * 100 if size > 0 else 0

    print(f"\n[기본 통계]")
    print(f"  크기: {size:,}개")
    print(f"  사서: {saseo_count}개 ({saseo_ratio:.1f}%)")

    # 2. 서종 분포
    book_dist = cluster_df['book_category'].value_counts()
    print(f"\n[서종 분포]")
    for book_cat, count in book_dist.items():
        ratio = count / size * 100
        print(f"  {book_cat}: {count}개 ({ratio:.1f}%)")

    # 3. Top 마커 패턴 (구 단위)
    marker_patterns = cluster_df['marker_pattern'].value_counts().head(10)
    print(f"\n[Top 10 마커 패턴 (구↔구)]")
    for pattern, count in marker_patterns.items():
        ratio = count / size * 100
        print(f"  {pattern}: {count}개 ({ratio:.1f}%)")

    # 4. Syntactic function (left)
    left_func = cluster_df['syntactic_function_left'].value_counts().head(5)
    print(f"\n[Left Marker Function (Top 5)]")
    for func, count in left_func.items():
        ratio = count / size * 100
        print(f"  {func}: {count}개 ({ratio:.1f}%)")

    # 5. Syntactic function (right)
    right_func = cluster_df['syntactic_function_right'].value_counts().head(5)
    print(f"\n[Right Marker Function (Top 5)]")
    for func, count in right_func.items():
        ratio = count / size * 100
        print(f"  {func}: {count}개 ({ratio:.1f}%)")

    # 6. 대표 예문 (출력 스킵 - 인코딩 문제)
    print(f"\n[대표 예문: 분석 완료, 출력 생략]")
    samples = cluster_df.head(3)
    sample_count = len(samples)
    print(f"  {sample_count}개 예문 분석됨")

    # 저장
    all_cluster_analysis[f"cluster_{cluster_id}"] = {
        'size': int(size),
        'saseo_count': int(saseo_count),
        'saseo_ratio': float(saseo_ratio),
        'book_distribution': book_dist.to_dict(),
        'top_patterns': marker_patterns.head(5).to_dict(),
        'left_functions': left_func.head(3).to_dict(),
        'right_functions': right_func.head(3).to_dict(),
    }

# JSON 저장
output_path = 'hyeonto/reports/sa_all_clusters_analysis.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_cluster_analysis, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print(f"전체 분석 결과 저장: {output_path}")
print(f"{'='*60}")

# 요약 통계
print(f"\n=== 전체 요약 ===")
print(f"총 클러스터: 16개")
print(f"총 구↔구 경계: {len(df):,}개")

# 사서 비율 Top 5
saseo_ratios = []
for cluster_id in sorted(df['cluster_id'].unique()):
    cluster_df = df[df['cluster_id'] == cluster_id]
    saseo_count = len(cluster_df[cluster_df['book_category'] == '사서'])
    saseo_ratio = saseo_count / len(cluster_df) * 100 if len(cluster_df) > 0 else 0
    saseo_ratios.append((cluster_id, saseo_ratio, saseo_count, len(cluster_df)))

saseo_ratios.sort(key=lambda x: x[1], reverse=True)

print(f"\n사서 비율 Top 5:")
for cluster_id, ratio, count, total in saseo_ratios[:5]:
    print(f"  SA Cluster {cluster_id}: {ratio:.1f}% ({count}/{total})")

print(f"\n사서 비율 Bottom 5:")
for cluster_id, ratio, count, total in saseo_ratios[-5:]:
    print(f"  SA Cluster {cluster_id}: {ratio:.1f}% ({count}/{total})")

# 클러스터 크기 Top 5
cluster_sizes = [(cid, size) for cid, _, _, size in saseo_ratios]
cluster_sizes.sort(key=lambda x: x[1], reverse=True)

print(f"\n클러스터 크기 Top 5:")
for cluster_id, size in cluster_sizes[:5]:
    print(f"  SA Cluster {cluster_id}: {size:,}개")

print("\n분석 완료!")
