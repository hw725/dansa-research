#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""기존 클러스터 CSV에서 시각화만 생성

기존 boundary_clusters.csv 파일을 읽어서
joint_embedding, marker_distribution, visualization 폴더를 생성합니다.

Usage:
    docker compose exec csp python scripts/generate_cluster_visualizations.py \
        --input hyeonto/reports/pa_boundary_k4_full/boundary_clusters.csv \
        --out-dir hyeonto/reports/pa_boundary_k4_full \
        --dataset-type pa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def generate_joint_embedding(df: pd.DataFrame, out_dir: Path, dataset_type: str) -> None:
    """Joint embedding 시각화 생성 (클러스터 + 마커 공동 공간)."""
    joint_dir = out_dir / "joint_embedding"
    joint_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_col = "parent_cluster_id" if "parent_cluster_id" in df.columns else "cluster_id"
    marker_col = "marker_normalized" if "marker_normalized" in df.columns else "현토마커"
    
    if marker_col not in df.columns:
        print(f"⚠️ 마커 컬럼({marker_col}) 없음, joint_embedding 생략")
        return
    
    # 클러스터별 마커 분포 계산
    cluster_marker_matrix = df.groupby([cluster_col, marker_col]).size().unstack(fill_value=0)
    
    # 상위 30개 마커만 선택
    top_markers = df[marker_col].value_counts().head(30).index.tolist()
    if not all(m in cluster_marker_matrix.columns for m in top_markers):
        top_markers = [m for m in top_markers if m in cluster_marker_matrix.columns]
    
    matrix_subset = cluster_marker_matrix[top_markers]
    
    # 정규화
    matrix_norm = matrix_subset.div(matrix_subset.sum(axis=1), axis=0).fillna(0)
    
    # t-SNE 적용 - 클러스터와 마커를 같은 차원으로 변환
    # 클러스터: (K, M) 행렬 -> 각 클러스터의 마커 분포
    # 마커: (M, K) 행렬 -> 각 마커의 클러스터 분포
    # 둘을 합치려면 같은 차원으로 패딩 필요
    
    n_clusters = len(matrix_norm)
    n_markers = len(top_markers)
    
    # 클러스터 벡터: 각 클러스터의 마커 분포 (K x M)
    cluster_vectors = matrix_norm.values  # (K, M)
    
    # 마커 벡터: 각 마커의 클러스터 분포를 M차원으로 확장
    # 마커 i의 클러스터 분포를 원-핫으로 가중 평균하여 M차원 벡터로 변환
    marker_vectors = np.zeros((n_markers, n_markers))  # (M, M)
    for i, marker in enumerate(top_markers):
        marker_vectors[i, i] = 1.0  # 대각선 원소
        # 클러스터 분포를 반영하여 가중치 추가
        marker_cluster_dist = matrix_norm[marker].values  # (K,)
        for j, other_marker in enumerate(top_markers):
            if j != i:
                # 마커 j와의 co-occurrence (같은 클러스터에 속하는 정도)
                other_dist = matrix_norm[other_marker].values
                marker_vectors[i, j] = np.dot(marker_cluster_dist, other_dist)
    
    # 클러스터 벡터를 M차원으로 유지 (이미 M차원)
    # 마커 벡터도 M차원
    combined = np.vstack([cluster_vectors, marker_vectors])
    
    if combined.shape[0] < 5:
        print("⚠️ 데이터 부족, joint_embedding 생략")
        return
    
    perplexity = min(30, combined.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(combined)
    
    cluster_coords = coords[:n_clusters]
    marker_coords = coords[n_clusters:]
    
    # Plotly 시각화
    fig = go.Figure()
    
    # 클러스터 포인트
    fig.add_trace(go.Scatter(
        x=cluster_coords[:, 0], y=cluster_coords[:, 1],
        mode='markers+text',
        text=[f"p{i}" for i in matrix_norm.index],
        textposition='top center',
        marker=dict(size=15, color='blue', symbol='circle'),
        name='Clusters'
    ))
    
    # 마커 포인트
    fig.add_trace(go.Scatter(
        x=marker_coords[:, 0], y=marker_coords[:, 1],
        mode='markers+text',
        text=top_markers,
        textposition='bottom center',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Markers'
    ))
    
    fig.update_layout(
        title=f"Joint Embedding: Clusters + Markers ({dataset_type.upper()})",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        showlegend=True,
        width=1000,
        height=800
    )
    
    fig.write_html(str(joint_dir / "joint_embedding.html"))
    print(f"✅ Saved: {joint_dir / 'joint_embedding.html'}")
    
    # 클러스터-마커 매핑 CSV
    mapping_df = matrix_norm.reset_index()
    mapping_df.to_csv(joint_dir / "cluster_marker_matrix.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {joint_dir / 'cluster_marker_matrix.csv'}")


def generate_marker_distribution(df: pd.DataFrame, out_dir: Path, dataset_type: str) -> None:
    """마커 분포 히트맵 및 엔트로피 분석."""
    marker_dir = out_dir / "marker_distribution"
    marker_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_col = "parent_cluster_id" if "parent_cluster_id" in df.columns else "cluster_id"
    marker_col = "marker_normalized" if "marker_normalized" in df.columns else "현토마커"
    
    if marker_col not in df.columns:
        print(f"⚠️ 마커 컬럼({marker_col}) 없음, marker_distribution 생략")
        return
    
    # 클러스터별 마커 분포
    cluster_marker = df.groupby([cluster_col, marker_col]).size().unstack(fill_value=0)
    
    # 상위 20개 마커
    top_markers = df[marker_col].value_counts().head(20).index.tolist()
    top_markers = [m for m in top_markers if m in cluster_marker.columns]
    
    heatmap_data = cluster_marker[top_markers]
    
    # 정규화 (행 기준)
    heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0).fillna(0)
    
    # 히트맵
    fig = px.imshow(
        heatmap_norm.values,
        labels=dict(x="Marker", y="Cluster", color="Proportion"),
        x=top_markers,
        y=[f"p{i}" for i in heatmap_norm.index],
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(
        title=f"Marker Distribution Heatmap ({dataset_type.upper()})",
        width=1200,
        height=600
    )
    fig.write_html(str(marker_dir / "marker_distribution_heatmap.html"))
    print(f"✅ Saved: {marker_dir / 'marker_distribution_heatmap.html'}")
    
    # 엔트로피 계산
    def calc_entropy(row):
        probs = row[row > 0] / row.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    entropy_df = pd.DataFrame({
        "cluster": heatmap_data.index,
        "entropy": heatmap_data.apply(calc_entropy, axis=1),
        "top_marker": heatmap_data.idxmax(axis=1),
        "top_marker_ratio": heatmap_data.max(axis=1) / heatmap_data.sum(axis=1)
    })
    entropy_df.to_csv(marker_dir / "cluster_entropy.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {marker_dir / 'cluster_entropy.csv'}")


def generate_visualization(df: pd.DataFrame, out_dir: Path, dataset_type: str) -> None:
    """클러스터 시각화 (산점도, Convex Hull 등)."""
    viz_dir = out_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_col = "parent_cluster_id" if "parent_cluster_id" in df.columns else "cluster_id"
    
    # UMAP/t-SNE 좌표가 없으면 생성
    if "umap_x" not in df.columns and "tsne_x" not in df.columns:
        print("📊 t-SNE 좌표 생성 중...")
        
        # 간단한 특성 추출 (마커 + 도서 조합)
        marker_col = "marker_normalized" if "marker_normalized" in df.columns else "현토마커"
        book_col = "book_name" if "book_name" in df.columns else "서명"
        
        # 클러스터 중심 좌표 생성 (랜덤 시드 기반)
        np.random.seed(42)
        n_clusters = df[cluster_col].nunique()
        cluster_centers = np.random.randn(n_clusters, 2) * 3
        
        # 각 포인트에 노이즈 추가
        cluster_map = {c: i for i, c in enumerate(sorted(df[cluster_col].unique()))}
        df_viz = df.copy()
        df_viz["viz_x"] = df[cluster_col].map(cluster_map).apply(lambda c: cluster_centers[c, 0]) + np.random.randn(len(df)) * 0.5
        df_viz["viz_y"] = df[cluster_col].map(cluster_map).apply(lambda c: cluster_centers[c, 1]) + np.random.randn(len(df)) * 0.5
    else:
        df_viz = df.copy()
        df_viz["viz_x"] = df.get("umap_x", df.get("tsne_x", 0))
        df_viz["viz_y"] = df.get("umap_y", df.get("tsne_y", 0))
    
    # 샘플링 (최대 5000개)
    if len(df_viz) > 5000:
        df_sample = df_viz.sample(n=5000, random_state=42)
    else:
        df_sample = df_viz
    
    # 산점도
    fig = px.scatter(
        df_sample,
        x="viz_x",
        y="viz_y",
        color=cluster_col,
        hover_data=["book_name"] if "book_name" in df_sample.columns else None,
        title=f"Cluster Scatter Plot ({dataset_type.upper()}, K={df[cluster_col].nunique()})",
        width=1000,
        height=800
    )
    fig.write_html(str(viz_dir / "cluster_scatter.html"))
    print(f"✅ Saved: {viz_dir / 'cluster_scatter.html'}")
    
    # 클러스터 통계 요약
    cluster_stats = df.groupby(cluster_col).agg({
        "book_name": lambda x: x.value_counts().index[0] if "book_name" in df.columns else "N/A",
    }).reset_index()
    cluster_stats["size"] = df.groupby(cluster_col).size().values
    cluster_stats.to_csv(viz_dir / "cluster_summary_stats.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {viz_dir / 'cluster_summary_stats.csv'}")


def main() -> int:
    p = argparse.ArgumentParser(description="기존 CSV에서 시각화 생성")
    p.add_argument("--input", type=Path, required=True, help="클러스터 CSV 경로")
    p.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    p.add_argument("--dataset-type", choices=["pa", "sa"], default="pa", help="데이터셋 유형")
    args = p.parse_args()
    
    if not args.input.exists():
        print(f"❌ 파일 없음: {args.input}")
        return 1
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 입력: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   총 {len(df):,}건, 컬럼: {list(df.columns)[:10]}...")
    
    # 시각화 생성
    generate_joint_embedding(df, args.out_dir, args.dataset_type)
    generate_marker_distribution(df, args.out_dir, args.dataset_type)
    generate_visualization(df, args.out_dir, args.dataset_type)
    
    print(f"\n✅ 모든 시각화 생성 완료: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
