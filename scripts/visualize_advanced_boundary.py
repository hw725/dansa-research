#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""고급 클러스터 시각화 스크립트 (테두리 및 추세선 포함)

- 개별 데이터 포인트 시각화 (샘플링)
- 클러스터별 경계(Convex Hull) 표시
- 권위성(Canonicity) 기반 추세 분석
- 인터랙티브 Plotly 시각화
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_data(csv_path: Path, npy_path: Optional[Path]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    embeddings = None
    if npy_path and npy_path.exists():
        embeddings = np.load(npy_path)
    return df, embeddings

def get_convex_hull(points: np.ndarray) -> np.ndarray:
    """포인트들의 Convex Hull 좌표 반환"""
    if len(points) < 3:
        return np.array([])
    try:
        hull = ConvexHull(points)
        # Hull 점들을 순서대로 반환 (폐곡선)
        hull_points = points[hull.vertices]
        return np.vstack([hull_points, hull_points[0]])
    except:
        return np.array([])

def main():
    parser = argparse.ArgumentParser(description="고급 클러스터 시각화")
    parser.add_argument("--csv", type=Path, required=True, help="클러스터링 결과 CSV")
    parser.add_argument("--npy", type=Path, help="임베딩 NPY 파일 (선택)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000, help="시각화할 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] 데이터 로딩...")
    df, X = load_data(args.csv, args.npy)
    
    # 샘플링
    if len(df) > args.sample_size:
        sample_df = df.sample(n=args.sample_size, random_state=args.seed)
        if X is not None:
            X_sample = X[sample_df.index]
    else:
        sample_df = df
        X_sample = X
        
    print(f"[2/4] 차원 축소 ({args.method})...")
    if X_sample is not None:
        if args.method == "tsne":
            # PCA로 사전 압축 후 t-SNE 실행 (속도 및 안정성)
            X_pca = PCA(n_components=min(50, X_sample.shape[1]), random_state=args.seed).fit_transform(X_sample)
            coords = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto').fit_transform(X_pca)
        else:
            coords = PCA(n_components=2, random_state=args.seed).fit_transform(X_sample)
        
        sample_df['x'] = coords[:, 0]
        sample_df['y'] = coords[:, 1]
    else:
        # 이미 x, y가 CSV에 있는 경우 (기존 시각화 결과 재활용)
        if 'x' not in sample_df.columns or 'y' not in sample_df.columns:
            raise ValueError("임베딩(NPY)이 없으면 CSV에 'x', 'y' 좌표가 포함되어 있어야 합니다.")

    print(f"[3/4] 시각화 생성...")
    
    # 1. 기본 산점도
    fig = px.scatter(
        sample_df,
        x='x', y='y',
        color='cluster_id',
        hover_data=['book_name', 'src_left', 'src_right'],
        opacity=0.6,
        size_max=10,
        title=f"Cluster Distribution with Boundaries ({args.method.upper()})"
    )
    
    # 2. 클러스터 테두리 (Convex Hull) 추가
    cluster_ids = sample_df['cluster_id'].unique()
    colors = px.colors.qualitative.Plotly
    
    for i, cid in enumerate(sorted(cluster_ids)):
        c_pnts = sample_df[sample_df['cluster_id'] == cid][['x', 'y']].values
        hull_pnts = get_convex_hull(c_pnts)
        
        if len(hull_pnts) > 0:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=hull_pnts[:, 0],
                y=hull_pnts[:, 1],
                mode='lines',
                line=dict(color=color, width=2),
                name=f"p{cid} Boundary",
                hoverinfo='skip',
                showlegend=False
            ))
            
            # 클러스터 중심에 라벨 추가
            center = c_pnts.mean(axis=0)
            fig.add_annotation(
                x=center[0], y=center[1],
                text=f"p{cid}",
                showarrow=False,
                font=dict(color="black", size=14, family="Arial Black"),
                bgcolor="white", opacity=0.8
            )

    # 3. 추세선 (Canonicity 기반)
    # 클러스터별 평균 좌표와 사서 비율(Canonicity) 계산
    # 사서 도서 목록
    CANON_BOOKS = ["논어", "맹자", "대학", "중용"]
    sample_df['is_canon'] = sample_df['book_name'].apply(lambda x: any(c in str(x) for c in CANON_BOOKS))
    
    cluster_stats = sample_df.groupby('cluster_id').agg({
        'x': 'mean',
        'y': 'mean',
        'is_canon': 'mean'
    }).reset_index()
    
    # 권위성(Canonicity)에 따른 색상 그라데이션이 있는 추세선
    cluster_stats = cluster_stats.sort_values('is_canon')
    
    if len(cluster_stats) > 1:
        fig.add_trace(go.Scatter(
            x=cluster_stats['x'],
            y=cluster_stats['y'],
            mode='lines+markers',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=3, dash='dash'),
            marker=dict(
                size=12,
                color=cluster_stats['is_canon'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Canonicity")
            ),
            name="Canonicity Trend",
            hovertext=[f"p{int(c)}: {v*100:.1f}%" for c, v in zip(cluster_stats['cluster_id'], cluster_stats['is_canon'])]
        ))

    # 레이아웃 정리
    fig.update_layout(
        template="plotly_white",
        width=1200, height=800,
        legend_title="Cluster ID",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    print(f"[4/4] 결과 저장...")
    out_html = args.out_dir / "advanced_cluster_viz.html"
    fig.write_html(str(out_html))
    
    # 클러스터별 요약 CSV 저장
    cluster_stats.to_csv(args.out_dir / "cluster_summary_stats.csv", index=False)
    
    print(f"완료: {out_html}")

if __name__ == "__main__":
    main()
