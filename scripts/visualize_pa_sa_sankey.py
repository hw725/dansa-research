#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA-SA 간 클러스터 Sankey 다이어그램

도서(book_name)를 공통 기준으로 PA 클러스터와 SA 클러스터 간의 매핑을 시각화합니다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_cluster_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_cross_dataset_sankey(pa_df: pd.DataFrame, sa_df: pd.DataFrame, pa_k: int, sa_k: int):
    """book_name + sentence_id를 기준으로 PA-SA 클러스터 간 Sankey 데이터 생성 (최적화)"""
    
    # PA 문장 키 생성
    pa_sent_col = 'left_sentence_id' if 'left_sentence_id' in pa_df.columns else 'paragraph_id'
    pa_df = pa_df[['book_name', pa_sent_col, 'cluster_id']].copy()
    pa_df.columns = ['book_name', 'sent_id', 'pa_cluster']
    pa_df['sent_key'] = pa_df['book_name'].astype(str) + '_' + pa_df['sent_id'].astype(str)
    
    # SA 문장 키 생성
    sa_df = sa_df[['book_name', 'sentence_id', 'cluster_id']].copy()
    sa_df.columns = ['book_name', 'sent_id', 'sa_cluster']
    sa_df['sent_key'] = sa_df['book_name'].astype(str) + '_' + sa_df['sent_id'].astype(str)
    
    # 공통 키 기반 병합 (vectorized)
    merged = pd.merge(
        pa_df[['sent_key', 'pa_cluster']], 
        sa_df[['sent_key', 'sa_cluster']], 
        on='sent_key', 
        how='inner'
    )
    
    print(f"  매핑된 쌍: {len(merged):,}개")
    
    # 클러스터 쌍별 카운트
    flow_df = merged.groupby(['pa_cluster', 'sa_cluster']).size().reset_index(name='count')
    
    # 노드 정의
    pa_nodes = [f"PA_p{i}" for i in range(pa_k)]
    sa_nodes = [f"SA_p{i}" for i in range(sa_k)]
    all_nodes = pa_nodes + sa_nodes
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    
    sources = []
    targets = []
    values = []
    
    for _, row in flow_df.iterrows():
        pa_cid = int(row['pa_cluster'])
        sa_cid = int(row['sa_cluster'])
        cnt = int(row['count'])
        if pa_cid < pa_k and sa_cid < sa_k:
            sources.append(node_indices[f"PA_p{pa_cid}"])
            targets.append(node_indices[f"SA_p{sa_cid}"])
            values.append(cnt)
    
    return all_nodes, sources, targets, values


def generate_sankey_html(
    all_nodes: list,
    sources: list,
    targets: list,
    values: list,
    pa_k: int,
    sa_k: int,
    out_path: Path
):
    # 색상
    pa_colors = [f"hsl({i * 360 // pa_k}, 70%, 50%)" for i in range(pa_k)]
    sa_colors = [f"hsl({i * 360 // sa_k}, 50%, 65%)" for i in range(sa_k)]
    node_colors = pa_colors + sa_colors
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=["rgba(150,150,150,0.3)" for _ in sources]
        )
    )])
    
    fig.update_layout(
        title=f"PA(K={pa_k}) → SA(K={sa_k}) 클러스터 연결 (도서 기반)",
        font_size=12,
        height=700,
        width=1000
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def generate_report(
    pa_df: pd.DataFrame,
    sa_df: pd.DataFrame,
    pa_k: int,
    sa_k: int,
    out_path: Path
):
    lines = [
        f"# PA(K={pa_k}) ↔ SA(K={sa_k}) 클러스터 연결 분석",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 개요",
        "",
        f"- PA 데이터: {len(pa_df):,}건, {pa_k}개 클러스터",
        f"- SA 데이터: {len(sa_df):,}건, {sa_k}개 클러스터",
        f"- 공통 도서 수: {len(set(pa_df['book_name'].unique()) & set(sa_df['book_name'].unique()))}",
        "",
        "## 해석",
        "",
        "Sankey 다이어그램은 **도서(book_name)**를 공통 기준으로 사용하여,",
        "PA의 각 클러스터에 속한 데이터가 SA에서는 어떤 클러스터에 분포하는지를 보여줍니다.",
        "",
        "- 굵은 연결선: 두 클러스터가 유사한 도서 구성을 공유함",
        "- 가는 연결선: 도서 구성이 다르지만 일부 겹침이 있음",
    ]
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PA-SA 간 클러스터 Sankey")
    parser.add_argument("--pa-csv", type=Path, required=True)
    parser.add_argument("--sa-csv", type=Path, required=True)
    parser.add_argument("--pa-k", type=int, required=True)
    parser.add_argument("--sa-k", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] PA 데이터 로드: {args.pa_csv}")
    pa_df = load_cluster_data(args.pa_csv)
    
    print(f"[2/4] SA 데이터 로드: {args.sa_csv}")
    sa_df = load_cluster_data(args.sa_csv)
    
    print(f"[3/4] Sankey 데이터 생성...")
    all_nodes, sources, targets, values = build_cross_dataset_sankey(
        pa_df, sa_df, args.pa_k, args.sa_k
    )
    
    print(f"[4/4] 시각화 생성...")
    generate_sankey_html(
        all_nodes, sources, targets, values,
        args.pa_k, args.sa_k,
        args.out_dir / f"pa_k{args.pa_k}_sa_k{args.sa_k}_sankey.html"
    )
    
    generate_report(
        pa_df, sa_df, args.pa_k, args.sa_k,
        args.out_dir / f"pa_k{args.pa_k}_sa_k{args.sa_k}_sankey.md"
    )
    
    print(f"완료: {args.out_dir}")


if __name__ == "__main__":
    main()
