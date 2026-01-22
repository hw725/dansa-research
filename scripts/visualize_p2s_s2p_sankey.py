#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2S-S2P 간 클러스터 Sankey 다이어그램

도서(book_name)를 공통 기준으로 P2S 클러스터와 S2P 클러스터 간의 매핑을 시각화합니다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_cluster_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_cross_dataset_sankey(p2s_df: pd.DataFrame, s2p_df: pd.DataFrame, p2s_k: int, s2p_k: int):
    """book_name + sentence_id를 기준으로 P2S-S2P 클러스터 간 Sankey 데이터 생성 (최적화)"""
    
    # P2S 문장 키 생성
    p2s_sent_col = 'left_sentence_id' if 'left_sentence_id' in p2s_df.columns else 'paragraph_id'
    p2s_df = p2s_df[['book_name', p2s_sent_col, 'cluster_id']].copy()
    p2s_df.columns = ['book_name', 'sent_id', 'p2s_cluster']
    p2s_df['sent_key'] = p2s_df['book_name'].astype(str) + '_' + p2s_df['sent_id'].astype(str)
    
    # S2P 문장 키 생성
    s2p_df = s2p_df[['book_name', 'sentence_id', 'cluster_id']].copy()
    s2p_df.columns = ['book_name', 'sent_id', 's2p_cluster']
    s2p_df['sent_key'] = s2p_df['book_name'].astype(str) + '_' + s2p_df['sent_id'].astype(str)
    
    # 공통 키 기반 병합 (vectorized)
    merged = pd.merge(
        p2s_df[['sent_key', 'p2s_cluster']], 
        s2p_df[['sent_key', 's2p_cluster']], 
        on='sent_key', 
        how='inner'
    )
    
    print(f"  매핑된 쌍: {len(merged):,}개")
    
    # 클러스터 쌍별 카운트
    flow_df = merged.groupby(['p2s_cluster', 's2p_cluster']).size().reset_index(name='count')
    
    # 노드 정의
    p2s_nodes = [f"P2S_p{i}" for i in range(p2s_k)]
    s2p_nodes = [f"S2P_p{i}" for i in range(s2p_k)]
    all_nodes = p2s_nodes + s2p_nodes
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    
    sources = []
    targets = []
    values = []
    
    for _, row in flow_df.iterrows():
        p2s_cid = int(row['p2s_cluster'])
        s2p_cid = int(row['s2p_cluster'])
        cnt = int(row['count'])
        if p2s_cid < p2s_k and s2p_cid < s2p_k:
            sources.append(node_indices[f"P2S_p{p2s_cid}"])
            targets.append(node_indices[f"S2P_p{s2p_cid}"])
            values.append(cnt)
    
    return all_nodes, sources, targets, values


def generate_sankey_html(
    all_nodes: list,
    sources: list,
    targets: list,
    values: list,
    p2s_k: int,
    s2p_k: int,
    out_path: Path
):
    # 색상
    p2s_colors = [f"hsl({i * 360 // p2s_k}, 70%, 50%)" for i in range(p2s_k)]
    s2p_colors = [f"hsl({i * 360 // s2p_k}, 50%, 65%)" for i in range(s2p_k)]
    node_colors = p2s_colors + s2p_colors
    
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
        title=f"P2S(K={p2s_k}) → S2P(K={s2p_k}) 클러스터 연결 (도서 기반)",
        font_size=12,
        height=700,
        width=1000
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def generate_report(
    p2s_df: pd.DataFrame,
    s2p_df: pd.DataFrame,
    p2s_k: int,
    s2p_k: int,
    out_path: Path
):
    lines = [
        f"# P2S(K={p2s_k}) ↔ S2P(K={s2p_k}) 클러스터 연결 분석",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 개요",
        "",
        f"- P2S 데이터: {len(p2s_df):,}건, {p2s_k}개 클러스터",
        f"- S2P 데이터: {len(s2p_df):,}건, {s2p_k}개 클러스터",
        f"- 공통 도서 수: {len(set(p2s_df['book_name'].unique()) & set(s2p_df['book_name'].unique()))}",
        "",
        "## 해석",
        "",
        "Sankey 다이어그램은 **도서(book_name)**를 공통 기준으로 사용하여,",
        "P2S의 각 클러스터에 속한 데이터가 S2P에서는 어떤 클러스터에 분포하는지를 보여줍니다.",
        "",
        "- 굵은 연결선: 두 클러스터가 유사한 도서 구성을 공유함",
        "- 가는 연결선: 도서 구성이 다르지만 일부 겹침이 있음",
    ]
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="P2S-S2P 간 클러스터 Sankey")
    parser.add_argument("--p2s-csv", type=Path, required=True)
    parser.add_argument("--s2p-csv", type=Path, required=True)
    parser.add_argument("--p2s-k", type=int, required=True)
    parser.add_argument("--s2p-k", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] P2S 데이터 로드: {args.p2s_csv}")
    p2s_df = load_cluster_data(args.p2s_csv)
    
    print(f"[2/4] S2P 데이터 로드: {args.s2p_csv}")
    s2p_df = load_cluster_data(args.s2p_csv)
    
    print(f"[3/4] Sankey 데이터 생성...")
    all_nodes, sources, targets, values = build_cross_dataset_sankey(
        p2s_df, s2p_df, args.p2s_k, args.s2p_k
    )
    
    print(f"[4/4] 시각화 생성...")
    generate_sankey_html(
        all_nodes, sources, targets, values,
        args.p2s_k, args.s2p_k,
        args.out_dir / f"p2s_k{args.p2s_k}_s2p_k{args.s2p_k}_sankey.html"
    )
    
    generate_report(
        p2s_df, s2p_df, args.p2s_k, args.s2p_k,
        args.out_dir / f"p2s_k{args.p2s_k}_s2p_k{args.s2p_k}_sankey.md"
    )
    
    print(f"완료: {args.out_dir}")


if __name__ == "__main__":
    main()

