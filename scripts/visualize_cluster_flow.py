#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K값 분화 시각화 (Sankey Diagram)

K=4 클러스터가 K=14로 어떻게 분화되는지 시각화합니다.
동일한 임베딩에서 두 K값으로 클러스터링한 결과를 비교합니다.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans


def load_embeddings(npy_path: Path) -> np.ndarray:
    X = np.load(npy_path)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def cluster(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init="auto")
    return km.fit_predict(X)


def build_sankey_data(labels_small: np.ndarray, labels_large: np.ndarray, k_small: int, k_large: int):
    """Sankey 다이어그램용 데이터 생성"""
    
    # 노드: K_small 클러스터들 (0~k_small-1) + K_large 클러스터들 (k_small ~ k_small+k_large-1)
    node_labels = [f"K{k_small}_p{i}" for i in range(k_small)] + [f"K{k_large}_p{i}" for i in range(k_large)]
    
    # 링크 카운트
    flow = defaultdict(int)
    for s, l in zip(labels_small, labels_large):
        flow[(int(s), int(l) + k_small)] += 1
    
    sources = []
    targets = []
    values = []
    
    for (src, tgt), cnt in flow.items():
        sources.append(src)
        targets.append(tgt)
        values.append(cnt)
    
    return node_labels, sources, targets, values


def generate_sankey_html(
    node_labels: list,
    sources: list,
    targets: list,
    values: list,
    k_small: int,
    k_large: int,
    out_path: Path
):
    """Sankey 다이어그램 HTML 생성"""
    
    # 색상 팔레트
    colors_small = [f"hsl({i * 360 // k_small}, 70%, 50%)" for i in range(k_small)]
    colors_large = [f"hsl({i * 360 // k_large}, 50%, 65%)" for i in range(k_large)]
    node_colors = colors_small + colors_large
    
    # 링크 색상 (소스 노드 색상 기반)
    link_colors = [f"rgba{tuple(list(int(colors_small[s].split('(')[1].split(',')[0]) for _ in range(3)) + [0.4])}" 
                   if s < k_small else "rgba(180,180,180,0.4)" 
                   for s in sources]
    # 간단하게 회색 반투명으로
    link_colors = ["rgba(150,150,150,0.4)" for _ in sources]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=f"클러스터 분화 시각화: K={k_small} → K={k_large}",
        font_size=12,
        height=700,
        width=1000
    )
    
    fig.write_html(str(out_path))


def generate_comparison_report(
    labels_small: np.ndarray,
    labels_large: np.ndarray,
    k_small: int,
    k_large: int,
    out_path: Path
):
    """분화 비교 리포트 생성"""
    
    lines = [
        f"# 클러스터 분화 분석: K={k_small} → K={k_large}",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**데이터 수**: {len(labels_small)}건",
        "",
        "---",
        "",
        "## 분화 매트릭스",
        "",
        f"각 K={k_small} 클러스터가 K={k_large}로 어떻게 분화되는지 보여줍니다.",
        "",
    ]
    
    for s in range(k_small):
        mask = labels_small == s
        large_dist = Counter(labels_large[mask])
        total = sum(large_dist.values())
        
        lines.append(f"### K{k_small}_p{s} (n={total:,})")
        lines.append("")
        lines.append(f"| K{k_large} 클러스터 | 건수 | 비율 |")
        lines.append("|:---:|---:|---:|")
        
        for l, cnt in large_dist.most_common():
            pct = cnt / total * 100
            lines.append(f"| p{l} | {cnt:,} | {pct:.1f}% |")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## 해석",
        "",
        f"K={k_small}의 각 클러스터는 K={k_large}에서 여러 하위 클러스터로 분화됩니다.",
        "분화 비율이 높을수록 해당 K값이 더 세분화된 기능 구분을 포착합니다.",
    ])
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="K값 분화 시각화")
    parser.add_argument("--npy", type=Path, required=True, help="임베딩 NPY 파일")
    parser.add_argument("--k-small", type=int, default=4, help="작은 K값")
    parser.add_argument("--k-large", type=int, default=14, help="큰 K값")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] 임베딩 로드: {args.npy}")
    X = load_embeddings(args.npy)
    print(f"  -> {len(X)}개 로드")
    
    print(f"[2/4] K={args.k_small} 클러스터링...")
    labels_small = cluster(X, args.k_small, args.seed)
    
    print(f"[3/4] K={args.k_large} 클러스터링...")
    labels_large = cluster(X, args.k_large, args.seed)
    
    print(f"[4/4] 시각화 생성...")
    node_labels, sources, targets, values = build_sankey_data(
        labels_small, labels_large, args.k_small, args.k_large
    )
    
    generate_sankey_html(
        node_labels, sources, targets, values,
        args.k_small, args.k_large,
        args.out_dir / f"cluster_flow_k{args.k_small}_to_k{args.k_large}.html"
    )
    
    generate_comparison_report(
        labels_small, labels_large,
        args.k_small, args.k_large,
        args.out_dir / f"cluster_flow_k{args.k_small}_to_k{args.k_large}.md"
    )
    
    print(f"완료: {args.out_dir}")


if __name__ == "__main__":
    main()
