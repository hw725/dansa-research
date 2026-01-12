#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA 클러스터와 SA 클러스터의 교차 분석 및 Sankey Diagram 생성 (V6)

목적: 문장 단위(PA) 클러스터가 구 단위(SA) 클러스터로 어떻게 분화되는지 시각화
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa-csv", type=Path, required=True, help="PA 클러스터 결과 (v6)")
    p.add_argument("--sa-csv", type=Path, required=True, help="SA 클러스터 결과 (v6)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🔄 PA/SA 데이터 로드 및 조인 중...")
    pa_df = pd.read_csv(args.pa_csv)
    sa_df = pd.read_csv(args.sa_csv)

    # PA의 경계 식별자: book_name, paragraph_id, left_sentence_id, right_sentence_id
    # SA의 경계 식별자: book_name, paragraph_id, sentence_id (left), left_clause_id, right_clause_id
    # PA 경계(S1/S2 사이)는 SA에서 (S1의 마지막 구 / S2의 첫 구) 사이의 경계와 일치함.
    # 하지만 데이터 구축 방식에 따라 SA가 문장 내 경계만 가질 수도 있음.
    # 여기서는 'src_left', 'src_right', 'tgt_left', 'tgt_right' 텍스트 일치를 보조 지표로 사용하거나 
    # 고유 ID가 일치하는 공통 경계만 추출함.

    # 텍스트 기반 매핑 (더 확실함)
    def normalize_text(t):
        return "".join(str(t).split())

    pa_df["match_key"] = pa_df["src_left"].apply(normalize_text) + "|" + pa_df["src_right"].apply(normalize_text)
    sa_df["match_key"] = sa_df["src_left"].apply(normalize_text) + "|" + sa_df["src_right"].apply(normalize_text)

    # 가중치 반영 (사서 5x)
    CANON = ["논어", "맹자", "대학", "중용"]
    def get_weight(book):
        if any(c in str(book) for c in CANON): return 5.0
        return 1.0

    pa_df["weight"] = pa_df["book_name"].apply(get_weight)

    # 조인
    merged = pd.merge(
        pa_df[["match_key", "cluster_id", "weight", "book_name"]],
        sa_df[["match_key", "cluster_id"]],
        on="match_key",
        suffixes=("_pa", "_sa")
    )

    print(f"✅ 매칭된 경계 수: {len(merged):,} 건")

    # 흐름 집계
    flow = merged.groupby(["cluster_id_pa", "cluster_id_sa"])["weight"].sum().reset_index()
    
    # 노드 인덱스 생성
    pa_labels = sorted(flow["cluster_id_pa"].unique())
    sa_labels = sorted(flow["cluster_id_sa"].unique())
    
    # PA 노드는 0 ~ N-1, SA 노드는 N ~ N+M-1
    node_labels = [f"PA-{l}" for l in pa_labels] + [f"SA-{l}" for l in sa_labels]
    pa_to_idx = {l: i for i, l in enumerate(pa_labels)}
    sa_to_idx = {l: i + len(pa_labels) for i, l in enumerate(sa_labels)}

    sources = flow["cluster_id_pa"].map(pa_to_idx)
    targets = flow["cluster_id_sa"].map(sa_to_idx)
    values = flow["weight"]

    # Sankey 생성
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = node_labels,
          color = "blue"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values,
          color = "rgba(200, 200, 200, 0.4)",
          hovertemplate = 'Source: %{source.label}<br />Target: %{target.label}<br />Weight: %{value:.1f}<extra></extra>'
        )
    )])

    fig.update_layout(title_text="PA → SA 클러스터 흐름 (Sankey Diagram - V6)", font_size=12, width=1200, height=800)
    
    out_html = args.out_dir / "pa_sa_sankey_v6.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    
    # 통계 CSV 저장
    flow.to_csv(args.out_dir / "pa_sa_flow_stats.csv", index=False, encoding="utf-8-sig")

    print(f"📊 Sankey 저장 완료: {out_html}")

if __name__ == "__main__":
    main()
