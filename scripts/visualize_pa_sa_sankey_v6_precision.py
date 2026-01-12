#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA-SA 고유 ID 기반 교차 분석 및 Sankey Diagram (V6 정밀 버전)

매칭 로직: 
- SA의 구 경계 중 문장의 마지막 구 경계는 PA의 문장 경계와 일치함.
- 문장식별자(Sentence ID)를 기준으로 두 층위의 클러스터를 조인.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa-clusters", type=Path, required=True, help="pa_boundary_v6_full/boundary_clusters.csv")
    p.add_argument("--sa-clusters", type=Path, required=True, help="sa_boundary_v6_full/sa_boundary_clusters.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 V6 고유 ID 기반 조인 시작...")
    
    # 1. PA 클러스터 로드
    pa_clust = pd.read_csv(args.pa_clusters)
    # v6 CSV의 paragraph_id, left_sentence_id 사용
    # 주의: UTF-8 BOM(﻿) 처리
    pa_clust.columns = [c.lstrip('\ufeff') for c in pa_clust.columns]
    
    # 2. SA 클러스터 로드
    sa_clust = pd.read_csv(args.sa_clusters)
    sa_clust.columns = [c.lstrip('\ufeff') for c in sa_clust.columns]

    # SA 데이터에서 문장 끝(PA와 겹치는 지점)만 필터링
    # V6 SA CSV 구조: cluster_id, book_name, sentence_id, left_phrase_id, right_phrase_id...
    # 문장 끝 경계는 보통 right_phrase_id가 0이거나 (새 문장 시작), 
    # 또는 left_phrase_id가 해당 문장의 최대값일 때임.
    # 여기서는 각 문장(sentence_id)의 마지막 Phrase ID를 가진 행을 찾음.
    
    # SA에서 각 (book_name, sentence_id)별로 가장 큰 left_phrase_id를 가진 행이 문장 끝임.
    sa_clust["is_sentence_end"] = sa_clust.groupby(["book_name", "sentence_id"])["left_phrase_id"].transform(max) == sa_clust["left_phrase_id"]
    sa_ends = sa_clust[sa_clust["is_sentence_end"]].copy()

    # 조인 키 생성: book_name + sentence_id
    # PA의 left_sentence_id가 SA의 sentence_id와 매핑됨.
    pa_clust["join_key"] = pa_clust["book_name"] + "_" + pa_clust["left_sentence_id"].astype(str)
    sa_ends["join_key"] = sa_ends["book_name"] + "_" + sa_ends["sentence_id"].astype(str)

    # 3. 조인
    merged = pd.merge(
        pa_clust[["join_key", "cluster_id", "book_name"]],
        sa_ends[["join_key", "cluster_id"]],
        on="join_key",
        suffixes=("_pa", "_sa")
    )

    print(f"📊 매칭 완료: {len(merged):,} 건 (PA 전체 {len(pa_clust):,} 건 중)")
    if len(merged) == 0:
        print("❌ 매칭된 경계가 없습니다. ID 체계를 재확인하십시오.")
        return

    # 가중치 (사서 5x)
    CANON = ["논어", "맹자", "대학", "중용"]
    merged["weight"] = merged["book_name"].apply(lambda x: 5.0 if any(c in str(x) for c in CANON) else 1.0)

    # 흐름 집계
    flow = merged.groupby(["cluster_id_pa", "cluster_id_sa"])["weight"].sum().reset_index()
    
    # Sankey 시각화 (동일 로직)
    pa_labels = sorted(flow["cluster_id_pa"].unique())
    sa_labels = sorted(flow["cluster_id_sa"].unique())
    node_labels = [f"PA-{l}" for l in pa_labels] + [f"SA-{l}" for l in sa_labels]
    pa_to_idx = {l: i for i, l in enumerate(pa_labels)}
    sa_to_idx = {l: i + len(pa_labels) for i, l in enumerate(sa_labels)}

    fig = go.Figure(data=[go.Sankey(
        node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = node_labels, color = "royalblue"),
        link = dict(
          source = flow["cluster_id_pa"].map(pa_to_idx),
          target = flow["cluster_id_sa"].map(sa_to_idx),
          value = flow["weight"],
          color = "rgba(200, 200, 200, 0.4)"
        )
    )])

    fig.update_layout(title_text="PA → SA 클러스터 흐름 (V6 정밀 분석)", font_size=14, width=1300, height=900)
    out_html = args.out_dir / "pa_sa_sankey_v6_final.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    flow.to_csv(args.out_dir / "pa_sa_flow_stats_v6.csv", index=False, encoding="utf-8-sig")
    print(f"✨ 최종 Sankey 저장 완료: {out_html}")

if __name__ == "__main__":
    main()
