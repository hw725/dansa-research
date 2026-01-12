#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""주요 현토의 Parent별 분포 분석

목표: 고빈도 현토가 어떤 parent에 얼마나 분포하는지 시각화
출력:
  - 주요 현토 Top N의 parent별 분포 히트맵
  - 현토별 분산 지수(Entropy): 분산 높으면 범용, 낮으면 장르 특화
  - 상세 분포 CSV

사용 예:
    python scripts/analyze_marker_distribution.py \
        --csv hyeonto/reports/recluster_k16_child/reclustered.csv \
        --out-dir hyeonto/reports/marker_distribution \
        --top-n 30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re

# 현토 추출 정규식
_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")

# 현토 정규화 규칙
HYEONTO_REPLACE_MAP = {
    "은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가",
}


def normalize_marker(marker: str) -> str:
    """현토 정규화 (이형태 통합)"""
    if not marker:
        return marker
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1 and marker[0] in ("이", "으"):
        return marker[1:]
    return marker


def extract_markers(text: object) -> list[str]:
    """텍스트에서 현토 추출"""
    if text is None or str(text) == "nan":
        return []
    out = []
    for m in _CJK_MARKER_RE.finditer(str(text)):
        marker = m.group("marker")
        if marker:
            out.append(normalize_marker(marker))
    return out


def compute_entropy(counts: np.ndarray) -> float:
    """분포의 엔트로피 계산 (0 = 한 곳에 집중, 높을수록 분산)"""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main() -> int:
    p = argparse.ArgumentParser(description="주요 현토의 Parent별 분포 분석")
    p.add_argument("--csv", type=Path, required=True, help="입력 CSV (reclustered.csv)")
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/marker_distribution"))
    p.add_argument("--src-cols", type=str, default="src_left,src_right")
    p.add_argument("--top-n", type=int, default=30, help="분석할 주요 현토 개수")
    p.add_argument("--min-total-count", type=int, default=50, help="최소 총 빈도")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    src_cols = [c.strip() for c in args.src_cols.split(",") if c.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 로드: {len(df):,}개 행")

    # Parent별 현토 빈도 집계
    marker_parent_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    marker_total: dict[str, int] = defaultdict(int)
    parent_set: set[int] = set()

    # Determine cluster ID column
    cluster_col = "parent_cluster_id"
    if cluster_col not in df.columns:
        if "cluster_id" in df.columns:
            cluster_col = "cluster_id"
        else:
            raise ValueError("클러스터 ID 컬럼(parent_cluster_id 또는 cluster_id)을 찾을 수 없습니다.")

    for _, row in df.iterrows():
        p_val = str(row.get(cluster_col, "p0"))
        # p0, p1... 또는 0, 1... 형식 모두 지원
        if isinstance(p_val, str) and p_val.startswith("p"):
            parent_id = int(p_val[1:])
        else:
            parent_id = int(float(p_val))
        parent_set.add(parent_id)

        for col in src_cols:
            for m in extract_markers(row.get(col)):
                marker_parent_counts[m][parent_id] += 1
                marker_total[m] += 1

    # 상위 N개 현토 선택
    top_markers = sorted(
        [m for m, cnt in marker_total.items() if cnt >= args.min_total_count],
        key=lambda m: -marker_total[m]
    )[:args.top_n]

    parents = sorted(parent_set)
    print(f"✅ 분석 대상: {len(top_markers)}개 현토, {len(parents)}개 parent")

    # 분포 행렬 생성
    dist_matrix = np.zeros((len(top_markers), len(parents)), dtype=np.float32)
    for i, m in enumerate(top_markers):
        for j, p in enumerate(parents):
            dist_matrix[i, j] = marker_parent_counts[m][p]

    # 정규화 (행별 비율)
    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    dist_matrix_norm = dist_matrix / row_sums

    # 엔트로피 계산
    entropies = []
    for i, m in enumerate(top_markers):
        ent = compute_entropy(dist_matrix[i])
        max_ent = np.log2(len(parents)) if len(parents) > 1 else 1
        norm_ent = ent / max_ent if max_ent > 0 else 0
        entropies.append({
            "marker": m,
            "total_count": int(marker_total[m]),
            "entropy": round(ent, 3),
            "normalized_entropy": round(norm_ent, 3),
            "interpretation": "범용" if norm_ent > 0.7 else ("중간" if norm_ent > 0.4 else "장르특화"),
        })

    # 결과 저장
    # 1. 분포 CSV
    dist_df = pd.DataFrame(dist_matrix, index=top_markers, columns=[f"p{p}" for p in parents])
    dist_df.insert(0, "total", [marker_total[m] for m in top_markers])
    dist_df.to_csv(args.out_dir / "marker_parent_distribution.csv", encoding="utf-8-sig")
    print(f"✅ 저장: marker_parent_distribution.csv")

    # 2. 정규화 분포 CSV
    dist_norm_df = pd.DataFrame(dist_matrix_norm, index=top_markers, columns=[f"p{p}" for p in parents])
    dist_norm_df.to_csv(args.out_dir / "marker_parent_distribution_normalized.csv", encoding="utf-8-sig")
    print(f"✅ 저장: marker_parent_distribution_normalized.csv")

    # 3. 엔트로피 CSV
    ent_df = pd.DataFrame(entropies)
    ent_df.to_csv(args.out_dir / "marker_entropy.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 저장: marker_entropy.csv")

    # 4. 히트맵 HTML
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # 히트맵 (정규화)
        fig = px.imshow(
            dist_matrix_norm,
            labels=dict(x="Parent", y="현토", color="비율"),
            x=[f"p{p}" for p in parents],
            y=top_markers,
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(
            title=f"주요 현토 Top {len(top_markers)}의 Parent별 분포 (정규화)",
            width=1000,
            height=max(600, len(top_markers) * 25),
        )
        fig.write_html(str(args.out_dir / "marker_distribution_heatmap.html"), include_plotlyjs="cdn")
        print(f"✅ 저장: marker_distribution_heatmap.html")

        # 엔트로피 바차트
        ent_df_sorted = ent_df.sort_values("normalized_entropy", ascending=True)
        colors = ["#d62728" if v < 0.4 else ("#ff7f0e" if v < 0.7 else "#2ca02c") 
                  for v in ent_df_sorted["normalized_entropy"]]
        
        fig2 = go.Figure(go.Bar(
            x=ent_df_sorted["normalized_entropy"],
            y=ent_df_sorted["marker"],
            orientation="h",
            marker_color=colors,
            text=[f'{v:.2f}' for v in ent_df_sorted["normalized_entropy"]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="현토별 분산 지수 (Normalized Entropy)",
            xaxis_title="Normalized Entropy (0=장르특화, 1=범용)",
            yaxis_title="현토",
            height=max(500, len(top_markers) * 22),
            width=800,
        )
        fig2.write_html(str(args.out_dir / "marker_entropy_chart.html"), include_plotlyjs="cdn")
        print(f"✅ 저장: marker_entropy_chart.html")

    except ImportError:
        print("⚠️ plotly 미설치 - 시각화 생략")

    # 5. 요약 출력
    print("\n📊 엔트로피 상위 10 (범용):")
    for row in sorted(entropies, key=lambda x: -x["normalized_entropy"])[:10]:
        print(f"  {row['marker']:>6} : {row['normalized_entropy']:.2f} ({row['interpretation']}, n={row['total_count']:,})")

    print("\n📊 엔트로피 하위 10 (장르특화):")
    for row in sorted(entropies, key=lambda x: x["normalized_entropy"])[:10]:
        print(f"  {row['marker']:>6} : {row['normalized_entropy']:.2f} ({row['interpretation']}, n={row['total_count']:,})")

    # Config
    cfg = {
        "csv": str(args.csv),
        "top_n": args.top_n,
        "min_total_count": args.min_total_count,
        "num_parents": len(parents),
        "num_markers_analyzed": len(top_markers),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 전체 결과: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
