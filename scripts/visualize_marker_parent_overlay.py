"""현토-Parent 임베딩을 겹쳐서 시각화.

기존에 생성된 임베딩 결과를 그대로 사용해서, marker_semantic_embedding.html과
parent_embedding.html을 같은 캔버스에 "오버레이" 한다.

- 입력: reports/k16_analysis/marker_semantic_embedding.csv
- 입력: reports/k16_analysis/parent_embedding.csv
- 출력: reports/k16_analysis/marker_parent_overlay.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


REPORTS_DIR = Path("reports/k16_analysis")


def load_embeddings() -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_path = REPORTS_DIR / "marker_semantic_embedding.csv"
    parent_path = REPORTS_DIR / "parent_embedding.csv"

    marker_df = pd.read_csv(marker_path)
    parent_df = pd.read_csv(parent_path)

    for col in ("marker", "count", "x", "y"):
        if col not in marker_df.columns:
            raise ValueError(f"marker 임베딩 CSV에 '{col}' 컬럼이 없습니다: {marker_path}")
    for col in ("parent_cluster_id", "rows", "x", "y"):
        if col not in parent_df.columns:
            raise ValueError(f"parent 임베딩 CSV에 '{col}' 컬럼이 없습니다: {parent_path}")

    return marker_df, parent_df


def create_overlay_plot(
    marker_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    *,
    label_top_n_markers: int,
) -> go.Figure:
    fig = go.Figure()

    # Parent: 큰 원 + 라벨
    parent_sizes = np.log10(parent_df["rows"].astype(float) + 1.0) * 18.0
    parent_text = parent_df.get(
        "parent_label",
        parent_df["parent_cluster_id"].map(lambda v: f"p{v}"),
    )

    fig.add_trace(
        go.Scatter(
            x=parent_df["x"],
            y=parent_df["y"],
            mode="markers+text",
            text=parent_text,
            textposition="top center",
            marker=dict(
                size=parent_sizes,
                symbol="circle",
                color=parent_df["parent_cluster_id"],
                colorscale="Viridis",
                opacity=0.75,
                line=dict(width=2, color="black"),
                showscale=True,
                colorbar=dict(title="Parent ID"),
            ),
            name="Parent",
            customdata=np.stack(
                [
                    parent_df["rows"].fillna(0).astype(int),
                    parent_df.get("top_children", "").fillna(""),
                    parent_df.get("top_markers", "").fillna(""),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "rows: %{customdata[0]}<br>"
                "top_children: %{customdata[1]}<br>"
                "top_markers: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    # Marker(현토): 작은 다이아몬드
    marker_sizes = np.log10(marker_df["count"].astype(float) + 1.0) * 10.0
    marker_custom = np.stack(
        [
            marker_df["count"].fillna(0).astype(int),
            marker_df.get("doc_freq", 0).fillna(0).astype(int),
            marker_df.get("num_groups", 0).fillna(0).astype(int),
            marker_df.get("top_groups", "").fillna(""),
        ],
        axis=1,
    )

    # 1) 전체 현토 점(라벨 없이)
    fig.add_trace(
        go.Scatter(
            x=marker_df["x"],
            y=marker_df["y"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                symbol="diamond",
                color="crimson",
                opacity=0.65,
                line=dict(width=1, color="darkred"),
            ),
            text=marker_df["marker"].astype(str),
            name="현토",
            customdata=marker_custom,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "count: %{customdata[0]}<br>"
                "doc_freq: %{customdata[1]}<br>"
                "num_groups: %{customdata[2]}<br>"
                "top_groups: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

    # 2) 상위 N개 현토만 텍스트 라벨을 항상 표시
    if label_top_n_markers != 0:
        top_n = int(label_top_n_markers)
        if top_n < 0:
            top_n = 0
        if top_n > 0:
            labeled = marker_df.sort_values(by="count", ascending=False).head(top_n)
            labeled_sizes = np.log10(labeled["count"].astype(float) + 1.0) * 10.0
            fig.add_trace(
                go.Scatter(
                    x=labeled["x"],
                    y=labeled["y"],
                    mode="markers+text",
                    marker=dict(
                        size=labeled_sizes,
                        symbol="diamond",
                        color="crimson",
                        opacity=0.85,
                        line=dict(width=1, color="darkred"),
                    ),
                    text=labeled["marker"].astype(str),
                    textposition="top center",
                    textfont=dict(size=10, color="darkred"),
                    name=f"현토 라벨(top {top_n})",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="현토 ↔ Parent 임베딩 오버레이 (多:多 확인)",
        xaxis_title="임베딩 좌표 X (차원축소 결과)",
        yaxis_title="임베딩 좌표 Y (차원축소 결과)",
        width=1200,
        height=800,
        hovermode="closest",
        showlegend=True,
    )

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="현토/Parent 임베딩 오버레이 시각화")
    ap.add_argument(
        "--label-top-n-markers",
        type=int,
        default=40,
        help="호버 없이 텍스트 라벨을 표시할 상위 N개 현토(0이면 라벨을 표시하지 않음)",
    )
    args = ap.parse_args()

    print("임베딩 CSV 로드 중...")
    marker_df, parent_df = load_embeddings()

    print("오버레이 시각화 생성 중...")
    fig = create_overlay_plot(marker_df, parent_df, label_top_n_markers=int(args.label_top_n_markers))

    output_path = REPORTS_DIR / "marker_parent_overlay.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    print(f"✅ 저장 완료: {output_path}")
    print(f"   - Parent: {len(parent_df)}개")
    print(f"   - 현토: {len(marker_df)}개")


if __name__ == "__main__":
    main()
