#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parent-Marker Joint Embedding 시각화 (확장 버전)

기능:
- Parent/Child 클러스터와 Marker를 동일 좌표계에서 시각화
- Convex Hull로 클러스터 영역 테두리 표시
- Parent-only 버전과 Parent+Child 버전 모두 생성
- 2D/3D 지원

사용 예:
    python scripts/visualize_parent_marker_joint_embedding_ext.py \\
        --csv hyeonto/reports/k16_analysis_minper50/recluster_k16_child_minper50/reclustered.csv \\
        --out-dir hyeonto/reports/joint_embedding_viz \\
        --method umap
"""

from __future__ import annotations

import argparse
import json
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE

# 도서별 위계적 가중치 설정 (사용자 요청 반영)
WEIGHT_MAP = {
    "논어": 5.0, "맹자": 5.0, "대학": 5.0, "중용": 5.0,
    "서경": 3.0,
    "시경": 2.0, "주역": 2.0
}

# 한자+현토 추출
_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")


# 현토 이형태 통합 규칙 및 매핑
# 데이터 분석 결과(은/는, 이/가, 을/를, 과/와) 및 접두사(이/으) 규칙 적용
HYEONTO_REPLACE_MAP = {
    "은": "는",
    "이": "가",
    "을": "를",
    "과": "와",
    "ㅣ": "가", # 고전 주격/서술격 표기
}

def normalize_marker(marker: str) -> str:
    """현토 마커 정규화 (규칙 기반 이형태 통합)"""
    if not marker:
        return marker
    
    # 0. 개별 변이형 매핑
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]

    # 1. '이-', '으-' 접두사 처리 (2글자 이상인 경우)
    # 예: 이라->라, 이면->면, 이나->나, 이요->요, 이니라->니라, 이어늘->어늘
    # 예: 으로->로, 으며->며, 으려->려 등
    if len(marker) > 1:
        if marker.startswith("이") or marker.startswith("으"):
            return marker[1:]
            
    return marker


def extract_markers(text: object) -> list[str]:
    """텍스트에서 현토 마커 추출 및 정규화"""
    if text is None or str(text) == "nan":
        return []
    out = []
    for m in _CJK_MARKER_RE.finditer(str(text)):
        marker = m.group("marker")
        if marker:
            out.append(normalize_marker(marker)) # 정규화 적용
    return out


def build_cooccurrence(
    df: pd.DataFrame,
    src_cols: list[str],
    group_level: str,
    min_marker_count: int,
    exclude_markers: set[str],
    saseo_weight: float = 1.0,
    split_markers: bool = False,
) -> tuple[np.ndarray, list[str], list[str], dict[str, int], dict[str, int]]:
    """
    공동 출현 행렬 생성
    split_markers=True 시 '고' -> '고@p1', '고@p5' 식으로 분리하여 다의성 분석
    """
    group_marker_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    marker_total: dict[str, float] = defaultdict(float)
    group_total: dict[str, int] = defaultdict(int)
    group_info: dict[str, dict] = {}

    for _, row in df.iterrows():
        # group 결정
        p_val = str(row["parent_cluster_id"])
        c_val = str(row.get("child_cluster_id", -1))
        parent_id_str = p_val[1:] if p_val.startswith("p") else p_val # "p1" -> "1"
        parent_id_int = int(parent_id_str)
        child_id_int = int(c_val) if c_val != "-1" else -1
        
        if group_level == "parent":
            group = f"p{parent_id_int}"
            ginfo = {"parent": parent_id_int, "child": -1}
        else:
            group = f"p{parent_id_int}_c{child_id_int}"
            ginfo = {"parent": parent_id_int, "child": child_id_int}

        if group not in group_info:
            group_info[group] = ginfo

        book_name = str(row.get("book_name", ""))
        # 차등 가중치 적용 로직
        weight = 1.0
        for kw, w in WEIGHT_MAP.items():
            if kw in book_name:
                weight = w
                break
        group_total[group] += 1

        for c in src_cols:
            for m in extract_markers(row.get(c)):
                if m in exclude_markers:
                    continue
                
                # 마커 키 생성 (분리 모드 유무)
                m_key = f"{m}@{parent_id_str}" if split_markers else m
                
                group_marker_counts[group][m_key] += weight
                marker_total[m_key] += weight

    # Filtering
    # 1. 마커 필터링
    filtered_markers = [m for m, count in marker_total.items() if count >= min_marker_count]
    # 2. 그룹 필터링 (마커가 없는 그룹은 제외)
    filtered_groups = [g for g, count in group_total.items() if count > 0 and any(m_key in group_marker_counts[g] for m_key in filtered_markers)]

    # 정렬
    markers = sorted(filtered_markers, key=lambda m: (-marker_total[m], m))
    groups = sorted(filtered_groups, key=lambda g: (
        int(g.split("_")[0][1:]) if "_" in g else int(g[1:]),
        int(g.split("_c")[1]) if "_c" in g else 0
    ))

    # Count matrix
    marker_to_j = {m: j for j, m in enumerate(markers)}
    C = np.zeros((len(groups), len(markers)), dtype=np.float32)

    group_counts = {}
    for i, g in enumerate(groups):
        total = 0
        for m, c in group_marker_counts[g].items():
            j = marker_to_j.get(m)
            if j is not None:
                C[i, j] = float(c)
                total += c
        group_counts[g] = total

    marker_counts = {m: marker_total[m] for m in markers}

    return C, groups, markers, group_counts, marker_counts


def ppmi_transform(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """PPMI 변환"""
    total = C.sum()
    if total <= 0:
        return np.zeros_like(C)
    
    p_ij = C / (total + eps)
    p_i = C.sum(axis=1, keepdims=True) / (total + eps)
    p_j = C.sum(axis=0, keepdims=True) / (total + eps)
    
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0).astype(np.float32)


def joint_embedding(C: np.ndarray, method: str, svd_dim: int, seed: int, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Group과 Marker의 공동 임베딩 (SVD → 2D/3D 축소)"""
    X = ppmi_transform(C)
    n_groups, n_markers = X.shape

    # SVD로 공동 잠재공간 생성
    svd_dim = min(svd_dim, min(n_groups, n_markers) - 1)
    if svd_dim < 2:
        svd_dim = 2

    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    group_emb = svd.fit_transform(X)  # (G, k)
    s = svd.singular_values_
    vt = svd.components_  # (k, M)
    marker_emb = (vt.T * s).astype(np.float32)  # (M, k)

    joint = np.vstack([group_emb, marker_emb])

    # 2D/3D 축소
    if method == "pca":
        reducer = PCA(n_components=dim, random_state=seed)
        coords = reducer.fit_transform(joint)
    elif method == "tsne":
        init = PCA(n_components=dim, random_state=seed).fit_transform(joint)
        reducer = TSNE(
            n_components=dim,
            random_state=seed,
            init=init,
            perplexity=min(30, max(5, len(joint) // 3)),
        )
        coords = reducer.fit_transform(joint)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=dim, random_state=seed, n_neighbors=min(30, len(joint) - 1), min_dist=0.1)
            coords = reducer.fit_transform(joint)
        except ImportError:
            print("Warning: umap-learn not installed, using PCA")
            reducer = PCA(n_components=dim, random_state=seed)
            coords = reducer.fit_transform(joint)
    else:
        reducer = PCA(n_components=dim, random_state=seed)
        coords = reducer.fit_transform(joint)

    group_coords = coords[:n_groups]
    marker_coords = coords[n_groups:]

    return group_coords.astype(np.float32), marker_coords.astype(np.float32)


def compute_convex_hulls(group_coords: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    """Parent별 Convex Hull 계산"""
    hulls = {}
    parent_points: dict[int, list] = defaultdict(list)

    for i, g in enumerate(groups):
        parent = int(g.split("_")[0][1:]) if "_" in g else int(g[1:])
        parent_points[parent].append(group_coords[i, :2])

    for parent, pts in parent_points.items():
        if len(pts) < 3:
            continue
        try:
            pts_arr = np.array(pts)
            hull = ConvexHull(pts_arr)
            verts = pts_arr[hull.vertices]
            verts = np.vstack([verts, verts[0]])  # close polygon
            hulls[f"p{parent}"] = verts
        except Exception:
            pass

    return hulls


def save_html_2d(
    out_path: Path,
    groups: list[str],
    group_coords: np.ndarray,
    group_counts: dict[str, int],
    markers: list[str],
    marker_coords: np.ndarray,
    marker_counts: dict[str, int],
    hulls: dict[str, np.ndarray],
    label_top_markers: int,
    title: str,
    method: str,
) -> None:
    """2D Plotly HTML 생성"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return

    fig = go.Figure()

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    # Convex hulls (클러스터 영역)
    for i, (pid, verts) in enumerate(hulls.items()):
        c = colors[i % len(colors)]
        rgb = tuple(int(c[j:j + 2], 16) for j in (1, 3, 5))
        fig.add_trace(go.Scatter(
            x=verts[:, 0], y=verts[:, 1],
            mode="lines",
            line=dict(color=c, width=1.5, dash="dot"),
            fill="toself",
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)",
            name=f"{pid} area",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Markers (녹색 원)
    mc_arr = np.array([marker_counts.get(m, 0) for m in markers])
    top_idx = set(np.argsort(mc_arr)[::-1][:label_top_markers])
    other_idx = [i for i in range(len(markers)) if i not in top_idx]

    if other_idx:
        fig.add_trace(go.Scatter(
            x=marker_coords[other_idx, 0],
            y=marker_coords[other_idx, 1],
            mode="markers",
            marker=dict(size=np.clip(np.log1p(mc_arr[other_idx]) * 2.5, 4, 12), color="#2ca02c", opacity=0.55),
            text=[markers[i] for i in other_idx],
            customdata=mc_arr[other_idx],
            hovertemplate="<b>%{text}</b><br>count=%{customdata}<extra>Marker</extra>",
            name="Marker",
        ))

    # Top markers (주황색 + 라벨)
    top_list = list(top_idx)
    if top_list:
        fig.add_trace(go.Scatter(
            x=marker_coords[top_list, 0],
            y=marker_coords[top_list, 1],
            mode="markers+text",
            marker=dict(size=np.clip(np.log1p(mc_arr[top_list]) * 3, 6, 16), color="#ff7f0e", opacity=0.85),
            text=[markers[i] for i in top_list],
            textposition="top center",
            textfont=dict(size=9),
            customdata=mc_arr[top_list],
            hovertemplate="<b>%{text}</b><br>count=%{customdata}<extra>Marker (top)</extra>",
            name=f"Marker (top {label_top_markers} labeled)",
        ))

    # Groups (다이아몬드)
    gc_arr = np.array([group_counts.get(g, 0) for g in groups])
    fig.add_trace(go.Scatter(
        x=group_coords[:, 0],
        y=group_coords[:, 1],
        mode="markers+text",
        marker=dict(
            size=np.clip(np.log1p(gc_arr) * 2.5, 10, 22),
            color="#17becf",
            symbol="diamond",
            line=dict(color="darkblue", width=1),
        ),
        text=groups,
        textposition="bottom center",
        textfont=dict(size=10, color="darkblue"),
        customdata=gc_arr,
        hovertemplate="<b>%{text}</b><br>rows=%{customdata}<extra>Parent</extra>",
        name="Parent",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"2D ({method})",
        yaxis_title=f"2D ({method})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        width=1100,
        height=850,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Saved: {out_path}")


def save_html_2d_clean_parent(
    out_path: Path,
    groups: list[str],
    group_coords: np.ndarray,
    group_counts: dict[str, int],
    markers: list[str],
    marker_coords: np.ndarray,
    marker_counts: dict[str, int],
    hulls: dict[str, np.ndarray],
    label_top_markers: int,
    title: str,
    method: str,
) -> None:
    """Parent 중심의 깔끔한 2D 시각화 (Child는 테두리용으로만 사용)"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return

    fig = go.Figure()

    # Parent별 정보 집계 (Child 점들의 중앙값 계산 등)
    parent_data = defaultdict(list)
    for i, g in enumerate(groups):
        pid = int(g.split("_")[0][1:]) if "_" in g else int(g[1:])
        parent_data[pid].append({
            "coord": group_coords[i, :2],
            "count": group_counts.get(g, 0)
        })

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    # 1. Convex hulls (각 Parent의 영역)
    pid_list = sorted(parent_data.keys())
    for i, pid in enumerate(pid_list):
        pname = f"p{pid}"
        if pname not in hulls:
            continue
        verts = hulls[pname]
        c = colors[i % len(colors)]
        rgb = tuple(int(c[j:j+2], 16) for j in (1, 3, 5))
        fig.add_trace(go.Scatter(
            x=verts[:, 0], y=verts[:, 1],
            mode="lines",
            line=dict(color=c, width=2, dash="solid"),
            fill="toself",
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
            name=f"{pname} area",
            legendgroup=pname,
            hoverinfo="skip",
        ))

    # 2. Markers (녹색 원형)
    mc_arr = np.array([marker_counts.get(m, 0) for m in markers])
    top_idx = set(np.argsort(mc_arr)[::-1][:label_top_markers])
    other_idx = [i for i in range(len(markers)) if i not in top_idx]

    if other_idx:
        fig.add_trace(go.Scatter(
            x=marker_coords[other_idx, 0],
            y=marker_coords[other_idx, 1],
            mode="markers",
            marker=dict(size=np.clip(np.log1p(mc_arr[other_idx]) * 2.5, 4, 12), color="#2ca02c", opacity=0.4),
            text=[markers[i] for i in other_idx],
            customdata=mc_arr[other_idx],
            hovertemplate="<b>%{text}</b><br>count=%{customdata}<extra>Marker</extra>",
            name="Marker",
        ))

    if top_idx:
        top_list = list(top_idx)
        fig.add_trace(go.Scatter(
            x=marker_coords[top_list, 0],
            y=marker_coords[top_list, 1],
            mode="markers+text",
            marker=dict(size=np.clip(np.log1p(mc_arr[top_list]) * 3, 6, 16), color="#ff7f0e", opacity=0.8),
            text=[markers[i] for i in top_list],
            textposition="top center",
            textfont=dict(size=10),
            customdata=mc_arr[top_list],
            hovertemplate="<b>%{text}</b><br>count=%{customdata}<extra>Marker (top)</extra>",
            name="Top Markers",
        ))

    # 3. Parent Centroids (유일한 다이아몬드)
    p_x, p_y, p_labels, p_counts = [], [], [], []
    for pid in pid_list:
        data = parent_data[pid]
        coords = np.array([d["coord"] for d in data])
        centroid = coords.mean(axis=0)
        total_count = sum(d["count"] for d in data)
        p_x.append(centroid[0])
        p_y.append(centroid[1])
        p_labels.append(f"p{pid}")
        p_counts.append(total_count)

    fig.add_trace(go.Scatter(
        x=p_x, y=p_y,
        mode="markers+text",
        marker=dict(
            size=np.clip(np.log1p(p_counts) * 3, 12, 25),
            color="#17becf",
            symbol="diamond",
            line=dict(color="black", width=1.5),
        ),
        text=p_labels,
        textposition="middle center",
        textfont=dict(size=12, color="white", family="Arial Black"),
        customdata=p_counts,
        hovertemplate="<b>%{text}</b><br>total_rows=%{customdata}<extra>Parent Group</extra>",
        name="Parent Group",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"2D ({method})",
        yaxis_title=f"2D ({method})",
        legend=dict(orientation="v", x=1.02, y=1),
        width=1200,
        height=900,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=True, zerolinecolor="gray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=True, zerolinecolor="gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Saved clean visualization: {out_path}")


def save_html_3d(
    out_path: Path,
    groups: list[str],
    group_coords: np.ndarray,
    group_counts: dict[str, int],
    markers: list[str],
    marker_coords: np.ndarray,
    marker_counts: dict[str, int],
    label_top_markers: int,
    title: str,
) -> None:
    """3D Plotly HTML 생성"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return

    fig = go.Figure()

    mc_arr = np.array([marker_counts.get(m, 0) for m in markers])
    top_idx = set(np.argsort(mc_arr)[::-1][:label_top_markers])
    other_idx = [i for i in range(len(markers)) if i not in top_idx]

    z_m = marker_coords[:, 2] if marker_coords.shape[1] > 2 else np.zeros(len(markers))

    if other_idx:
        fig.add_trace(go.Scatter3d(
            x=marker_coords[other_idx, 0],
            y=marker_coords[other_idx, 1],
            z=z_m[other_idx],
            mode="markers",
            marker=dict(size=np.clip(np.log1p(mc_arr[other_idx]) * 1.5, 2, 8), color="#2ca02c", opacity=0.5),
            text=[markers[i] for i in other_idx],
            hovertemplate="<b>%{text}</b><extra>Marker</extra>",
            name="Marker",
        ))

    top_list = list(top_idx)
    if top_list:
        fig.add_trace(go.Scatter3d(
            x=marker_coords[top_list, 0],
            y=marker_coords[top_list, 1],
            z=z_m[top_list],
            mode="markers+text",
            marker=dict(size=np.clip(np.log1p(mc_arr[top_list]) * 2, 4, 12), color="#ff7f0e", opacity=0.8),
            text=[markers[i] for i in top_list],
            hovertemplate="<b>%{text}</b><extra>Marker (top)</extra>",
            name=f"Marker (top {label_top_markers})",
        ))

    gc_arr = np.array([group_counts.get(g, 0) for g in groups])
    z_g = group_coords[:, 2] if group_coords.shape[1] > 2 else np.zeros(len(groups))
    fig.add_trace(go.Scatter3d(
        x=group_coords[:, 0],
        y=group_coords[:, 1],
        z=z_g,
        mode="markers+text",
        marker=dict(size=np.clip(np.log1p(gc_arr) * 1.5, 5, 14), color="#17becf", symbol="diamond"),
        text=groups,
        hovertemplate="<b>%{text}</b><br>rows=%{customdata}<extra>Parent</extra>",
        name="Parent",
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3"),
        width=1100,
        height=850,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Saved: {out_path}")


def save_html_3d_clean_parent(
    out_path: Path,
    groups: list[str],
    group_coords: np.ndarray,
    group_counts: dict[str, int],
    markers: list[str],
    marker_coords: np.ndarray,
    marker_counts: dict[str, int],
    label_top_markers: int,
    title: str,
) -> None:
    """Parent 중심의 깔끔한 3D 시각화"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return

    fig = go.Figure()

    # Parent별 정보 집계
    parent_data = defaultdict(list)
    for i, g in enumerate(groups):
        pid = int(g.split("_")[0][1:]) if "_" in g else int(g[1:])
        parent_data[pid].append({
            "coord": group_coords[i, :3],
            "count": group_counts.get(g, 0)
        })

    # 1. Markers (녹색/주황색 원)
    mc_arr = np.array([marker_counts.get(m, 0) for m in markers])
    top_idx = set(np.argsort(mc_arr)[::-1][:label_top_markers])
    other_idx = [i for i in range(len(markers)) if i not in top_idx]
    z_m = marker_coords[:, 2] if marker_coords.shape[1] > 2 else np.zeros(len(markers))

    if other_idx:
        fig.add_trace(go.Scatter3d(
            x=marker_coords[other_idx, 0],
            y=marker_coords[other_idx, 1],
            z=z_m[other_idx],
            mode="markers",
            marker=dict(size=np.clip(np.log1p(mc_arr[other_idx]) * 1.5, 2, 8), color="#2ca02c", opacity=0.4),
            text=[markers[i] for i in other_idx],
            hovertemplate="<b>%{text}</b><extra>Marker</extra>",
            name="Marker",
        ))

    if top_idx:
        top_list = list(top_idx)
        fig.add_trace(go.Scatter3d(
            x=marker_coords[top_list, 0],
            y=marker_coords[top_list, 1],
            z=z_m[top_list],
            mode="markers+text",
            marker=dict(size=np.clip(np.log1p(mc_arr[top_list]) * 2, 4, 12), color="#ff7f0e", opacity=0.8),
            text=[markers[i] for i in top_list],
            hovertemplate="<b>%{text}</b><extra>Marker (top)</extra>",
            name="Top Markers",
        ))

    # 2. Parent Centroids (유일한 다이아몬드)
    pid_list = sorted(parent_data.keys())
    p_x, p_y, p_z, p_labels, p_counts = [], [], [], [], []
    for pid in pid_list:
        data = parent_data[pid]
        coords = np.array([d["coord"] for d in data])
        centroid = coords.mean(axis=0)
        total_count = sum(d["count"] for d in data)
        p_x.append(centroid[0])
        p_y.append(centroid[1])
        p_z.append(centroid[2])
        p_labels.append(f"p{pid}")
        p_counts.append(total_count)

    fig.add_trace(go.Scatter3d(
        x=p_x, y=p_y, z=p_z,
        mode="markers+text",
        marker=dict(
            size=np.clip(np.log1p(p_counts) * 2, 10, 20),
            color="#17becf",
            symbol="diamond",
            line=dict(color="black", width=1),
        ),
        text=p_labels,
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{text}</b><br>total_rows=%{customdata}<extra>Parent Group</extra>",
        customdata=p_counts,
        name="Parent Group",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3",
            xaxis=dict(gridcolor="lightgray"),
            yaxis=dict(gridcolor="lightgray"),
            zaxis=dict(gridcolor="lightgray"),
        ),
        width=1200, height=900,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Saved clean 3D visualization: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Parent-Marker Joint Embedding (확장)")
    p.add_argument("--csv", type=Path, required=True, help="입력 CSV")
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/joint_embedding_viz"))
    p.add_argument("--src-cols", type=str, default="src_left,src_right")
    p.add_argument("--method", choices=["umap", "pca", "tsne"], default="umap")
    p.add_argument("--svd-dim", type=int, default=32, help="SVD 차원")
    p.add_argument("--dim", type=int, default=2, choices=[2, 3], help="최종 시각화 차원")
    p.add_argument("--min-count", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saseo-weight", type=float, default=2.0, help="사서(Four Books) 예시에 부여할 가중치")
    p.add_argument("--label-top-markers", type=int, default=40)
    p.add_argument("--exclude-markers", type=str, default="<NULL>")
    p.add_argument("--split-markers", action="store_true", help="마커를 Parent ID별로 분리하여 분석 (다의성 분석용)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    src_cols = [c.strip() for c in args.src_cols.split(",") if c.strip()]
    exclude = {m.strip() for m in args.exclude_markers.split(",") if m.strip()}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 분석 수행 =====
    mode_str = "Split Mode" if args.split_markers else "Normal Mode"
    suffix = "split" if args.split_markers else "normal"
    
    print(f"\n📊 {mode_str} 분석 중... (Weighting: Tiered [Saseo:5x, Seogyung:3x, others:2x/1x])")
    C, groups, markers, g_counts, m_counts = build_cooccurrence(
        df, src_cols, "parent_child", args.min_count, exclude, saseo_weight=args.saseo_weight, split_markers=args.split_markers
    )
    
    # 2D 임베딩
    print("  - Computing 2D joint embedding...")
    gxy_2d, mxy_2d = joint_embedding(C, args.method, args.svd_dim, args.seed, dim=2)
    hulls_2d = compute_convex_hulls(gxy_2d, groups)
    
    save_html_2d_clean_parent(
        args.out_dir / f"joint_embedding_{suffix}_2d.html",
        groups, gxy_2d, g_counts,
        markers, mxy_2d, m_counts,
        hulls_2d, args.label_top_markers,
        f"Parent-Marker {mode_str} ({args.method.upper()})",
        args.method,
    )

    # 3D 임베딩
    print("  - Computing 3D joint embedding...")
    gxy_3d, mxy_3d = joint_embedding(C, args.method, args.svd_dim, args.seed, dim=3)
    save_html_3d_clean_parent(
        args.out_dir / f"joint_embedding_{suffix}_3d.html",
        groups, gxy_3d, g_counts,
        markers, mxy_3d, m_counts,
        args.label_top_markers,
        f"Parent-Marker 3D {mode_str} ({args.method.upper()})",
    )

    # 통합 CSV 저장
    out_csv = args.out_dir / "joint_embedding_parent_only.csv"
    records = []
    pid_map = defaultdict(list)
    for i, g in enumerate(groups):
        pid = int(g.split("_")[0][1:]) if "_" in g else int(g[1:])
        pid_map[pid].append(gxy_2d[i])
    
    # Parent 위치 (2D 기준 centroid) 계산하여 저장
    for pid in sorted(pid_map.keys()):
        coords = np.array(pid_map[pid])
        centroid = coords.mean(axis=0)
        total_count = sum(g_counts.get(g, 0) for g in groups if (int(g.split("_")[0][1:]) if "_" in g else int(g[1:])) == pid)
        records.append({"kind": "parent", "name": f"p{pid}", "x": centroid[0], "y": centroid[1], "count": total_count})
        
    for i, m in enumerate(markers):
        records.append({"kind": "marker", "name": m, "x": mxy_2d[i, 0], "y": mxy_2d[i, 1], "count": m_counts.get(m, 0)})
    
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved CSV: {out_csv}")

    # Config
    cfg = {
        "csv": str(args.csv),
        "method": args.method,
        "svd_dim": args.svd_dim,
        "dim": args.dim,
        "min_count": args.min_count,
        "seed": args.seed,
        "label_top_markers": args.label_top_markers,
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 모든 결과물 저장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
