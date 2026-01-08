#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""장르 잔차화 기반 현토 분석

장르 효과를 통계적으로 제거하여 순수한 문법적 기능 차원의 클러스터링 수행.

Usage:
    python scripts/analyze_residualized_markers.py \
        --input hyeonto/datasets/pa_train_full.csv \
        --clusters hyeonto/reports/recluster_k16_child/reclustered.csv \
        --genre-level detail \
        --weight-saseo 1.0 \
        --weight-samgyeong 1.0 \
        --out-dir hyeonto/reports/residualized_analysis
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

# ============================================================================
# 장르 분류 함수 (經史子集 기반)
# ============================================================================

def classify_genre_detail(book: str) -> str:
    """세부 장르 분류 (7개)"""
    book = str(book)
    # 經 (경전)
    if any(x in book for x in ['논어', '맹자', '대학', '중용']):
        return '經_사서'
    elif any(x in book for x in ['시경', '서경', '주역']):
        return '經_삼경'
    elif '춘추좌씨전' in book:
        return '經_춘추'
    elif '예기' in book:
        return '經_예기'
    # 史 (역사서)
    elif '자치통감' in book:
        return '史'
    # 集 (문집)
    elif '당송팔대가' in book:
        return '集_산문'
    elif '당시삼백수' in book:
        return '集_시가'
    return '기타'


def classify_genre_top(book: str) -> str:
    """상위 장르 분류 (3개)"""
    detail = classify_genre_detail(book)
    if detail.startswith('經'):
        return '經'
    elif detail.startswith('史'):
        return '史'
    elif detail.startswith('集'):
        return '集'
    return '기타'


# ============================================================================
# 현토 마커 추출 및 정규화
# ============================================================================

_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")

HYEONTO_REPLACE_MAP = {
    "은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가",
}

def normalize_marker(marker: str) -> str:
    """현토 마커 정규화"""
    if not marker:
        return marker
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1 and (marker.startswith("이") or marker.startswith("으")):
        return marker[1:]
    return marker


def extract_markers(text: object) -> list[str]:
    """텍스트에서 현토 마커 추출"""
    if text is None or str(text) == "nan":
        return []
    out = []
    for m in _CJK_MARKER_RE.finditer(str(text)):
        marker = m.group("marker")
        if marker:
            out.append(normalize_marker(marker))
    return out


# ============================================================================
# 잔차화 핵심 로직
# ============================================================================

def compute_genre_marker_means(
    df: pd.DataFrame,
    genre_col: str,
    src_cols: list[str],
) -> dict[str, dict[str, float]]:
    """
    장르별 마커 평균 빈도율 계산
    
    Returns:
        {genre: {marker: mean_freq_per_row}}
    """
    genre_marker_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    genre_row_counts: dict[str, int] = defaultdict(int)
    
    for _, row in df.iterrows():
        genre = row[genre_col]
        genre_row_counts[genre] += 1
        for col in src_cols:
            for marker in extract_markers(row.get(col)):
                genre_marker_counts[genre][marker] += 1
    
    # 장르별 마커 평균 (출현 빈도 / 장르 내 총 행 수)
    genre_means: dict[str, dict[str, float]] = {}
    for genre, markers in genre_marker_counts.items():
        total_rows = genre_row_counts[genre]
        genre_means[genre] = {m: c / total_rows for m, c in markers.items()}
    
    return genre_means


def build_residualized_cooccurrence(
    df: pd.DataFrame,
    genre_col: str,
    group_col: str,
    src_cols: list[str],
    genre_means: dict[str, dict[str, float]],
    weight_map: dict[str, float],
    min_marker_count: int = 50,
) -> tuple[np.ndarray, list[str], list[str], dict[str, int], dict[str, float]]:
    """
    잔차화 공동출현 행렬 생성
    
    각 마커 출현에서 장르 평균을 차감한 잔차를 누적
    """
    group_marker_residuals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    marker_total: dict[str, float] = defaultdict(float)
    group_total: dict[str, int] = defaultdict(int)
    
    for _, row in df.iterrows():
        genre = row[genre_col]
        
        # 그룹 결정 (parent_cluster_id)
        p_val = str(row[group_col])
        parent_id_str = p_val[1:] if p_val.startswith("p") else p_val
        group = f"p{parent_id_str}"
        
        group_total[group] += 1
        
        # 가중치 결정
        book_name = str(row.get("book_name", ""))
        weight = 1.0
        for kw, w in weight_map.items():
            if kw in book_name:
                weight = w
                break
        
        for col in src_cols:
            for marker in extract_markers(row.get(col)):
                # 잔차 계산: raw(1) - 장르평균
                genre_mean = genre_means.get(genre, {}).get(marker, 0)
                residual = (1.0 - genre_mean) * weight
                
                group_marker_residuals[group][marker] += residual
                marker_total[marker] += abs(residual)  # 절대값으로 필터링용
    
    # 필터링
    filtered_markers = [m for m, count in marker_total.items() if count >= min_marker_count]
    filtered_groups = [g for g, count in group_total.items() if count > 0]
    
    # 정렬
    markers = sorted(filtered_markers, key=lambda m: (-marker_total[m], m))
    groups = sorted(filtered_groups, key=lambda g: int(g[1:]))
    
    # 행렬 생성
    marker_to_j = {m: j for j, m in enumerate(markers)}
    C = np.zeros((len(groups), len(markers)), dtype=np.float32)
    
    group_counts = {}
    for i, g in enumerate(groups):
        for m, r in group_marker_residuals[g].items():
            j = marker_to_j.get(m)
            if j is not None:
                C[i, j] = float(r)
        group_counts[g] = group_total[g]
    
    marker_counts = {m: marker_total[m] for m in markers}
    
    return C, groups, markers, group_counts, marker_counts


def ppmi_transform(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """PPMI 변환 (잔차에도 적용 가능하도록 수정)"""
    # 음수 값 처리: 잔차는 음수일 수 있으므로 shift
    C_shifted = C - C.min() + eps
    
    total = C_shifted.sum()
    if total <= 0:
        return np.zeros_like(C)
    
    p_ij = C_shifted / (total + eps)
    p_i = C_shifted.sum(axis=1, keepdims=True) / (total + eps)
    p_j = C_shifted.sum(axis=0, keepdims=True) / (total + eps)
    
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0).astype(np.float32)


def joint_embedding(C: np.ndarray, svd_dim: int, seed: int, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Group과 Marker의 공동 임베딩"""
    X = ppmi_transform(C)
    n_groups, n_markers = X.shape
    
    svd_dim = min(svd_dim, min(n_groups, n_markers) - 1)
    if svd_dim < 2:
        svd_dim = 2
    
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    group_emb = svd.fit_transform(X)
    s = svd.singular_values_
    vt = svd.components_
    marker_emb = (vt.T * s).astype(np.float32)
    
    joint = np.vstack([group_emb, marker_emb])
    
    # UMAP 시도, 실패 시 PCA
    try:
        import umap
        reducer = umap.UMAP(n_components=dim, random_state=seed, n_neighbors=min(30, len(joint) - 1), min_dist=0.1)
        coords = reducer.fit_transform(joint)
    except ImportError:
        reducer = PCA(n_components=dim, random_state=seed)
        coords = reducer.fit_transform(joint)
    
    group_coords = coords[:n_groups]
    marker_coords = coords[n_groups:]
    
    return group_coords.astype(np.float32), marker_coords.astype(np.float32)


# ============================================================================
# 시각화
# ============================================================================

def save_html_2d(
    out_path: Path,
    groups: list[str],
    group_coords: np.ndarray,
    group_counts: dict[str, int],
    markers: list[str],
    marker_coords: np.ndarray,
    marker_counts: dict[str, float],
    label_top_markers: int,
    title: str,
) -> None:
    """2D Plotly HTML 생성"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return
    
    fig = go.Figure()
    
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    
    # Markers
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
            hovertemplate="<b>%{text}</b><extra>Marker</extra>",
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
            hovertemplate="<b>%{text}</b><extra>Top Marker</extra>",
            name="Top Markers",
        ))
    
    # Groups
    gc_arr = np.array([group_counts.get(g, 0) for g in groups])
    fig.add_trace(go.Scatter(
        x=group_coords[:, 0],
        y=group_coords[:, 1],
        mode="markers+text",
        marker=dict(
            size=np.clip(np.log1p(gc_arr) * 3, 12, 25),
            color="#17becf",
            symbol="diamond",
            line=dict(color="black", width=1.5),
        ),
        text=groups,
        textposition="middle center",
        textfont=dict(size=12, color="white", family="Arial Black"),
        hovertemplate="<b>%{text}</b><br>rows=%{customdata}<extra>Parent</extra>",
        customdata=gc_arr,
        name="Parent Group",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Dim1",
        yaxis_title="Dim2",
        width=1200,
        height=900,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Saved: {out_path}")


def compute_genre_entropy(df: pd.DataFrame, genre_col: str, group_col: str) -> dict[str, float]:
    """각 클러스터 내 장르 분포 엔트로피 계산 (높을수록 장르 혼재)"""
    from scipy.stats import entropy
    
    result = {}
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        genre_counts = subset[genre_col].value_counts()
        probs = genre_counts.values / genre_counts.sum()
        result[str(group)] = entropy(probs)
    
    return result


# ============================================================================
# 메인
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description="장르 잔차화 기반 현토 분석")
    p.add_argument("--input", type=Path, default=Path("hyeonto/datasets/pa_train_full.csv"))
    p.add_argument("--clusters", type=Path, default=Path("hyeonto/reports/recluster_k16_child/reclustered.csv"))
    p.add_argument("--genre-level", choices=["detail", "top"], default="detail")
    p.add_argument("--weight-saseo", type=float, default=1.0, help="사서 가중치")
    p.add_argument("--weight-samgyeong", type=float, default=1.0, help="삼경 가중치")
    p.add_argument("--weight-other-gyeong", type=float, default=1.0, help="기타 경전 가중치")
    p.add_argument("--min-count", type=int, default=50)
    p.add_argument("--svd-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-top-markers", type=int, default=40)
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/residualized_analysis"))
    args = p.parse_args()
    
    # 데이터 로드
    if not args.input.exists():
        print(f"❌ 입력 파일 없음: {args.input}")
        return 1
    
    if not args.clusters.exists():
        print(f"❌ 클러스터 파일 없음: {args.clusters}")
        return 1
    
    df_full = pd.read_csv(args.input)
    df_clusters = pd.read_csv(args.clusters)
    
    print(f"📊 원본 데이터: {len(df_full):,} rows")
    print(f"📊 클러스터 데이터: {len(df_clusters):,} rows")
    
    # 장르 라벨 부여
    classify_func = classify_genre_detail if args.genre_level == "detail" else classify_genre_top
    df_clusters["genre"] = df_clusters["book_name"].apply(classify_func)
    
    print(f"\n📊 장르 분포 ({args.genre_level}):")
    print(df_clusters["genre"].value_counts())
    
    # 가중치 맵 구성
    weight_map = {
        "논어": args.weight_saseo, "맹자": args.weight_saseo, 
        "대학": args.weight_saseo, "중용": args.weight_saseo,
        "시경": args.weight_samgyeong, "서경": args.weight_samgyeong, 
        "주역": args.weight_samgyeong,
        "춘추좌씨전": args.weight_other_gyeong,
        "예기": args.weight_other_gyeong,
    }
    
    print(f"\n⚖️ 가중치 설정:")
    print(f"   사서: {args.weight_saseo}x")
    print(f"   삼경: {args.weight_samgyeong}x")
    print(f"   기타 경전: {args.weight_other_gyeong}x")
    print(f"   史/集: 1.0x")
    
    # 장르별 마커 평균 계산
    src_cols = ["src_left", "src_right"]
    print("\n🔄 장르별 마커 평균 계산 중...")
    genre_means = compute_genre_marker_means(df_clusters, "genre", src_cols)
    
    # 잔차화 공동출현 행렬 생성
    print("🔄 잔차화 공동출현 행렬 생성 중...")
    C, groups, markers, g_counts, m_counts = build_residualized_cooccurrence(
        df_clusters, "genre", "parent_cluster_id", src_cols,
        genre_means, weight_map, args.min_count
    )
    
    print(f"   - 그룹 수: {len(groups)}")
    print(f"   - 마커 수: {len(markers)}")
    
    # 임베딩
    print("🔄 Joint Embedding 계산 중...")
    gxy, mxy = joint_embedding(C, args.svd_dim, args.seed, dim=2)
    
    # 장르 엔트로피 계산
    print("🔄 장르 엔트로피 계산 중...")
    entropy_scores = compute_genre_entropy(df_clusters, "genre", "parent_cluster_id")
    avg_entropy = np.mean(list(entropy_scores.values()))
    print(f"   - 평균 장르 엔트로피: {avg_entropy:.3f}")
    
    # 결과 저장
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 시각화
    title = f"Residualized Joint Embedding (사서:{args.weight_saseo}x, 삼경:{args.weight_samgyeong}x)"
    save_html_2d(
        args.out_dir / "joint_embedding_residualized_2d.html",
        groups, gxy, g_counts,
        markers, mxy, m_counts,
        args.label_top_markers, title
    )
    
    # 장르 평균 저장
    genre_means_records = []
    for genre, mdict in genre_means.items():
        for marker, mean in mdict.items():
            genre_means_records.append({"genre": genre, "marker": marker, "mean_freq": mean})
    pd.DataFrame(genre_means_records).to_csv(
        args.out_dir / "genre_marker_means.csv", index=False, encoding="utf-8-sig"
    )
    print(f"✅ Saved: {args.out_dir / 'genre_marker_means.csv'}")
    
    # 엔트로피 저장
    entropy_df = pd.DataFrame([
        {"group": g, "genre_entropy": e} for g, e in entropy_scores.items()
    ])
    entropy_df.to_csv(args.out_dir / "genre_entropy.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {args.out_dir / 'genre_entropy.csv'}")
    
    # 설정 저장
    config = {
        "input": str(args.input),
        "clusters": str(args.clusters),
        "genre_level": args.genre_level,
        "weight_saseo": args.weight_saseo,
        "weight_samgyeong": args.weight_samgyeong,
        "weight_other_gyeong": args.weight_other_gyeong,
        "min_count": args.min_count,
        "svd_dim": args.svd_dim,
        "seed": args.seed,
        "n_groups": len(groups),
        "n_markers": len(markers),
        "avg_genre_entropy": avg_entropy,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Saved: {args.out_dir / 'config.json'}")
    
    print(f"\n✅ 모든 결과 저장 완료: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
