#!/usr/bin/env python3
"""Parent-현토(marker) 공동 임베딩(같은 좌표계) 생성

왜 필요한가?
- 기존 marker_embedding, parent_embedding은 서로 다른 특징공간/프로세스에서 각각 2D로 내려온 좌표라서,
  overlay는 "한 화면에 같이 보기"에는 좋지만 "같은 좌표계"라고 볼 수 없습니다.

이 스크립트는 다음 방식으로 공동 좌표계를 만듭니다.
1) (parent, marker_left) 공기행렬 C (P x M) 구성
2) PPMI(Positive PMI)로 가중치 변환
3) TruncatedSVD로 저차원 공동 잠재공간 k차원 생성
   - parent_emb = U S  (X @ V)
   - marker_emb = V S  (V^T에서 복원)
   -> parent와 marker가 같은 k차원 공간에 존재
4) parent+marker 전체를 합쳐 2D로 축소(UMAP/TSNE/PCA)

출력:
- reports/k16_analysis/parent_marker_joint_embedding.csv
- reports/k16_analysis/parent_marker_joint_embedding.html
- reports/k16_analysis/parent_marker_joint_embedding_config.json

사용 예:
  .\docker.ps1 python scripts/visualize_parent_marker_joint_embedding.py --method umap --svd-dim 32
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.graph_objects as go
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = WORKSPACE_ROOT / "reports"


def _reduce_2d(X: np.ndarray, *, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        # TSNE는 초기값/스케일에 민감하므로 PCA init을 사용
        init = PCA(n_components=2, random_state=seed).fit_transform(X)
        return TSNE(
            n_components=2,
            random_state=seed,
            init=init,
            learning_rate="auto",
            perplexity=min(30, max(5, (len(X) - 1) // 3)),
        ).fit_transform(X)
    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception as e:
            raise SystemExit(f"umap-learn이 설치되어 있지 않습니다: {e}")
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1).fit_transform(X)
    raise ValueError(f"지원하지 않는 method: {method}")


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ppmi(counts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """PPMI 변환.

    PMI(i,j) = log( p(i,j) / (p(i)p(j)) )
    PPMI = max(PMI, 0)
    """
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float32)

    p_ij = counts / (total + eps)
    p_i = counts.sum(axis=1, keepdims=True) / (total + eps)
    p_j = counts.sum(axis=0, keepdims=True) / (total + eps)

    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    ppmi = np.maximum(pmi, 0.0)
    return ppmi.astype(np.float32)


def build_cooccurrence(
    rows: List[dict],
    *,
    drop_empty: bool,
    empty_token: str,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """(parent, marker_left) 공기행렬 구성"""
    parents = sorted({int(r["parent"]) for r in rows if "parent" in r})

    # marker vocabulary
    markers_set = set()
    for r in rows:
        m = str(r.get("marker_left", ""))
        if not m:
            if drop_empty:
                continue
            m = empty_token
        markers_set.add(m)

    markers = sorted(markers_set)

    parent_to_idx = {p: i for i, p in enumerate(parents)}
    marker_to_idx = {m: j for j, m in enumerate(markers)}

    C = np.zeros((len(parents), len(markers)), dtype=np.int64)

    for r in tqdm(rows, desc="Count co-occurrence"):
        if "parent" not in r:
            continue
        p = int(r["parent"])
        m = str(r.get("marker_left", ""))
        if not m:
            if drop_empty:
                continue
            m = empty_token
        C[parent_to_idx[p], marker_to_idx[m]] += 1

    return C, parents, markers


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint embedding of parent & marker in a single space")

    ap.add_argument(
        "--source-jsonl",
        default=str(REPORTS_ROOT / "k16_analysis" / "pa_k16_child_classification.jsonl"),
        help="(parent, child, marker_left 포함) JSONL",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPORTS_ROOT / "k16_analysis"),
    )

    ap.add_argument("--method", choices=["pca", "tsne", "umap"], default="umap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--svd-dim", type=int, default=32)
    ap.add_argument("--drop-empty", action="store_true", help="marker_left가 빈 샘플 제외")
    ap.add_argument("--empty-token", default="<EMPTY>")

    ap.add_argument("--label-top-n-markers", type=int, default=40, help="라벨을 항상 표시할 현토 상위 N개")

    args = ap.parse_args()

    src = Path(args.source_jsonl)
    if not src.exists():
        print(f"❌ 파일 없음: {src}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(src)
    print(f"📄 로드: {len(rows):,}개")

    C, parents, markers = build_cooccurrence(rows, drop_empty=args.drop_empty, empty_token=args.empty_token)
    print(f"✅ 공기행렬: parents={len(parents)}, markers={len(markers)}, nnz={(C>0).sum():,}")

    X = _ppmi(C)

    svd_dim = min(args.svd_dim, min(X.shape) - 1) if min(X.shape) > 1 else 1
    if svd_dim < 2:
        print("❌ svd-dim이 너무 작습니다 (데이터 크기 부족)")
        return 1

    svd = TruncatedSVD(n_components=svd_dim, random_state=args.seed)
    # parent embedding: U S
    parent_emb = svd.fit_transform(X)  # shape (P, k)
    s = svd.singular_values_  # shape (k,)
    vt = svd.components_  # shape (k, M) = V^T
    # marker embedding: V S = (V^T)^T S
    marker_emb = (vt.T * s.reshape(1, -1)).astype(np.float32)  # (M, k)

    joint = np.vstack([parent_emb, marker_emb])
    coords = _reduce_2d(joint, method=args.method, seed=args.seed)

    parent_xy = coords[: len(parents)]
    marker_xy = coords[len(parents) :]

    # marker 빈도 (라벨 우선순위)
    marker_freq = Counter()
    for r in rows:
        m = str(r.get("marker_left", ""))
        if not m:
            if args.drop_empty:
                continue
            m = args.empty_token
        marker_freq[m] += 1

    top_markers = [m for m, _ in marker_freq.most_common(max(0, args.label_top_n_markers))]
    top_marker_set = set(top_markers)

    # CSV 저장 (공동 좌표계)
    out_csv = out_dir / "parent_marker_joint_embedding.csv"
    records = []

    for i, p in enumerate(parents):
        records.append({"type": "parent", "id": int(p), "label": f"parent_{p}", "x": float(parent_xy[i, 0]), "y": float(parent_xy[i, 1])})

    for j, m in enumerate(markers):
        records.append({"type": "marker", "id": j, "label": m, "x": float(marker_xy[j, 0]), "y": float(marker_xy[j, 1]), "count": int(marker_freq.get(m, 0))})

    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ 저장: {out_csv.name}")

    # Config 저장
    cfg = {
        "source_jsonl": str(src.relative_to(WORKSPACE_ROOT)) if src.is_absolute() else str(src),
        "method": args.method,
        "seed": args.seed,
        "svd_dim": int(svd_dim),
        "drop_empty": bool(args.drop_empty),
        "empty_token": args.empty_token,
        "label_top_n_markers": int(args.label_top_n_markers),
        "parents": len(parents),
        "markers": len(markers),
    }
    out_cfg = out_dir / "parent_marker_joint_embedding_config.json"
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plotly 시각화
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=parent_xy[:, 0],
            y=parent_xy[:, 1],
            mode="markers+text",
            text=[f"p{p}" for p in parents],
            textposition="top center",
            name="Parent",
            marker=dict(size=14, symbol="diamond"),
            hovertemplate="parent=%{text}<extra></extra>",
        )
    )

    # marker: 라벨 표시용(상위 N개)
    if args.label_top_n_markers > 0:
        idx_top = [i for i, m in enumerate(markers) if m in top_marker_set]
        if idx_top:
            fig.add_trace(
                go.Scatter(
                    x=marker_xy[idx_top, 0],
                    y=marker_xy[idx_top, 1],
                    mode="markers+text",
                    text=[markers[i] for i in idx_top],
                    textposition="top center",
                    name=f"Marker (top {len(idx_top)} labeled)",
                    marker=dict(size=8, opacity=0.9),
                    hovertemplate="marker=%{text}<extra></extra>",
                )
            )

    # marker: 전체
    fig.add_trace(
        go.Scatter(
            x=marker_xy[:, 0],
            y=marker_xy[:, 1],
            mode="markers",
            name="Marker",
            marker=dict(size=6, opacity=0.55),
            text=markers,
            customdata=[marker_freq.get(m, 0) for m in markers],
            hovertemplate="marker=%{text}<br>count=%{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Parent–Marker Joint Embedding (shared coordinate system)",
        xaxis_title=f"2D ({args.method})",
        yaxis_title=f"2D ({args.method})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=1200,
        height=800,
    )

    out_html = out_dir / "parent_marker_joint_embedding.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"✅ 저장: {out_html.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
