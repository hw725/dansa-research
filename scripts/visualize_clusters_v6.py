#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6 클러스터 결과를 시각화합니다.

v6 구조: cluster_id (단일 레벨, parent/child 없음)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

_CJK_RANGES = "\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
_CJK_OPTIONAL_MARKER_RE = re.compile(rf"(?P<cjk>[{_CJK_RANGES}]+)(?P<marker>\p{{Hangul}}+)?")


def extract_markers(text: object) -> list[str]:
    if text is None:
        return []
    s = str(text)
    if not s or s == "nan":
        return []
    out: list[str] = []
    for m in _CJK_OPTIONAL_MARKER_RE.finditer(s):
        marker = m.group("marker")
        if marker:
            out.append(marker)
    return out


def _embed(mat: np.ndarray, method: str, seed: int) -> np.ndarray:
    X = np.sqrt(np.clip(mat, 0.0, 1.0))
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        init = PCA(n_components=2, random_state=seed).fit_transform(X)
        perplexity = min(5, max(2, len(X) - 1))
        return TSNE(
            n_components=2,
            random_state=seed,
            init=init,
            learning_rate="auto",
            perplexity=perplexity,
            n_iter=1500,
        ).fit_transform(X)
    raise SystemExit(f"지원하지 않는 method: {method}")


def _try_save_plotly_html(out_html: Path, df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except Exception as e:
        print(f"plotly가 필요합니다: {e}")
        return

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="dominant_marker",
        size="rows",
        size_max=28,
        text="cluster_label",
        hover_name="cluster_label",
        hover_data={
            "rows": True,
            "cluster_id": True,
            "canonicity": ":.1f",
            "x": ":.4f",
            "y": ":.4f",
        },
        title="Cluster Visualization (marker-distribution embedding)",
    )
    fig.update_traces(textposition="top center", textfont_size=13, mode="markers+text")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> int:
    ap = argparse.ArgumentParser(description="V6 클러스터 시각화")
    ap.add_argument("--csv", type=Path, required=True, help="클러스터 CSV 경로")
    ap.add_argument("--src-cols", type=str, default="src_left,src_right")
    ap.add_argument("--marker-vocab-top-n", type=int, default=400)
    ap.add_argument("--min-marker-count", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    
    # cluster_id 컬럼 확인
    if "cluster_id" not in df.columns:
        raise SystemExit(f"cluster_id 컬럼이 없습니다. 사용 가능: {sorted(df.columns)}")

    src_cols = [c.strip() for c in str(args.src_cols).split(",") if c.strip()]
    for c in src_cols:
        if c not in df.columns:
            raise SystemExit(f"src 컬럼 없음: {c}. 사용 가능: {sorted(df.columns)}")

    # 사서 도서 목록
    CANON_BOOKS = ["논어", "맹자", "대학", "중용"]

    cluster_rows: Counter[int] = Counter()
    cluster_markers: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_canon: Counter[int] = Counter()
    marker_global: Counter[str] = Counter()

    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        cluster_rows[cid] += 1
        
        book_name = str(row.get("book_name", ""))
        if any(canon in book_name for canon in CANON_BOOKS):
            cluster_canon[cid] += 1

        markers: list[str] = []
        for col in src_cols:
            ms = extract_markers(row.get(col))
            markers.extend(ms)

        for m in markers:
            cluster_markers[cid][m] += 1
            marker_global[m] += 1

    # vocabulary
    vocab = [m for m, n in marker_global.most_common(int(args.marker_vocab_top_n)) if int(n) >= int(args.min_marker_count)]
    if not vocab:
        raise SystemExit("marker vocabulary가 비었습니다.")

    clusters = sorted(cluster_rows.keys())
    mat = np.zeros((len(clusters), len(vocab)), dtype=np.float32)
    marker_to_j = {m: j for j, m in enumerate(vocab)}

    for i, cid in enumerate(clusters):
        counts = cluster_markers[cid]
        total = float(sum(counts.values()))
        if total <= 0.0:
            continue
        for m, c in counts.items():
            j = marker_to_j.get(m)
            if j is None:
                continue
            mat[i, j] = float(c)

    mat = normalize(mat, norm="l1", axis=1)
    coords = _embed(mat, method=str(args.method), seed=int(args.seed))

    # build output rows
    rows_out = []
    for i, cid in enumerate(clusters):
        total = cluster_rows[cid]
        canon = cluster_canon[cid]
        canonicity = (canon / total) * 100 if total > 0 else 0.0
        
        top_markers = cluster_markers[cid].most_common(int(args.top_k))
        top_markers_str = "; ".join([f"{m}:{cnt}" for m, cnt in top_markers[:10]])
        dominant_marker = top_markers[0][0] if top_markers else ""

        rows_out.append({
            "cluster_id": cid,
            "cluster_label": f"p{cid}",
            "rows": int(total),
            "canonicity": canonicity,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "dominant_marker": dominant_marker,
            "top_markers": top_markers_str,
        })

    out_df = pd.DataFrame(rows_out).sort_values(by="cluster_id")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "cluster_embedding.csv"
    out_html = args.out_dir / "cluster_embedding.html"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _try_save_plotly_html(out_html, out_df)

    cfg = {
        "csv": str(args.csv),
        "src_cols": src_cols,
        "marker_vocab_top_n": int(args.marker_vocab_top_n),
        "min_marker_count": int(args.min_marker_count),
        "top_k": int(args.top_k),
        "method": str(args.method),
        "seed": int(args.seed),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows: {len(df)}")
    print(f"clusters: {len(clusters)}")
    print(f"vocab: {len(vocab)}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
