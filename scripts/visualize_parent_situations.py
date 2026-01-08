from __future__ import annotations

import argparse
import csv
import json
import math
import regex as re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

_CJK_RANGES = "\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
_CJK_OPTIONAL_MARKER_RE = re.compile(rf"(?P<cjk>[{_CJK_RANGES}]+)(?P<marker>\p{{Hangul}}+)?")


def extract_markers(text: object, null_marker: str) -> list[str]:
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


def _load_merge_mapping(merge_groups_csv: Path | None) -> dict[str, str]:
    if merge_groups_csv is None or not merge_groups_csv.exists():
        return {}
    df = pd.read_csv(merge_groups_csv)
    if "rep" not in df.columns or "members" not in df.columns:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        rep = str(row["rep"]).strip()
        members = str(row["members"]).strip().split()
        if not rep or not members:
            continue
        for m in members:
            mapping[m] = rep
    return mapping


def _embed(mat: np.ndarray, method: str, seed: int) -> np.ndarray:
    # Hellinger transform for distributions
    X = np.sqrt(np.clip(mat, 0.0, 1.0))
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        init = PCA(n_components=2, random_state=seed).fit_transform(X)
        return TSNE(
            n_components=2,
            random_state=seed,
            init=init,
            learning_rate="auto",
            perplexity=5,  # only 15 parents
            n_iter=1500,
        ).fit_transform(X)
    raise SystemExit(f"지원하지 않는 method: {method}")


def _try_save_plotly_html(out_html: Path, df: pd.DataFrame, *, color_by: str) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception as e:
        raise SystemExit(f"plotly가 필요합니다: {e}")

    hover_top_markers = (
        df["top_markers"].astype(str)
        .str.replace(";", "<br>", regex=False)
        .str.replace("\t", " ", regex=False)
    )
    hover_top_markers_weighted = (
        df.get("top_markers_weighted", "").astype(str)
        .str.replace(";", "<br>", regex=False)
        .str.replace("\t", " ", regex=False)
    )
    hover_top_children = (
        df["top_children"].astype(str)
        .str.replace(";", "<br>", regex=False)
        .str.replace("\t", " ", regex=False)
    )

    if color_by == "dominant_tfidf":
        color_col = "dominant_marker_weighted" if "dominant_marker_weighted" in df.columns else "dominant_marker"
    else:
        color_col = "dominant_marker"

    if color_col == "dominant_marker_weighted":
        # weighted 컬럼이 있지만 비어있는 경우(가중치 미사용)에는 raw로 폴백
        if df["dominant_marker_weighted"].astype(str).str.len().sum() == 0:
            color_col = "dominant_marker"
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        size="rows",
        size_max=28,
        text="parent_label",
        hover_name="parent_label",
        hover_data={
            "rows": True,
            "parent_cluster_id": True,
            "x": ":.4f",
            "y": ":.4f",
        },
        title="Parent situations (marker-distribution embedding)",
    )
    fig.update_traces(textposition="top center", textfont_size=13, mode="markers+text")

    has_weighted = "top_markers_weighted" in df.columns
    if has_weighted:
        fig.update_traces(
            hovertemplate=(
                fig.data[0].hovertemplate
                + "<br><b>top_children</b><br>%{customdata[0]}"
                + "<br><b>top_markers(raw)</b><br>%{customdata[1]}"
                + "<br><b>top_markers(tfidf)</b><br>%{customdata[2]}"
                + "<extra></extra>"
            )
        )
        fig.update_traces(
            customdata=np.stack(
                [hover_top_children.to_numpy(), hover_top_markers.to_numpy(), hover_top_markers_weighted.to_numpy()],
                axis=1,
            )
        )
    else:
        fig.update_traces(
            hovertemplate=(
                fig.data[0].hovertemplate
                + "<br><b>top_children</b><br>%{customdata[0]}"
                + "<br><b>top_markers</b><br>%{customdata[1]}"
                + "<extra></extra>"
            )
        )
        fig.update_traces(customdata=np.stack([hover_top_children.to_numpy(), hover_top_markers.to_numpy()], axis=1))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "parent_cluster_id(상황 대분류)를 현토(통합 반영) 분포로 2D에 배치해 시각화한다. "
            "hover에는 parent별 상위 child/현토 리스트를 표시한다."
        )
    )
    ap.add_argument("--csv", type=Path, default=Path("reports/recluster_k16_child/reclustered.csv"))
    ap.add_argument("--src-cols", type=str, default="src_left,src_right")
    ap.add_argument("--null-marker", type=str, default="<NULL>")
    ap.add_argument(
        "--include-null",
        action="store_true",
        help="<NULL>을 분포 계산에 포함한다(기본은 제외).",
    )
    ap.add_argument(
        "--dominant-exclude",
        type=str,
        default="<NULL>",
        help="dominant(대표) marker 계산에서 제외할 marker들(쉼표 구분). 기본은 <NULL> 제외.",
    )
    ap.add_argument(
        "--merge-groups-csv",
        type=Path,
        default=Path("reports/hyeonto_mixture_morph_only/auto_merge_groups.csv"),
        help="현토 통합 그룹 CSV(auto_merge_groups.csv). 없으면 미적용.",
    )
    ap.add_argument(
        "--marker-vocab-top-n",
        type=int,
        default=400,
        help="feature로 쓸 marker vocabulary 상위 N개(전체 빈도 기준).",
    )
    ap.add_argument("--min-marker-count", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=25, help="hover에 표시할 상위 marker/child 개수")
    ap.add_argument(
        "--weighting",
        choices=["freq", "tfidf"],
        default="freq",
        help="marker feature 가중치. freq는 raw 빈도 기반 분포, tfidf는 parent별 특징성을 강조.",
    )
    ap.add_argument(
        "--color-by",
        choices=["dominant", "dominant_tfidf"],
        default="dominant",
        help="점 색상 기준. dominant는 raw 최빈 현토, dominant_tfidf는 tfidf 기준 최상위 현토.",
    )
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/parent_situation_viz"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = ["parent_cluster_id", "child_cluster_id"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"입력 CSV에 {c} 컬럼이 없습니다. 사용 가능: {sorted(df.columns)}")

    src_cols = [c.strip() for c in str(args.src_cols).split(",") if c.strip()]
    for c in src_cols:
        if c not in df.columns:
            raise SystemExit(f"src 컬럼 없음: {c}. 사용 가능: {sorted(df.columns)}")

    merge_map = _load_merge_mapping(args.merge_groups_csv if args.merge_groups_csv.exists() else None)

    dominant_exclude = {m.strip() for m in str(args.dominant_exclude).split(",") if m.strip()}

    parent_rows: Counter[int] = Counter()
    parent_child: dict[int, Counter[int]] = defaultdict(Counter)
    parent_markers: dict[int, Counter[str]] = defaultdict(Counter)
    marker_global: Counter[str] = Counter()

    for _, row in df.iterrows():
        p = int(row["parent_cluster_id"])
        c = int(row["child_cluster_id"])
        parent_rows[p] += 1
        parent_child[p][c] += 1

        markers: list[str] = []
        for col in src_cols:
            # <NULL>은 "CJK 조각마다"가 아니라 "해당 셀에 현토가 하나도 없을 때" 1회만 넣는다.
            ms = extract_markers(row.get(col), null_marker="")
            if ms:
                markers.extend(ms)
            elif bool(args.include_null) and str(args.null_marker):
                markers.append(str(args.null_marker))

        for m in markers:
            m2 = merge_map.get(m, m)
            if (not args.include_null) and m2 == "<NULL>":
                continue
            parent_markers[p][m2] += 1
            marker_global[m2] += 1

    # vocabulary
    vocab = [m for m, n in marker_global.most_common(int(args.marker_vocab_top_n)) if int(n) >= int(args.min_marker_count)]
    vocab = [m for m in vocab if args.include_null or m != "<NULL>"]
    if not vocab:
        raise SystemExit("marker vocabulary가 비었습니다. min-marker-count/marker-vocab-top-n을 조정하세요.")

    parents = sorted(parent_rows.keys())
    mat = np.zeros((len(parents), len(vocab)), dtype=np.float32)
    marker_to_j = {m: j for j, m in enumerate(vocab)}

    for i, p in enumerate(parents):
        counts = parent_markers[p]
        total = float(sum(counts.values()))
        if total <= 0.0:
            continue
        for m, c in counts.items():
            j = marker_to_j.get(m)
            if j is None:
                continue
            mat[i, j] = float(c)

    mat = normalize(mat, norm="l1", axis=1)

    weighted_mat: np.ndarray | None = None
    if str(args.weighting) == "tfidf":
        # parent-level tf-idf: tf = P(m|parent), idf = log((P+1)/(df+1)) + 1
        df_counts = np.count_nonzero(mat > 0.0, axis=0).astype(np.float32)
        P = float(mat.shape[0])
        idf = (np.log((P + 1.0) / (df_counts + 1.0)) + 1.0).astype(np.float32)
        weighted_mat = (mat * idf.reshape(1, -1)).astype(np.float32)
        weighted_mat = normalize(weighted_mat, norm="l1", axis=1)

    emb_mat = weighted_mat if weighted_mat is not None else mat
    coords = _embed(emb_mat, method=str(args.method), seed=int(args.seed))

    # build hover strings
    rows_out = []
    for i, p in enumerate(parents):
        top_children = ";".join([f"c{cid}:{cnt}" for cid, cnt in parent_child[p].most_common(int(args.top_k))])
        top_markers = ";".join([f"{m}:{cnt}" for m, cnt in parent_markers[p].most_common(int(args.top_k))])

        # dominant는 기본적으로 <NULL> 같은 범용 토큰을 제외한 값으로 잡는다.
        dominant_marker_raw = parent_markers[p].most_common(1)[0][0] if parent_markers[p] else ""
        dominant_marker = ""
        if parent_markers[p]:
            for m, _cnt in parent_markers[p].most_common():
                if m in dominant_exclude:
                    continue
                dominant_marker = m
                break
        if not dominant_marker:
            dominant_marker = dominant_marker_raw

        dominant_marker_weighted_raw = ""
        dominant_marker_weighted = ""
        top_markers_weighted = ""
        if weighted_mat is not None and len(vocab) > 0:
            w = weighted_mat[i]
            if np.any(w > 0):
                j_max = int(np.argmax(w))
                dominant_marker_weighted_raw = str(vocab[j_max])

                # weighted dominant도 <NULL> 제외
                for j in np.argsort(-w):
                    if float(w[int(j)]) <= 0:
                        break
                    cand = str(vocab[int(j)])
                    if cand in dominant_exclude:
                        continue
                    dominant_marker_weighted = cand
                    break
                if not dominant_marker_weighted:
                    dominant_marker_weighted = dominant_marker_weighted_raw

                top_js = np.argsort(-w)[: int(args.top_k)]
                parts = []
                for j in top_js:
                    score = float(w[j])
                    if score <= 0:
                        continue
                    parts.append(f"{vocab[int(j)]}:{score:.4f}")
                top_markers_weighted = ";".join(parts)

        rows_out.append(
            {
                "parent_cluster_id": p,
                "parent_label": f"p{p}",
                "rows": int(parent_rows[p]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "dominant_marker": str(dominant_marker),
                "dominant_marker_raw": str(dominant_marker_raw),
                "dominant_marker_weighted": dominant_marker_weighted,
                "dominant_marker_weighted_raw": dominant_marker_weighted_raw,
                "top_children": top_children,
                "top_markers": top_markers,
                "top_markers_weighted": top_markers_weighted,
            }
        )

    out_df = pd.DataFrame(rows_out).sort_values(by="parent_cluster_id")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "parent_embedding.csv"
    out_html = args.out_dir / "parent_embedding.html"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _try_save_plotly_html(out_html, out_df, color_by=str(args.color_by))

    cfg = {
        "csv": str(args.csv),
        "src_cols": src_cols,
        "null_marker": str(args.null_marker),
        "include_null": bool(args.include_null),
        "merge_groups_csv": str(args.merge_groups_csv),
        "marker_vocab_top_n": int(args.marker_vocab_top_n),
        "min_marker_count": int(args.min_marker_count),
        "top_k": int(args.top_k),
        "weighting": str(args.weighting),
        "color_by": str(args.color_by),
        "method": str(args.method),
        "seed": int(args.seed),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("rows", len(df))
    print("parents", len(parents))
    print("vocab", len(vocab))
    print("saved", str(out_csv))
    print("saved", str(out_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
