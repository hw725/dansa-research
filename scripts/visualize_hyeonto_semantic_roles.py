from __future__ import annotations

import argparse
import json
import math
import regex as re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# NOTE: 옛한글(자모/확장)까지 포함하려면 표준 re가 아니라 `regex`의 유니코드 프로퍼티가 안전하다.
# - \p{Han}    : 한자(통합 한자 계열)
# - \p{Hangul} : 한글(음절/자모/확장 포함)
_CJK_OPTIONAL_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")


@dataclass(frozen=True)
class GroupKey:
    parent: int
    child: int

    def to_parent(self) -> str:
        return f"p{self.parent}"

    def to_parent_child(self) -> str:
        return f"p{self.parent}_c{self.child}"


@dataclass(frozen=True)
class GroupEmbedRow:
    group_level: str
    group: str
    parent_cluster_id: int
    child_cluster_id: int


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


def _parse_normalize_pairs(s: str) -> dict[str, str]:
    # format: "이라=라,으로=로"
    out: dict[str, str] = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


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
        elif null_marker:
            out.append(null_marker)
    return out


def _ensure_parent_child_cols(df: pd.DataFrame) -> None:
    needed = ["parent_cluster_id", "child_cluster_id"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"입력 CSV에 {c} 컬럼이 없습니다. 사용 가능: {sorted(df.columns)}")


def _make_feature_matrix(
    df: pd.DataFrame,
    src_cols: list[str],
    null_marker: str,
    normalize_pairs: dict[str, str],
    merge_map: dict[str, str],
    group_level: str,
    min_count: int,
    exclude_markers: set[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    _ensure_parent_child_cols(df)

    marker_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    marker_total_count: Counter[str] = Counter()
    marker_doc_freq: Counter[str] = Counter()
    corpus_group_counts: Counter[str] = Counter()

    for _, row in df.iterrows():
        gk = GroupKey(parent=int(row["parent_cluster_id"]), child=int(row["child_cluster_id"]))
        group = gk.to_parent() if group_level == "parent" else gk.to_parent_child()

        row_markers: list[str] = []
        for c in src_cols:
            row_markers.extend(extract_markers(row.get(c), null_marker=null_marker))

        if not row_markers:
            continue

        # normalize within the row (count-based totals still use multiplicity)
        normalized_row_markers: list[str] = []
        normalized_row_marker_set: set[str] = set()
        for m in row_markers:
            m2 = normalize_pairs.get(m, m)
            m2 = merge_map.get(m2, m2)
            if m2 in exclude_markers:
                continue
            normalized_row_markers.append(m2)
            normalized_row_marker_set.add(m2)

        for m2 in normalized_row_markers:
            marker_group_counts[m2][group] += 1
            marker_total_count[m2] += 1
            corpus_group_counts[group] += 1
        for m2 in normalized_row_marker_set:
            marker_doc_freq[m2] += 1

    # filter
    markers = [m for m, n in marker_total_count.items() if int(n) >= int(min_count)]
    markers = sorted(markers, key=lambda m: (-int(marker_total_count[m]), m))

    # columns
    groups = sorted({g for m in markers for g in marker_group_counts[m].keys()})
    if not groups:
        raise SystemExit("그룹이 비어 있습니다. 입력/옵션을 확인하세요.")

    group_to_j = {g: j for j, g in enumerate(groups)}

    mat = np.zeros((len(markers), len(groups)), dtype=np.float32)
    for i, m in enumerate(markers):
        total = float(sum(marker_group_counts[m].values()))
        if total <= 0.0:
            continue
        for g, c in marker_group_counts[m].items():
            j = group_to_j.get(g)
            if j is None:
                continue
            mat[i, j] = float(c)

    # to probability distribution
    mat = normalize(mat, norm="l1", axis=1)

    # corpus group distribution (for reweighting)
    corpus_total = float(sum(corpus_group_counts.values()))
    if corpus_total <= 0.0:
        corpus_p = np.full((len(groups),), 1.0 / float(len(groups)), dtype=np.float32)
    else:
        corpus_p = np.array([float(corpus_group_counts.get(g, 0)) / corpus_total for g in groups], dtype=np.float32)

    meta_rows = []
    for m in markers:
        gcounts = marker_group_counts[m]
        # top parents/groups
        top = sorted(gcounts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top_str = ";".join([f"{g}:{c}" for g, c in top])
        meta_rows.append(
            {
                "marker": m,
                "count": int(marker_total_count[m]),
                "doc_freq": int(marker_doc_freq[m]),
                "num_groups": int(len(gcounts)),
                "top_groups": top_str,
            }
        )

    meta = pd.DataFrame(meta_rows)
    return meta, mat, groups, corpus_p


def _make_group_feature_matrix(
    df: pd.DataFrame,
    src_cols: list[str],
    null_marker: str,
    normalize_pairs: dict[str, str],
    merge_map: dict[str, str],
    group_level: str,
    min_count: int,
    exclude_markers: set[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    """Build group->marker distribution matrix.

    Rows: groups (parent or parent_child)
    Cols: markers
    """
    _ensure_parent_child_cols(df)

    group_marker_counts: dict[GroupEmbedRow, Counter[str]] = defaultdict(Counter)
    marker_total_count: Counter[str] = Counter()
    group_total_count: Counter[GroupEmbedRow] = Counter()

    for _, row in df.iterrows():
        parent = int(row["parent_cluster_id"])
        child = int(row["child_cluster_id"])
        if group_level == "parent":
            g = GroupEmbedRow(group_level="parent", group=f"p{parent}", parent_cluster_id=parent, child_cluster_id=-1)
        else:
            g = GroupEmbedRow(
                group_level="parent_child",
                group=f"p{parent}_c{child}",
                parent_cluster_id=parent,
                child_cluster_id=child,
            )

        row_markers: list[str] = []
        for c in src_cols:
            row_markers.extend(extract_markers(row.get(c), null_marker=null_marker))
        if not row_markers:
            continue

        normalized_row_markers: list[str] = []
        for m in row_markers:
            m2 = normalize_pairs.get(m, m)
            m2 = merge_map.get(m2, m2)
            if m2 in exclude_markers:
                continue
            normalized_row_markers.append(m2)

        for m2 in normalized_row_markers:
            group_marker_counts[g][m2] += 1
            marker_total_count[m2] += 1
            group_total_count[g] += 1

    markers = [m for m, n in marker_total_count.items() if int(n) >= int(min_count)]
    markers = sorted(markers, key=lambda m: (-int(marker_total_count[m]), m))
    if not markers:
        raise SystemExit("marker가 비어 있습니다. --min-count 또는 exclude/정규화 옵션을 확인하세요.")

    groups = sorted(group_total_count.keys(), key=lambda r: (r.parent_cluster_id, r.child_cluster_id, r.group))
    if not groups:
        raise SystemExit("그룹이 비어 있습니다. 입력/옵션을 확인하세요.")

    marker_to_j = {m: j for j, m in enumerate(markers)}

    mat = np.zeros((len(groups), len(markers)), dtype=np.float32)
    for i, g in enumerate(groups):
        counts = group_marker_counts[g]
        for m, c in counts.items():
            j = marker_to_j.get(m)
            if j is None:
                continue
            mat[i, j] = float(c)

    # to probability distribution p(marker|group)
    mat = normalize(mat, norm="l1", axis=1)

    # corpus marker distribution p(marker)
    total = float(sum(marker_total_count[m] for m in markers))
    if total <= 0.0:
        corpus_p = np.full((len(markers),), 1.0 / float(len(markers)), dtype=np.float32)
    else:
        corpus_p = np.array([float(marker_total_count[m]) / total for m in markers], dtype=np.float32)

    meta_rows = []
    eps = 1e-12
    for i, g in enumerate(groups):
        row_p = mat[i]
        lift = row_p / np.clip(corpus_p, eps, None)
        top_lift_idx = np.argsort(lift)[::-1][:10]
        top_count_idx = np.argsort(row_p)[::-1][:10]
        top_lift = " ".join([f"{markers[j]}" for j in top_lift_idx if row_p[j] > 0])
        top_count = " ".join([f"{markers[j]}" for j in top_count_idx if row_p[j] > 0])
        meta_rows.append(
            {
                "group_level": g.group_level,
                "group": g.group,
                "parent_cluster_id": int(g.parent_cluster_id),
                "child_cluster_id": int(g.child_cluster_id),
                "row_count": int(group_total_count[g]),
                "top_markers_lift": top_lift,
                "top_markers": top_count,
            }
        )

    meta = pd.DataFrame(meta_rows)
    return meta, mat, markers, corpus_p


def _make_group_marker_count_matrix(
    df: pd.DataFrame,
    src_cols: list[str],
    null_marker: str,
    normalize_pairs: dict[str, str],
    merge_map: dict[str, str],
    group_level: str,
    min_count: int,
    exclude_markers: set[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build raw count matrix: rows=groups, cols=markers.

    This is used for CA biplot where row/col points share one coordinate system.
    """
    _ensure_parent_child_cols(df)

    group_marker_counts: dict[GroupEmbedRow, Counter[str]] = defaultdict(Counter)
    marker_total_count: Counter[str] = Counter()
    group_total_count: Counter[GroupEmbedRow] = Counter()

    for _, row in df.iterrows():
        parent = int(row["parent_cluster_id"])
        child = int(row["child_cluster_id"])
        if group_level == "parent":
            g = GroupEmbedRow(group_level="parent", group=f"p{parent}", parent_cluster_id=parent, child_cluster_id=-1)
        else:
            g = GroupEmbedRow(
                group_level="parent_child",
                group=f"p{parent}_c{child}",
                parent_cluster_id=parent,
                child_cluster_id=child,
            )

        row_markers: list[str] = []
        for c in src_cols:
            row_markers.extend(extract_markers(row.get(c), null_marker=null_marker))
        if not row_markers:
            continue

        normalized_row_markers: list[str] = []
        for m in row_markers:
            m2 = normalize_pairs.get(m, m)
            m2 = merge_map.get(m2, m2)
            if m2 in exclude_markers:
                continue
            normalized_row_markers.append(m2)

        for m2 in normalized_row_markers:
            group_marker_counts[g][m2] += 1
            marker_total_count[m2] += 1
            group_total_count[g] += 1

    markers = [m for m, n in marker_total_count.items() if int(n) >= int(min_count)]
    markers = sorted(markers, key=lambda m: (-int(marker_total_count[m]), m))
    if not markers:
        raise SystemExit("marker가 비어 있습니다. --min-count 또는 exclude/정규화 옵션을 확인하세요.")

    groups = sorted(group_total_count.keys(), key=lambda r: (r.parent_cluster_id, r.child_cluster_id, r.group))
    if not groups:
        raise SystemExit("그룹이 비어 있습니다. 입력/옵션을 확인하세요.")

    marker_to_j = {m: j for j, m in enumerate(markers)}
    mat = np.zeros((len(groups), len(markers)), dtype=np.float32)
    for i, g in enumerate(groups):
        counts = group_marker_counts[g]
        for m, c in counts.items():
            j = marker_to_j.get(m)
            if j is None:
                continue
            mat[i, j] = float(c)

    meta_rows = []
    eps = 1e-12
    corpus_total = float(sum(marker_total_count[m] for m in markers))
    if corpus_total <= 0.0:
        corpus_p = np.full((len(markers),), 1.0 / float(len(markers)), dtype=np.float32)
    else:
        corpus_p = np.array([float(marker_total_count[m]) / corpus_total for m in markers], dtype=np.float32)

    for i, g in enumerate(groups):
        row_count = int(group_total_count[g])
        row = mat[i]
        if row.sum() <= 0.0:
            top_lift = ""
            top_count = ""
        else:
            row_p = row / float(row.sum())
            lift = row_p / np.clip(corpus_p, eps, None)
            top_lift_idx = np.argsort(lift)[::-1][:10]
            top_count_idx = np.argsort(row_p)[::-1][:10]
            top_lift = " ".join([f"{markers[j]}" for j in top_lift_idx if row_p[j] > 0])
            top_count = " ".join([f"{markers[j]}" for j in top_count_idx if row_p[j] > 0])
        meta_rows.append(
            {
                "group_level": g.group_level,
                "group": g.group,
                "parent_cluster_id": int(g.parent_cluster_id),
                "child_cluster_id": int(g.child_cluster_id),
                "row_count": row_count,
                "top_markers_lift": top_lift,
                "top_markers": top_count,
            }
        )

    meta = pd.DataFrame(meta_rows)
    return meta, mat, markers


def _ca_biplot(mat_counts: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correspondence Analysis biplot coordinates.

    Returns (row_coords, col_coords, singular_values) where row/col points are comparable.
    """
    if mat_counts.size == 0 or float(mat_counts.sum()) <= 0.0:
        raise SystemExit("CA 입력 행렬이 비어 있습니다.")

    N = mat_counts.astype(np.float64, copy=False)
    P = N / float(N.sum())
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    Dr_inv_sqrt = 1.0 / np.sqrt(np.clip(r, eps, None))
    Dc_inv_sqrt = 1.0 / np.sqrt(np.clip(c, eps, None))

    # standardized residuals
    S = (P - r[:, None] * c[None, :]) * (Dr_inv_sqrt[:, None] * Dc_inv_sqrt[None, :])

    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    k = 2
    Uk = U[:, :k]
    Vk = Vt.T[:, :k]
    sk = s[:k]

    row_coords = (Uk * sk[None, :]) * Dr_inv_sqrt[:, None]
    col_coords = (Vk * sk[None, :]) * Dc_inv_sqrt[:, None]
    return row_coords.astype(np.float32), col_coords.astype(np.float32), sk.astype(np.float32)


def _try_save_plotly_biplot(out_html: Path, points: pd.DataFrame, title: str) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return

    work = points.copy()
    fig = px.scatter(
        work,
        x="x",
        y="y",
        color="kind",
        symbol="kind",
        size="size",
        size_max=22,
        text="label",
        hover_name="name",
        hover_data={
            "kind": True,
            "parent_cluster_id": True,
            "child_cluster_id": True,
            "row_count": True,
            "marker_count": True,
            "top_markers_lift": True,
            "x": ":.4f",
            "y": ":.4f",
        },
        title=title,
    )
    fig.update_traces(textposition="top center", textfont_size=10, mode="markers+text")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _try_save_plotly_group_embedding(out_html: Path, df: pd.DataFrame, title: str) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return

    work = df.copy()
    # compact text labels: only show for larger groups
    work["label"] = work["group"].where(work["row_count"] >= work["row_count"].quantile(0.8), "")

    fig = px.scatter(
        work,
        x="x",
        y="y",
        color="parent_cluster_id",
        symbol="group_level",
        size="row_count",
        size_max=22,
        text="label",
        hover_name="group",
        hover_data={
            "group_level": True,
            "parent_cluster_id": True,
            "child_cluster_id": True,
            "row_count": True,
            "top_markers_lift": True,
            "top_markers": True,
            "x": ":.4f",
            "y": ":.4f",
        },
        title=title,
    )
    fig.update_traces(textposition="top center", textfont_size=10, mode="markers+text")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _write_group_report(out_md: Path, df: pd.DataFrame, reclustered: pd.DataFrame, group_level: str, examples: int) -> None:
    """Create a readable report that explains why groups differ."""
    work = df.sort_values(["row_count"], ascending=False).copy()
    lines: list[str] = []
    lines.append(f"# group profiles ({group_level})\n")
    lines.append("- 설명 기준: marker 분포의 lift(과대표) + 대표 예문")
    lines.append("- 주의: 라벨은 확정이 아니라 귀납적 설명 초안\n")

    def _examples_for(parent: int, child: int) -> pd.DataFrame:
        if group_level == "parent":
            sub = reclustered.loc[reclustered.parent_cluster_id == parent]
        else:
            sub = reclustered.loc[(reclustered.parent_cluster_id == parent) & (reclustered.child_cluster_id == child)]
        return sub.sort_values(["book_name", "paragraph_id", "left_sentence_id"], kind="mergesort").head(int(examples))

    for _, r in work.iterrows():
        parent = int(r["parent_cluster_id"])
        child = int(r["child_cluster_id"]) if group_level != "parent" else -1
        group = str(r["group"])
        lines.append(f"## {group} (rows={int(r['row_count'])})")
        lines.append(f"- top_markers_lift: {str(r.get('top_markers_lift',''))}")
        lines.append(f"- top_markers: {str(r.get('top_markers',''))}")
        ex = _examples_for(parent, child)
        for _, e in ex.iterrows():
            lines.append(f"- book={e.get('book_name','')}, para={e.get('paragraph_id','')}, sent={e.get('left_sentence_id','')}→{e.get('right_sentence_id','')}")
            lines.append(f"  - src_L: {str(e.get('src_left',''))}")
            lines.append(f"  - src_R: {str(e.get('src_right',''))}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _reweight_by_corpus(mat: np.ndarray, corpus_p: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return mat
    if mode != "lift":
        raise SystemExit(f"지원하지 않는 reweight mode: {mode}")

    eps = 1e-12
    denom = np.clip(corpus_p, eps, None)
    out = mat / denom[None, :]
    out = normalize(out, norm="l1", axis=1)
    return out


def _embed(mat: np.ndarray, method: str, seed: int) -> np.ndarray:
    # Hellinger transform helps for distributions
    X = np.sqrt(np.clip(mat, 0.0, 1.0))

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        # TSNE is sensitive; PCA init stabilizes
        init = PCA(n_components=2, random_state=seed).fit_transform(X)
        return TSNE(
            n_components=2,
            random_state=seed,
            init=init,
            learning_rate="auto",
            perplexity=30,
            n_iter=1000,
        ).fit_transform(X)
    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"umap-learn이 설치되어 있지 않습니다: {e}")
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1).fit_transform(X)
    raise SystemExit(f"지원하지 않는 method: {method}")


def _configure_matplotlib_korean_font() -> None:
    """Prefer Korean-capable fonts if available.

    Docker 이미지에 한글 폰트가 설치되어 있어도(matplotlib 기본값은 DejaVu) 명시하지 않으면
    PNG 생성 시 글리프 누락 경고가 날 수 있다.
    """
    try:
        import matplotlib as mpl  # type: ignore
        from matplotlib import font_manager  # type: ignore
    except Exception:
        return

    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Noto Sans CJK",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in available), None)
    if chosen:
        mpl.rcParams["font.family"] = [chosen]
    mpl.rcParams["axes.unicode_minus"] = False


def _try_save_plot(out_png: Path, df: pd.DataFrame, label_top_n: int) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    _configure_matplotlib_korean_font()

    # color by dominant parent/group
    dom = df["dominant"].astype(str)
    uniq = sorted(dom.unique())
    color_map = {k: i for i, k in enumerate(uniq)}
    colors = [color_map[k] for k in dom]

    sizes = [max(6.0, 2.5 * math.log1p(float(c))) for c in df["count"].tolist()]

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df["x"], df["y"], c=colors, s=sizes, alpha=0.7)

    # label top-N frequent markers
    top_df = df.sort_values(by="count", ascending=False).head(int(label_top_n))
    for _, r in top_df.iterrows():
        plt.text(float(r["x"]), float(r["y"]), str(r["marker"]), fontsize=8)

    plt.title("Hyeonto semantic roles (parent distribution embedding)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _try_save_plotly_html(out_html: Path, df: pd.DataFrame, label_top_n_html: int) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return

    work = df.copy()
    label_top_n_html = int(label_top_n_html)
    if label_top_n_html > 0:
        top_markers = set(work.sort_values(by="count", ascending=False).head(label_top_n_html)["marker"].astype(str).tolist())
        work["label"] = work["marker"].astype(str).where(work["marker"].astype(str).isin(top_markers), "")
    else:
        work["label"] = work["marker"].astype(str)

    # hover에서 top_groups가 길 수 있어 줄바꿈을 넣어준다(HTML 표시용)
    hover_text = (
        work["top_groups"].astype(str)
        .str.replace(";", "<br>", regex=False)
        .str.replace("\t", " ", regex=False)
    )

    fig = px.scatter(
        work,
        x="x",
        y="y",
        color="dominant",
        size="count",
        size_max=22,
        text="label",
        hover_name="marker",
        hover_data={
            "count": True,
            "doc_freq": True,
            "num_groups": True,
            "dominant": True,
            "x": ":.4f",
            "y": ":.4f",
        },
        title="Hyeonto semantic roles (distribution embedding)",
    )

    # 라벨이 겹칠 수 있으니 크기를 줄이고, 점+텍스트 모드로
    fig.update_traces(textposition="top center", textfont_size=10, mode="markers+text")

    # 추가 hover 내용(상위 그룹들)
    fig.update_traces(hovertemplate=fig.data[0].hovertemplate + "<br><b>top_groups</b><br>%{customdata[0]}<extra></extra>")
    fig.update_traces(customdata=np.stack([hover_text.to_numpy()], axis=1))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _compute_embedding_df(
    *,
    df: pd.DataFrame,
    src_cols: list[str],
    null_marker: str,
    normalize_pairs: dict[str, str],
    merge_map: dict[str, str],
    group_level: str,
    min_count: int,
    exclude_markers: set[str],
    reweight_by_corpus: str,
    method: str,
    seed: int,
) -> pd.DataFrame:
    meta, mat, groups, corpus_p = _make_feature_matrix(
        df=df,
        src_cols=src_cols,
        null_marker=null_marker,
        normalize_pairs=normalize_pairs,
        merge_map=merge_map,
        group_level=group_level,
        min_count=min_count,
        exclude_markers=exclude_markers,
    )

    mat2 = _reweight_by_corpus(mat, corpus_p=corpus_p, mode=reweight_by_corpus)
    coords = _embed(mat2, method=method, seed=seed)

    dominant_idx = np.argmax(mat2, axis=1)
    dominant = [groups[int(i)] for i in dominant_idx]

    out = meta.copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out["dominant"] = dominant
    out["method"] = method
    out["group_level"] = group_level
    out["reweight_by_corpus"] = reweight_by_corpus
    return out


def _try_save_plotly_html_parent_vs_parent_child(out_html: Path, parent: pd.DataFrame, parent_child: pd.DataFrame, label_top_n_html: int) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return

    work_parent = parent.copy()
    work_parent_child = parent_child.copy()
    work_parent["view"] = "parent"
    work_parent_child["view"] = "parent_child"
    work = pd.concat([work_parent, work_parent_child], ignore_index=True)

    label_top_n_html = int(label_top_n_html)
    if label_top_n_html > 0:
        top_markers = set(work.sort_values(by="count", ascending=False).head(label_top_n_html)["marker"].astype(str).tolist())
        work["label"] = work["marker"].astype(str).where(work["marker"].astype(str).isin(top_markers), "")
    else:
        work["label"] = work["marker"].astype(str)

    hover_text = work["top_groups"].astype(str).str.replace(";", "<br>", regex=False).str.replace("\t", " ", regex=False)

    fig = px.scatter(
        work,
        x="x",
        y="y",
        color="dominant",
        size="count",
        size_max=22,
        text="label",
        facet_col="view",
        hover_name="marker",
        hover_data={
            "count": True,
            "doc_freq": True,
            "num_groups": True,
            "dominant": True,
            "view": True,
            "x": ":.4f",
            "y": ":.4f",
        },
        title="Hyeonto semantic roles: parent vs parent_child (distribution embedding)",
    )
    fig.update_traces(textposition="top center", textfont_size=10, mode="markers+text")
    fig.update_traces(hovertemplate=fig.data[0].hovertemplate + "<br><b>top_groups</b><br>%{customdata[0]}<extra></extra>")
    fig.update_traces(customdata=np.stack([hover_text.to_numpy()], axis=1))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("view=", "")))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "통합(merge)된 현토를 parent(또는 parent_child) 분포 기반으로 2D에 배치해 시각화/분석한다. "
            "기본은 parent 기준(PCA)이며, 결과는 CSV(좌표+메타)로 저장된다."
        )
    )
    ap.add_argument("--csv", type=Path, default=Path("reports/recluster_k16_child/reclustered.csv"))
    ap.add_argument("--src-cols", type=str, default="src_left,src_right")
    ap.add_argument("--in-dir", type=Path, default=Path("reports/hyeonto_mixture_morph_only"), help="병합 결과(선택): auto_merge_groups.csv가 있으면 적용")
    ap.add_argument("--merge-groups-csv", type=Path, default=None, help="명시적 병합 그룹 CSV(auto_merge_groups.csv). 미지정 시 in-dir에서 탐색")
    ap.add_argument("--normalize-pairs", type=str, default="", help="수동 통합 매핑. 예: '이라=라,으로=로'")
    ap.add_argument("--null-marker", type=str, default="", help="(선택) 현토 없음 토큰. 예: '<NULL>'")
    ap.add_argument("--group-level", choices=["parent", "parent_child"], default="parent")
    ap.add_argument("--include-null", action="store_true", help="<NULL> 현토도 시각화에 포함한다(기본은 제외)")
    ap.add_argument("--exclude-markers", type=str, default="<NULL>", help="시각화에서 제외할 marker들(쉼표 구분). 기본은 <NULL> 제외")
    ap.add_argument("--min-count", type=int, default=50, help="이 빈도 미만 marker는 제외")
    ap.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--reweight-by-corpus",
        choices=["none", "lift"],
        default="none",
        help=(
            "현토 분포를 코퍼스 전체 분포로 재가중치한다. "
            "lift는 p(group|marker)/p(group)로 희귀 situation을 상대적으로 강조해 'p9 편향'을 완화하는 데 유용."
        ),
    )
    ap.add_argument("--label-top-n", type=int, default=30, help="PNG에 라벨링할 상위 N개 marker")
    ap.add_argument("--save-html", action="store_true", help="Plotly가 설치되어 있으면 HTML 인터랙티브 산출물을 저장")
    ap.add_argument(
        "--save-parent-child-comparison",
        action="store_true",
        help="parent / parent_child 임베딩을 각각 계산하고, 두 패널로 동시에 비교하는 HTML을 추가 생성한다.",
    )
    ap.add_argument(
        "--embed-groups",
        action="store_true",
        help="marker가 아니라 (parent 또는 parent_child) 그룹 자체를 marker 분포로 임베딩/시각화한다.",
    )
    ap.add_argument(
        "--biplot-group-marker",
        action="store_true",
        help=(
            "그룹(=parent/parent_child)과 marker를 같은 좌표평면에 함께 표시한다. "
            "group×marker count 행렬에 대해 대응분석(CA) biplot을 수행한다."
        ),
    )
    ap.add_argument(
        "--biplot-label-top-markers",
        type=int,
        default=40,
        help="biplot에서 라벨로 표시할 상위 marker 수(빈도 기준).",
    )
    ap.add_argument(
        "--biplot-label-top-groups",
        type=int,
        default=40,
        help="biplot에서 라벨로 표시할 상위 그룹 수(row_count 기준).",
    )
    ap.add_argument(
        "--group-examples",
        type=int,
        default=6,
        help="그룹 설명 리포트에 포함할 대표 예문 수",
    )
    ap.add_argument(
        "--label-top-n-html",
        type=int,
        default=40,
        help="HTML에서 hover 없이도 보이도록 라벨링할 상위 N개 marker(0이면 전부 라벨)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("reports/hyeonto_semantic_viz_parent"))
    args = ap.parse_args()

    src_cols = [c.strip() for c in str(args.src_cols).split(",") if c.strip()]
    exclude_markers = {m.strip() for m in str(args.exclude_markers).split(",") if m.strip()}
    if bool(args.include_null) and "<NULL>" in exclude_markers:
        exclude_markers.remove("<NULL>")

    df = pd.read_csv(args.csv)

    merge_groups_csv = args.merge_groups_csv
    if merge_groups_csv is None:
        cand = args.in_dir / "auto_merge_groups.csv"
        merge_groups_csv = cand if cand.exists() else None

    merge_map = _load_merge_mapping(merge_groups_csv)
    normalize_pairs = _parse_normalize_pairs(str(args.normalize_pairs))

    out = _compute_embedding_df(
        df=df,
        src_cols=src_cols,
        null_marker=str(args.null_marker),
        normalize_pairs=normalize_pairs,
        merge_map=merge_map,
        group_level=str(args.group_level),
        min_count=int(args.min_count),
        exclude_markers=exclude_markers,
        reweight_by_corpus=str(args.reweight_by_corpus),
        method=str(args.method),
        seed=int(args.seed),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "marker_semantic_embedding.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # optional plot (only if matplotlib is installed)
    out_png = args.out_dir / "marker_semantic_embedding.png"
    _try_save_plot(out_png, out, label_top_n=int(args.label_top_n))

    out_html = args.out_dir / "marker_semantic_embedding.html"
    if bool(args.save_html):
        _try_save_plotly_html(out_html, out, label_top_n_html=int(args.label_top_n_html))

    if bool(args.save_parent_child_comparison):
        out_parent = _compute_embedding_df(
            df=df,
            src_cols=src_cols,
            null_marker=str(args.null_marker),
            normalize_pairs=normalize_pairs,
            merge_map=merge_map,
            group_level="parent",
            min_count=int(args.min_count),
            exclude_markers=exclude_markers,
            reweight_by_corpus=str(args.reweight_by_corpus),
            method=str(args.method),
            seed=int(args.seed),
        )
        out_parent_child = _compute_embedding_df(
            df=df,
            src_cols=src_cols,
            null_marker=str(args.null_marker),
            normalize_pairs=normalize_pairs,
            merge_map=merge_map,
            group_level="parent_child",
            min_count=int(args.min_count),
            exclude_markers=exclude_markers,
            reweight_by_corpus=str(args.reweight_by_corpus),
            method=str(args.method),
            seed=int(args.seed),
        )

        out_parent.to_csv(args.out_dir / "marker_semantic_embedding_parent.csv", index=False, encoding="utf-8-sig")
        out_parent_child.to_csv(args.out_dir / "marker_semantic_embedding_parent_child.csv", index=False, encoding="utf-8-sig")

        if bool(args.save_html):
            _try_save_plotly_html_parent_vs_parent_child(
                args.out_dir / "marker_semantic_embedding_parent_vs_parent_child.html",
                out_parent,
                out_parent_child,
                label_top_n_html=int(args.label_top_n_html),
            )

    # Optional: embed groups (parent / parent_child) onto 2D plane
    if bool(args.embed_groups):
        reclustered_df = df
        group_levels = ["parent", "parent_child"]
        all_group_embeds: list[pd.DataFrame] = []
        for gl in group_levels:
            gmeta, gmat, markers2, marker_corpus_p = _make_group_feature_matrix(
                df=reclustered_df,
                src_cols=src_cols,
                null_marker=str(args.null_marker),
                normalize_pairs=normalize_pairs,
                merge_map=merge_map,
                group_level=gl,
                min_count=int(args.min_count),
                exclude_markers=exclude_markers,
            )
            coords = _embed(gmat, method=str(args.method), seed=int(args.seed))
            gout = gmeta.copy()
            gout["x"] = coords[:, 0]
            gout["y"] = coords[:, 1]
            gout["method"] = str(args.method)
            gout["min_count"] = int(args.min_count)
            gout["marker_count"] = int(len(markers2))
            all_group_embeds.append(gout)

            out_csv_g = args.out_dir / f"group_embedding_{gl}.csv"
            gout.to_csv(out_csv_g, index=False, encoding="utf-8-sig")

            if bool(args.save_html):
                _try_save_plotly_group_embedding(
                    args.out_dir / f"group_embedding_{gl}.html",
                    gout,
                    title=f"Group embedding ({gl}) by marker distribution",
                )

            _write_group_report(
                args.out_dir / f"group_profiles_{gl}.md",
                gout,
                reclustered=reclustered_df,
                group_level=gl,
                examples=int(args.group_examples),
            )

        if bool(args.save_html) and all_group_embeds:
            both = pd.concat(all_group_embeds, ignore_index=True)
            _try_save_plotly_group_embedding(
                args.out_dir / "group_embedding_parent_vs_parent_child.html",
                both,
                title="Group embedding: parent vs parent_child",
            )

    # Optional: CA biplot where groups + markers share the same coordinates
    if bool(args.biplot_group_marker) and bool(args.save_html):
        reclustered_df = df
        for gl in ["parent", "parent_child"]:
            gmeta, count_mat, markers2 = _make_group_marker_count_matrix(
                df=reclustered_df,
                src_cols=src_cols,
                null_marker=str(args.null_marker),
                normalize_pairs=normalize_pairs,
                merge_map=merge_map,
                group_level=gl,
                min_count=int(args.min_count),
                exclude_markers=exclude_markers,
            )
            row_coords, col_coords, sv = _ca_biplot(count_mat)

            gpoints = gmeta.copy()
            gpoints["kind"] = "group"
            gpoints["name"] = gpoints["group"].astype(str)
            gpoints["x"] = row_coords[:, 0]
            gpoints["y"] = row_coords[:, 1]
            gpoints["marker_count"] = ""
            gpoints["size"] = gpoints["row_count"].astype(float)

            # label only top groups by size
            top_g = set(gpoints.sort_values(by="row_count", ascending=False).head(int(args.biplot_label_top_groups))["group"].astype(str))
            gpoints["label"] = gpoints["group"].astype(str).where(gpoints["group"].astype(str).isin(top_g), "")

            mcounts = count_mat.sum(axis=0).astype(np.float32)
            mpoints = pd.DataFrame(
                {
                    "kind": "marker",
                    "name": markers2,
                    "group_level": gl,
                    "group": "",
                    "parent_cluster_id": "",
                    "child_cluster_id": "",
                    "row_count": "",
                    "top_markers_lift": "",
                    "top_markers": "",
                    "marker_count": mcounts.astype(int),
                    "x": col_coords[:, 0],
                    "y": col_coords[:, 1],
                    "size": np.clip(mcounts, 1.0, None),
                }
            )

            top_m = set(
                pd.DataFrame({"marker": markers2, "count": mcounts})
                .sort_values(by="count", ascending=False)
                .head(int(args.biplot_label_top_markers))["marker"]
                .astype(str)
            )
            mpoints["label"] = mpoints["name"].astype(str).where(mpoints["name"].astype(str).isin(top_m), "")

            points = pd.concat([gpoints, mpoints], ignore_index=True)
            points["sv1"] = float(sv[0])
            points["sv2"] = float(sv[1]) if len(sv) > 1 else 0.0

            out_points_csv = args.out_dir / f"biplot_group_marker_{gl}.csv"
            points.to_csv(out_points_csv, index=False, encoding="utf-8-sig")

            _try_save_plotly_biplot(
                args.out_dir / f"biplot_group_marker_{gl}.html",
                points,
                title=f"CA biplot (groups + markers): {gl}",
            )

    cfg = {
        "csv": str(args.csv),
        "src_cols": src_cols,
        "in_dir": str(args.in_dir),
        "merge_groups_csv": str(merge_groups_csv) if merge_groups_csv else "",
        "normalize_pairs": normalize_pairs,
        "null_marker": str(args.null_marker),
        "exclude_markers": sorted(exclude_markers),
        "min_count": int(args.min_count),
        "method": str(args.method),
        "seed": int(args.seed),
        "group_level": str(args.group_level),
        "reweight_by_corpus": str(args.reweight_by_corpus),
        "biplot_group_marker": bool(args.biplot_group_marker),
        "biplot_label_top_markers": int(args.biplot_label_top_markers),
        "biplot_label_top_groups": int(args.biplot_label_top_groups),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("rows", len(df))
    print("markers", len(out))
    print("dominant_groups", int(out["dominant"].nunique()))
    print("saved", str(out_csv))
    if out_png.exists():
        print("saved", str(out_png))
    else:
        print("plot_skipped", "matplotlib not installed")
    if bool(args.save_html):
        if out_html.exists():
            print("saved", str(out_html))
        else:
            print("html_skipped", "plotly not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
