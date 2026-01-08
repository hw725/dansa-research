#!/usr/bin/env python3
"""Emit final, human-reviewable tables for cluster confirmation.

Produces CSV/Markdown tables that combine:
- group coordinates (x,y) from group embedding
- group size
- discriminative marker signals (top_markers_lift)
- dominant textual boundary patterns (src_left prefixes, ratios)
- representative examples

Intended output location: hyeonto/reports/k16_analysis_minper50/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re


_RE_LEFT_SAYS = re.compile(r"曰\s*$")
_RE_ATTR = re.compile(r"^(?:子曰|有子曰|孟子曰|程子曰|范氏曰|何氏曰|謝氏曰|史記世家曰|又曰)$")
_RE_DEFINE = re.compile(r"^[^\s]{1,12}(?:은|는)\s+.+(?:也|矣|라)[^\s]*$")
_RE_YEAR = re.compile(r"[一二三四五六七八九十百千0-9]{1,4}年")


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s == "nan" else s


def _quantile_len(series: pd.Series, q: float) -> float:
    vals = series.astype(str).map(len).to_numpy(dtype=np.float32)
    if vals.size == 0:
        return 0.0
    return float(np.quantile(vals, q))


def _top_left_prefixes(left: pd.Series, k: int = 5) -> str:
    # Prefer canonical "...曰" tokens; else fall back to first 6 chars.
    def norm(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        if _RE_ATTR.match(s):
            return s
        if s.endswith("曰"):
            return s
        return s[:6]

    items = left.astype(str).map(_safe_str).map(norm)
    vc = items[items != ""].value_counts().head(int(k))
    return ";".join([f"{i}:{int(n)}" for i, n in vc.items()])


def _definition_from_stats(stats: dict[str, float], top_lift: str) -> str:
    frac_says = stats.get("frac_left_endswith_曰", 0.0)
    frac_define = stats.get("frac_define_like", 0.0)
    frac_year = stats.get("frac_year", 0.0)
    frac_short_long = stats.get("frac_short_left_and_long_right", 0.0)

    if frac_says >= 0.55:
        return "문답/인용 도입(…曰) 경계"
    if frac_define >= 0.25:
        return "용어 풀이/주석 정의형 경계(…은/…는 …라/…也)"
    if frac_year >= 0.15:
        return "연표/서술 전개(연도·사건 나열) 경계"
    if frac_short_long >= 0.35:
        return "짧은 도입부 → 본문 전개(접속/요약) 경계"

    if top_lift:
        # still useful: name by over-represented markers
        return f"혼합/세부패턴: 과대표 현토({top_lift.split()[0]}) 중심"
    return "혼합/세부패턴(추가 검토 필요)"


def _compute_group_table(reclustered: pd.DataFrame, group_embed: pd.DataFrame, group_level: str, examples: int) -> pd.DataFrame:
    rows = []

    for _, g in group_embed.iterrows():
        parent = int(g["parent_cluster_id"])
        child = int(g["child_cluster_id"])

        if group_level == "parent":
            sub = reclustered.loc[reclustered.parent_cluster_id == parent]
            group = f"p{parent}"
        else:
            sub = reclustered.loc[(reclustered.parent_cluster_id == parent) & (reclustered.child_cluster_id == child)]
            group = f"p{parent}_c{child}"

        if sub.empty:
            continue

        left = sub["src_left"].astype(str).map(_safe_str)
        right = sub["src_right"].astype(str).map(_safe_str)

        n = float(len(sub))
        frac_says = float(left.map(lambda s: bool(_RE_LEFT_SAYS.search(s.strip()))).sum()) / n
        frac_define = float(left.map(lambda s: bool(_RE_DEFINE.match(s.strip()))).sum()) / n
        frac_year = float((left + " " + right).map(lambda s: bool(_RE_YEAR.search(s))).sum()) / n

        left_len = left.map(len)
        right_len = right.map(len)
        frac_short_long = float(((left_len <= 6) & (right_len >= 18)).sum()) / n

        stats = {
            "frac_left_endswith_曰": frac_says,
            "frac_define_like": frac_define,
            "frac_year": frac_year,
            "frac_short_left_and_long_right": frac_short_long,
            "left_len_p50": float(left_len.median()),
            "right_len_p50": float(right_len.median()),
            "left_len_p90": float(np.quantile(left_len.to_numpy(dtype=np.float32), 0.9)),
            "right_len_p90": float(np.quantile(right_len.to_numpy(dtype=np.float32), 0.9)),
        }

        top_lift = _safe_str(g.get("top_markers_lift"))
        definition = _definition_from_stats(stats, top_lift=top_lift)

        ex = sub.sort_values(["book_name", "paragraph_id", "left_sentence_id"], kind="mergesort").head(int(examples))
        ex0 = ex.iloc[0]

        rows.append(
            {
                "group_level": group_level,
                "group": group,
                "parent_cluster_id": parent,
                "child_cluster_id": child if group_level != "parent" else "",
                "x": float(g.get("x")),
                "y": float(g.get("y")),
                "row_count": int(g.get("row_count", len(sub))),
                "top_markers_lift": top_lift,
                "top_markers": _safe_str(g.get("top_markers")),
                "top_src_left_prefixes": _top_left_prefixes(left, k=5),
                "frac_left_endswith_曰": round(frac_says, 4),
                "frac_define_like": round(frac_define, 4),
                "frac_year": round(frac_year, 4),
                "frac_short_left_long_right": round(frac_short_long, 4),
                "definition_draft": definition,
                "example_book": _safe_str(ex0.get("book_name")),
                "example_para": int(ex0.get("paragraph_id")),
                "example_sent": f"{int(ex0.get('left_sentence_id'))}→{int(ex0.get('right_sentence_id'))}",
                "example_src_left": _safe_str(ex0.get("src_left")),
                "example_src_right": _safe_str(ex0.get("src_right")),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["row_count"], ascending=False).reset_index(drop=True)
    return out


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    # Keep it compact for human review
    cols = [
        "group",
        "row_count",
        "definition_draft",
        "top_markers_lift",
        "top_src_left_prefixes",
        "x",
        "y",
        "example_book",
        "example_para",
        "example_sent",
        "example_src_left",
        "example_src_right",
    ]
    work = df[cols].head(int(max_rows)).copy()

    # shorten very long strings
    for c in ["top_markers_lift", "top_src_left_prefixes", "example_src_left", "example_src_right"]:
        work[c] = work[c].astype(str).map(lambda s: (s[:180] + "…") if len(s) > 180 else s)

    def esc(s: object) -> str:
        t = "" if s is None else str(s)
        t = t.replace("\n", " ").replace("\r", " ")
        t = t.replace("|", "\\|")
        return t

    headers = [esc(c) for c in cols]
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in work.iterrows():
        row = [esc(r.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit final cluster confirmation tables")
    ap.add_argument("--reclustered", type=Path, required=True)
    ap.add_argument("--group-embed-parent", type=Path, required=True)
    ap.add_argument("--group-embed-parent-child", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    reclustered = pd.read_csv(args.reclustered)
    parent_embed = pd.read_csv(args.group_embed_parent)
    pc_embed = pd.read_csv(args.group_embed_parent_child)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    parent_table = _compute_group_table(reclustered, parent_embed, group_level="parent", examples=int(args.examples))
    pc_table = _compute_group_table(reclustered, pc_embed, group_level="parent_child", examples=int(args.examples))

    out_parent_csv = args.out_dir / "final_cluster_table_parent.csv"
    out_pc_csv = args.out_dir / "final_cluster_table_parent_child.csv"
    parent_table.to_csv(out_parent_csv, index=False, encoding="utf-8-sig")
    pc_table.to_csv(out_pc_csv, index=False, encoding="utf-8-sig")

    (args.out_dir / "final_cluster_table_parent.md").write_text(_to_markdown_table(parent_table), encoding="utf-8")
    (args.out_dir / "final_cluster_table_parent_child.md").write_text(_to_markdown_table(pc_table), encoding="utf-8")

    print("wrote", out_parent_csv)
    print("wrote", out_pc_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
