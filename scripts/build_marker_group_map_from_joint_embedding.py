#!/usr/bin/env python3
"""Build a marker grouping map from a joint (parent+marker) 2D embedding.

Goal
- Group low-frequency markers into nearby high-frequency marker groups.
- Use parent points as a "bridge": pick nearest parent for a marker, then
  prefer top markers that are also nearest to that parent.

Input CSV format (as produced by existing pipeline):
- columns: type,id,label,x,y,count
- type: 'parent' | 'marker'
- marker label '<EMPTY>' represents empty marker.

Output JSON format:
{
  "meta": {...},
  "top_markers": [...],
  "marker_map": {"raw_marker": "mapped_marker", ...}
}

Notes
- This script is intentionally conservative. If a rare marker is not close to
  any top marker, it maps to OTHER.
- Normalization removes whitespace in marker strings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _norm_marker(s: str) -> str:
    return "".join(str(s).split())


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> int:
    p = argparse.ArgumentParser(description="Build marker grouping map from joint embedding CSV")
    p.add_argument(
        "--csv",
        default=str(Path("reports") / "k16_analysis" / "parent_marker_joint_embedding.csv"),
        help="Joint embedding CSV path",
    )
    p.add_argument("--top-n", type=int, default=100, help="Number of top markers to keep as group heads")
    p.add_argument("--min-count-keep", type=int, default=20, help="Keep markers with count >= this (no grouping)")
    p.add_argument("--max-dist", type=float, default=2.0, help="Max distance to map a rare marker to a top marker")
    p.add_argument("--other-token", default="__OTHER__", help="Fallback group label")
    p.add_argument(
        "--out",
        default=str(Path("reports") / "k16_analysis" / "marker_group_map.json"),
        help="Output JSON path",
    )

    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"missing csv: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"type", "label", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in csv: {sorted(missing)}")

    parents_df = df[df["type"] == "parent"].copy()
    markers_df = df[df["type"] == "marker"].copy()

    parents: Dict[str, Tuple[float, float]] = {}
    for _, r in parents_df.iterrows():
        parents[str(r["label"])] = (float(r["x"]), float(r["y"]))

    markers: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}
    for _, r in markers_df.iterrows():
        label = str(r["label"])
        if label == "<EMPTY>":
            label = ""
        label = _norm_marker(label)
        markers[label] = (float(r["x"]), float(r["y"]))
        c = r.get("count", 0)
        try:
            counts[label] = int(float(c)) if c == c else 0
        except Exception:
            counts[label] = 0

    # Choose top markers by count (excluding empty)
    sorted_by_count = sorted(
        [(m, counts.get(m, 0)) for m in markers.keys() if m != ""],
        key=lambda kv: (-kv[1], kv[0]),
    )
    top_n = max(0, int(args.top_n))
    top_markers = [m for m, _ in sorted_by_count[:top_n]]

    # For each top marker, precompute nearest parent label
    top_to_parent: Dict[str, str] = {}
    parent_labels = list(parents.keys())
    for m in top_markers:
        mp = markers[m]
        best_parent = None
        best_d = 10**9
        for pl in parent_labels:
            d = _dist(mp, parents[pl])
            if d < best_d:
                best_d = d
                best_parent = pl
        if best_parent is not None:
            top_to_parent[m] = best_parent

    # Build mapping
    min_keep = int(args.min_count_keep)
    max_dist = float(args.max_dist)
    other = str(args.other_token)

    marker_map: Dict[str, str] = {}

    # Always keep empty marker as-is
    marker_map[""] = ""

    # Helper: nearest parent for any marker
    def nearest_parent(pt: Tuple[float, float]) -> str | None:
        best_parent = None
        best_d = 10**9
        for pl in parent_labels:
            d = _dist(pt, parents[pl])
            if d < best_d:
                best_d = d
                best_parent = pl
        return best_parent

    for m, pt in markers.items():
        if m == "":
            continue

        c = counts.get(m, 0)
        if c >= min_keep:
            marker_map[m] = m
            continue
        if m in top_markers:
            marker_map[m] = m
            continue

        # Prefer top markers that share the nearest parent
        np_label = nearest_parent(pt)
        candidates: List[str] = []
        if np_label is not None:
            candidates = [t for t in top_markers if top_to_parent.get(t) == np_label]

        if not candidates:
            candidates = top_markers

        best_t = None
        best_d = 10**9
        for t in candidates:
            d = _dist(pt, markers[t])
            if d < best_d:
                best_d = d
                best_t = t

        if best_t is not None and best_d <= max_dist:
            marker_map[m] = best_t
        else:
            marker_map[m] = other

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "csv": csv_path.as_posix(),
            "top_n": top_n,
            "min_count_keep": min_keep,
            "max_dist": max_dist,
            "other_token": other,
            "normalization": "remove_whitespace",
        },
        "top_markers": top_markers,
        "marker_map": marker_map,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = sum(1 for k, v in marker_map.items() if k == v)
    mapped_to_other = sum(1 for v in marker_map.values() if v == other)
    print(f"✅ wrote: {out_path.as_posix()}")
    print(f"   markers_in_map={len(marker_map):,}")
    print(f"   kept_self={kept:,}")
    print(f"   mapped_to_other={mapped_to_other:,}")
    print(f"   top_markers={len(top_markers):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
