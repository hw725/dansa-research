import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _to_int(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _bucket_pos(pos: str) -> str:
    """pos is first_diff_pos_norm (character index). bucket coarsely."""
    i = _to_int(pos)
    if i <= 0:
        return "<=0"
    if i <= 50:
        return "1-50"
    if i <= 100:
        return "51-100"
    if i <= 200:
        return "101-200"
    if i <= 400:
        return "201-400"
    return ">400"


def _safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def _safe_median(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize delta patterns from compare_boundary_mismatch_*.csv")
    p.add_argument("--input", required=True, help="compare CSV (output of compare_boundary_mismatch_reports.py)")
    p.add_argument("--out-prefix", required=True, help="output prefix path (no extension)")
    p.add_argument("--topk", type=int, default=20, help="topk examples to include")
    args = p.parse_args()

    in_path = Path(args.input)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # classify rows with both sides present (delta_f1 not empty)
    common = [r for r in rows if (r.get("delta_f1") or "").strip() != ""]
    worsened = [r for r in common if _to_float(r["delta_f1"]) < 0]
    improved = [r for r in common if _to_float(r["delta_f1"]) > 0]
    unchanged = [r for r in common if _to_float(r["delta_f1"]) == 0]

    # build pattern counters for worsened
    buck = Counter()
    fp_dir = Counter()
    fn_dir = Counter()
    tp_dir = Counter()

    df1_list = []
    dfp_list = []
    dfn_list = []

    for r in worsened:
        buck[_bucket_pos(r.get("b_first_diff_pos_norm", ""))] += 1
        d_fp = _to_int(r.get("delta_fp", "0"))
        d_fn = _to_int(r.get("delta_fn", "0"))
        d_tp = _to_int(r.get("delta_tp", "0"))
        df1 = _to_float(r.get("delta_f1", "nan"))

        df1_list.append(df1)
        dfp_list.append(float(d_fp))
        dfn_list.append(float(d_fn))

        fp_dir["fp++" if d_fp > 0 else "fp--" if d_fp < 0 else "fp=="] += 1
        fn_dir["fn++" if d_fn > 0 else "fn--" if d_fn < 0 else "fn=="] += 1
        tp_dir["tp++" if d_tp > 0 else "tp--" if d_tp < 0 else "tp=="] += 1

    # write filtered CSVs
    worsened_path = prefix.with_suffix(".worsened.csv")
    improved_path = prefix.with_suffix(".improved.csv")
    summary_path = prefix.with_suffix(".summary.txt")

    def write_csv(path: Path, subset: List[Dict[str, str]]) -> None:
        if not subset:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subset[0].keys()))
            w.writeheader()
            w.writerows(subset)

    write_csv(worsened_path, sorted(worsened, key=lambda r: _to_float(r["delta_f1"])) )
    write_csv(improved_path, sorted(improved, key=lambda r: _to_float(r["delta_f1"]), reverse=True))

    # topk examples
    topk = max(0, int(args.topk))
    worsened_top = sorted(worsened, key=lambda r: _to_float(r["delta_f1"]))[:topk]
    improved_top = sorted(improved, key=lambda r: _to_float(r["delta_f1"]), reverse=True)[:topk]

    lines = []
    lines.append("Boundary delta pattern summary")
    lines.append(f"input: {in_path}")
    lines.append("")
    lines.append(f"common rows: {len(common)}")
    lines.append(f"improved: {len(improved)}")
    lines.append(f"worsened: {len(worsened)}")
    lines.append(f"unchanged: {len(unchanged)}")
    lines.append("")

    if worsened:
        lines.append("Worsened: delta stats")
        lines.append(f"  delta_f1 mean={_safe_mean(df1_list):+.4f} median={_safe_median(df1_list):+.4f} min={min(df1_list):+.4f}")
        lines.append(f"  delta_fp mean={_safe_mean(dfp_list):+.2f} median={_safe_median(dfp_list):+.2f}")
        lines.append(f"  delta_fn mean={_safe_mean(dfn_list):+.2f} median={_safe_median(dfn_list):+.2f}")
        lines.append("")

        lines.append("Worsened: first_diff_pos_norm buckets (using B side)")
        for k, c in buck.most_common():
            lines.append(f"  {k}: {c}")
        lines.append("")

        lines.append("Worsened: direction counts")
        lines.append(f"  {dict(fp_dir)}")
        lines.append(f"  {dict(fn_dir)}")
        lines.append(f"  {dict(tp_dir)}")
        lines.append("")

    lines.append(f"worsened_csv: {worsened_path}")
    lines.append(f"improved_csv: {improved_path}")
    lines.append("")

    def one_line(r: Dict[str, str]) -> str:
        return (
            f"{r.get('book_name','')} / {r.get('문단식별자','')} "
            f"delta_f1={_to_float(r.get('delta_f1','nan')):+.4f} "
            f"(tp {r.get('a_tp','')}→{r.get('b_tp','')}, fp {r.get('a_fp','')}→{r.get('b_fp','')}, fn {r.get('a_fn','')}→{r.get('b_fn','')})"
        )

    if improved_top:
        lines.append("Top improved examples")
        for r in improved_top:
            lines.append("  " + one_line(r))
        lines.append("")

    if worsened_top:
        lines.append("Top worsened examples")
        for r in worsened_top:
            lines.append("  " + one_line(r))
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("=")
    print("Delta pattern summary saved")
    print("=")
    print(f"summary: {summary_path}")
    print(f"worsened_csv: {worsened_path}")
    print(f"improved_csv: {improved_path}")


if __name__ == "__main__":
    main()
