import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Key = Tuple[str, str]


@dataclass(frozen=True)
class Row:
    book_name: str
    paragraph_id: str
    pred_n: int
    gold_n: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    pred_boundaries: str
    gold_boundaries: str
    first_diff_pos_norm: str
    first_pos_in_pred: str
    first_pos_in_gold: str
    pred_snip_norm: str
    gold_snip_norm: str


def _to_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_report(path: Path) -> Dict[Key, Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"book_name", "문단식별자", "tp", "fp", "fn", "precision", "recall", "f1"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        out: Dict[Key, Row] = {}
        for r in reader:
            book = (r.get("book_name") or "").strip()
            pid = (r.get("문단식별자") or "").strip()
            key = (book, pid)
            out[key] = Row(
                book_name=book,
                paragraph_id=pid,
                pred_n=_to_int(r.get("pred_n", "0")),
                gold_n=_to_int(r.get("gold_n", "0")),
                tp=_to_int(r.get("tp", "0")),
                fp=_to_int(r.get("fp", "0")),
                fn=_to_int(r.get("fn", "0")),
                precision=_to_float(r.get("precision", "nan")),
                recall=_to_float(r.get("recall", "nan")),
                f1=_to_float(r.get("f1", "nan")),
                pred_boundaries=r.get("pred_boundaries", ""),
                gold_boundaries=r.get("gold_boundaries", ""),
                first_diff_pos_norm=r.get("first_diff_pos_norm", ""),
                first_pos_in_pred=r.get("first_pos_in_pred", ""),
                first_pos_in_gold=r.get("first_pos_in_gold", ""),
                pred_snip_norm=r.get("pred_snip_norm", ""),
                gold_snip_norm=r.get("gold_snip_norm", ""),
            )
        return out


def _safe_median(values: List[float]) -> float:
    vals = [v for v in values if v == v]  # filter NaN
    if not vals:
        return float("nan")
    return statistics.median(vals)


def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    return statistics.mean(vals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two boundary mismatch reports (tgt-exact subset) keyed by (book_name, 문단식별자). "
            "Outputs a merged CSV with deltas and prints a concise summary."
        )
    )
    parser.add_argument("--a", required=True, help="Path to report A (baseline), e.g. seed2")
    parser.add_argument("--b", required=True, help="Path to report B (candidate), e.g. thr0.5")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--topk", type=int, default=15, help="How many top improved/worsened rows to print")
    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a = load_report(a_path)
    b = load_report(b_path)

    keys = sorted(set(a.keys()) | set(b.keys()))

    # Build merged rows
    merged_rows: List[dict] = []
    delta_f1_common: List[float] = []

    improved: List[Tuple[float, Key]] = []
    worsened: List[Tuple[float, Key]] = []

    only_a = 0
    only_b = 0
    common = 0

    for key in keys:
        ra = a.get(key)
        rb = b.get(key)

        if ra is None:
            only_b += 1
        elif rb is None:
            only_a += 1
        else:
            common += 1

        def put(prefix: str, row: Optional[Row]) -> dict:
            if row is None:
                return {
                    f"{prefix}_pred_n": "",
                    f"{prefix}_gold_n": "",
                    f"{prefix}_tp": "",
                    f"{prefix}_fp": "",
                    f"{prefix}_fn": "",
                    f"{prefix}_precision": "",
                    f"{prefix}_recall": "",
                    f"{prefix}_f1": "",
                    f"{prefix}_first_diff_pos_norm": "",
                    f"{prefix}_pred_snip_norm": "",
                    f"{prefix}_gold_snip_norm": "",
                }
            return {
                f"{prefix}_pred_n": row.pred_n,
                f"{prefix}_gold_n": row.gold_n,
                f"{prefix}_tp": row.tp,
                f"{prefix}_fp": row.fp,
                f"{prefix}_fn": row.fn,
                f"{prefix}_precision": row.precision,
                f"{prefix}_recall": row.recall,
                f"{prefix}_f1": row.f1,
                f"{prefix}_first_diff_pos_norm": row.first_diff_pos_norm,
                f"{prefix}_pred_snip_norm": row.pred_snip_norm,
                f"{prefix}_gold_snip_norm": row.gold_snip_norm,
            }

        row_out = {
            "book_name": key[0],
            "문단식별자": key[1],
        }
        row_out.update(put("a", ra))
        row_out.update(put("b", rb))

        # deltas when both exist
        if ra is not None and rb is not None:
            df1 = rb.f1 - ra.f1
            dtp = rb.tp - ra.tp
            dfp = rb.fp - ra.fp
            dfn = rb.fn - ra.fn
            row_out.update(
                {
                    "delta_f1": df1,
                    "delta_tp": dtp,
                    "delta_fp": dfp,
                    "delta_fn": dfn,
                }
            )
            delta_f1_common.append(df1)
            if df1 == df1:  # not NaN
                if df1 > 0:
                    improved.append((df1, key))
                elif df1 < 0:
                    worsened.append((df1, key))
        else:
            row_out.update({"delta_f1": "", "delta_tp": "", "delta_fp": "", "delta_fn": ""})

        merged_rows.append(row_out)

    # Write merged CSV
    fieldnames = list(merged_rows[0].keys()) if merged_rows else ["book_name", "문단식별자"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    improved.sort(reverse=True, key=lambda x: x[0])
    worsened.sort(key=lambda x: x[0])

    unchanged = common - len(improved) - len(worsened)

    print("=")
    print("Boundary mismatch report comparison")
    print("=")
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"keys: a={len(a)} b={len(b)} common={common} only_a={only_a} only_b={only_b}")
    print(
        "delta_f1 over common keys: "
        f"mean={_safe_mean(delta_f1_common):.4f} median={_safe_median(delta_f1_common):.4f} "
        f"improved={len(improved)} worsened={len(worsened)} unchanged={unchanged}"
    )
    print(f"out: {out_path}")

    topk = max(0, int(args.topk))
    if topk:
        def fmt_key(k: Key) -> str:
            return f"{k[0]} / {k[1]}"

        print("\nTop improved (delta_f1):")
        for df1, k in improved[:topk]:
            print(f"  {df1:+.4f}  {fmt_key(k)}")

        print("\nTop worsened (delta_f1):")
        for df1, k in worsened[:topk]:
            print(f"  {df1:+.4f}  {fmt_key(k)}")


if __name__ == "__main__":
    main()
