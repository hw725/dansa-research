#!/usr/bin/env python3
"""Summarize PA trace selection behavior.

This script answers questions like:
- Which candidate family is actually selected most? (boundary/supar/whitespace_dp/...)
- How often were candidate sets skipped due to "insufficient" length?
- How often did we end up effectively evaluating only one candidate family?

Inputs
- Either a single --trace-jsonl, or a --run-dir containing pa_trace_seed*.jsonl

Output
- Console summary + optional CSV per-seed summary.

Note:
- By default, this reads both selection stages:
    - src_matched_selected
    - src_matched_selected_fallback
- Bonus usage is computed from candidate-level `prior_bonus` fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _family(tag: Optional[str]) -> str:
    if not tag:
        return "(none)"
    if "(" in tag:
        return tag.split("(", 1)[0]
    return tag


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class SeedSummary:
    seed: int
    trace_file: str
    stages: str
    records: int

    records_fallback_stage: int

    best_family_boundary: int
    best_family_supar: int
    best_family_whitespace_dp: int
    best_family_other: int

    sufficient_exists_true: int
    considered_eq_1: int

    candidates_total_sum: int
    candidates_considered_sum: int
    candidates_skipped_insufficient_sum: int

    prior_bonus_any_hits_sum: int
    prior_bonus_any_sum_sum: float
    prior_bonus_best_hits_sum: int
    prior_bonus_best_sum_sum: float

    skipped_insufficient_for_desired_sum: int
    skipped_insufficient_for_desired_boundary_sum: int
    skipped_insufficient_for_desired_supar_sum: int
    skipped_insufficient_for_desired_whitespace_dp_sum: int
    skipped_insufficient_for_desired_other_sum: int


def _iter_selection_records(
    trace_jsonl: Path, *, stages: Tuple[str, ...]
) -> Iterable[Dict[str, Any]]:
    for rec in _read_jsonl(trace_jsonl):
        st = rec.get("stage")
        if st in stages:
            yield rec


def summarize_trace(trace_jsonl: Path, *, stages: Tuple[str, ...], seed: int) -> SeedSummary:
    best_family = Counter()
    sufficient_exists_true = 0
    considered_eq_1 = 0

    candidates_total_sum = 0
    candidates_considered_sum = 0
    candidates_skipped_insufficient_sum = 0

    records_fallback_stage = 0

    prior_bonus_any_hits_sum = 0
    prior_bonus_any_sum_sum = 0.0
    prior_bonus_best_hits_sum = 0
    prior_bonus_best_sum_sum = 0.0

    skipped_insufficient_for_desired_sum = 0
    skipped_insufficient_for_desired_boundary_sum = 0
    skipped_insufficient_for_desired_supar_sum = 0
    skipped_insufficient_for_desired_whitespace_dp_sum = 0
    skipped_insufficient_for_desired_other_sum = 0

    records = 0

    for rec in _iter_selection_records(trace_jsonl, stages=stages):
        meta = rec.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        records += 1

        st = rec.get("stage")
        if st == "src_matched_selected_fallback":
            records_fallback_stage += 1

        bt = meta.get("best_tag")
        if isinstance(bt, str):
            best_family[_family(bt)] += 1
        else:
            best_family["(none)"] += 1

        se = meta.get("sufficient_exists")
        if se is True:
            sufficient_exists_true += 1

        cc = meta.get("candidates_considered")
        if isinstance(cc, int) and cc == 1:
            considered_eq_1 += 1

        ct = meta.get("candidates_total")
        if isinstance(ct, int):
            candidates_total_sum += ct

        if isinstance(cc, int):
            candidates_considered_sum += cc

        csi = meta.get("candidates_skipped_insufficient")
        if isinstance(csi, int):
            candidates_skipped_insufficient_sum += csi

        # Bonus usage: infer from candidate-level prior_bonus.
        top_candidates = meta.get("top_candidates")
        if isinstance(top_candidates, list):
            best_tag = meta.get("best_tag")
            best_prior_bonus: Optional[float] = None
            for c in top_candidates:
                if not isinstance(c, dict):
                    continue
                pb = c.get("prior_bonus")
                if isinstance(pb, (int, float)) and float(pb) > 0.0:
                    prior_bonus_any_hits_sum += 1
                    prior_bonus_any_sum_sum += float(pb)

                tag = c.get("tag")
                if best_prior_bonus is None and isinstance(best_tag, str) and tag == best_tag:
                    if isinstance(pb, (int, float)):
                        best_prior_bonus = float(pb)

            if best_prior_bonus is not None and best_prior_bonus > 0.0:
                prior_bonus_best_hits_sum += 1
                prior_bonus_best_sum_sum += best_prior_bonus

        # Skip reasons are only available on some stages (notably *_fallback)
        candidate_reports = meta.get("candidate_reports")
        if isinstance(candidate_reports, list):
            for rep in candidate_reports:
                if not isinstance(rep, dict):
                    continue
                if rep.get("skip_reason") != "insufficient_for_desired":
                    continue
                skipped_insufficient_for_desired_sum += 1
                tag = rep.get("tag")
                fam = _family(tag) if isinstance(tag, str) else "(none)"
                if fam == "boundary":
                    skipped_insufficient_for_desired_boundary_sum += 1
                elif fam == "supar":
                    skipped_insufficient_for_desired_supar_sum += 1
                elif fam == "whitespace_dp":
                    skipped_insufficient_for_desired_whitespace_dp_sum += 1
                else:
                    skipped_insufficient_for_desired_other_sum += 1

    boundary = best_family.get("boundary", 0)
    supar = best_family.get("supar", 0)
    ws = best_family.get("whitespace_dp", 0)

    other = int(sum(best_family.values()) - boundary - supar - ws)

    return SeedSummary(
        seed=int(seed),
        trace_file=str(trace_jsonl.as_posix()),
        stages=",".join(stages),
        records=int(records),

        records_fallback_stage=int(records_fallback_stage),
        best_family_boundary=int(boundary),
        best_family_supar=int(supar),
        best_family_whitespace_dp=int(ws),
        best_family_other=int(other),
        sufficient_exists_true=int(sufficient_exists_true),
        considered_eq_1=int(considered_eq_1),
        candidates_total_sum=int(candidates_total_sum),
        candidates_considered_sum=int(candidates_considered_sum),
        candidates_skipped_insufficient_sum=int(candidates_skipped_insufficient_sum),
        prior_bonus_any_hits_sum=int(prior_bonus_any_hits_sum),
        prior_bonus_any_sum_sum=float(prior_bonus_any_sum_sum),
        prior_bonus_best_hits_sum=int(prior_bonus_best_hits_sum),
        prior_bonus_best_sum_sum=float(prior_bonus_best_sum_sum),

        skipped_insufficient_for_desired_sum=int(skipped_insufficient_for_desired_sum),
        skipped_insufficient_for_desired_boundary_sum=int(skipped_insufficient_for_desired_boundary_sum),
        skipped_insufficient_for_desired_supar_sum=int(skipped_insufficient_for_desired_supar_sum),
        skipped_insufficient_for_desired_whitespace_dp_sum=int(skipped_insufficient_for_desired_whitespace_dp_sum),
        skipped_insufficient_for_desired_other_sum=int(skipped_insufficient_for_desired_other_sum),
    )


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "-"
    return f"{(100.0 * n / d):5.1f}%"


def _print_seed_summary(s: SeedSummary) -> None:
    r = s.records
    print(f"seed={s.seed} records={r} trace={Path(s.trace_file).name}")
    print(
        "  best_family: "
        f"boundary={s.best_family_boundary}({_pct(s.best_family_boundary, r)}), "
        f"supar={s.best_family_supar}({_pct(s.best_family_supar, r)}), "
        f"whitespace_dp={s.best_family_whitespace_dp}({_pct(s.best_family_whitespace_dp, r)}), "
        f"other={s.best_family_other}({_pct(s.best_family_other, r)})"
    )
    print(
        "  selection_health: "
        f"sufficient_exists=True {s.sufficient_exists_true}({_pct(s.sufficient_exists_true, r)}), "
        f"considered==1 {s.considered_eq_1}({_pct(s.considered_eq_1, r)})"
    )

    if r > 0:
        print(
            "  candidates: "
            f"avg_total={(s.candidates_total_sum / r):.2f}, "
            f"avg_considered={(s.candidates_considered_sum / r):.2f}, "
            f"avg_skipped_insufficient={(s.candidates_skipped_insufficient_sum / r):.2f}"
        )
        print(
            "  prior_bonus(any_candidate): "
            f"hits_sum={s.prior_bonus_any_hits_sum}, sum_sum={s.prior_bonus_any_sum_sum:.4f}"
        )
        print(
            "  prior_bonus(best_candidate): "
            f"hits_sum={s.prior_bonus_best_hits_sum}, sum_sum={s.prior_bonus_best_sum_sum:.4f}"
        )
        if s.records_fallback_stage:
            print(
                "  skip_reason(insufficient_for_desired): "
                f"total={s.skipped_insufficient_for_desired_sum}, "
                f"boundary={s.skipped_insufficient_for_desired_boundary_sum}, "
                f"supar={s.skipped_insufficient_for_desired_supar_sum}, "
                f"whitespace_dp={s.skipped_insufficient_for_desired_whitespace_dp_sum}, "
                f"other={s.skipped_insufficient_for_desired_other_sum}"
            )



def _discover_traces(run_dir: Path) -> List[Tuple[int, Path]]:
    traces: List[Tuple[int, Path]] = []
    for p in sorted(run_dir.glob("pa_trace_seed*.jsonl")):
        # parse seed from filename: pa_trace_seed{seed}.jsonl
        name = p.stem
        seed = None
        for token in name.split("seed", 1)[1:]:
            try:
                seed = int(token)
                break
            except Exception:
                pass
        if seed is None:
            # fallback regex-free parse
            digits = "".join(ch for ch in name if ch.isdigit())
            if digits:
                seed = int(digits)
        traces.append((int(seed or 0), p))
    return traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="src_matched_selected,src_matched_selected_fallback",
        help="Comma-separated stages to include",
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-jsonl", type=str, default=None)
    src.add_argument("--run-dir", type=str, default=None)

    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    stages = tuple(s.strip() for s in str(args.stage).split(",") if s.strip())
    if not stages:
        raise SystemExit("--stage produced no stages")
    summaries: List[SeedSummary] = []

    if args.trace_jsonl:
        p = Path(str(args.trace_jsonl))
        if not p.exists():
            raise SystemExit(f"trace not found: {p}")
        summaries.append(summarize_trace(p, stages=stages, seed=0))
    else:
        run_dir = Path(str(args.run_dir))
        if not run_dir.exists():
            raise SystemExit(f"run-dir not found: {run_dir}")
        traces = _discover_traces(run_dir)
        if not traces:
            raise SystemExit(f"no pa_trace_seed*.jsonl under: {run_dir}")
        for seed, p in traces:
            summaries.append(summarize_trace(p, stages=stages, seed=seed))

    # Print per-seed
    for s in sorted(summaries, key=lambda x: x.seed):
        _print_seed_summary(s)
        print()

    # Aggregate
    agg = SeedSummary(
        seed=-1,
        trace_file="(aggregate)",
        stages=",".join(stages),
        records=sum(x.records for x in summaries),

        records_fallback_stage=sum(x.records_fallback_stage for x in summaries),
        best_family_boundary=sum(x.best_family_boundary for x in summaries),
        best_family_supar=sum(x.best_family_supar for x in summaries),
        best_family_whitespace_dp=sum(x.best_family_whitespace_dp for x in summaries),
        best_family_other=sum(x.best_family_other for x in summaries),
        sufficient_exists_true=sum(x.sufficient_exists_true for x in summaries),
        considered_eq_1=sum(x.considered_eq_1 for x in summaries),
        candidates_total_sum=sum(x.candidates_total_sum for x in summaries),
        candidates_considered_sum=sum(x.candidates_considered_sum for x in summaries),
        candidates_skipped_insufficient_sum=sum(x.candidates_skipped_insufficient_sum for x in summaries),
        prior_bonus_any_hits_sum=sum(x.prior_bonus_any_hits_sum for x in summaries),
        prior_bonus_any_sum_sum=sum(x.prior_bonus_any_sum_sum for x in summaries),
        prior_bonus_best_hits_sum=sum(x.prior_bonus_best_hits_sum for x in summaries),
        prior_bonus_best_sum_sum=sum(x.prior_bonus_best_sum_sum for x in summaries),

        skipped_insufficient_for_desired_sum=sum(x.skipped_insufficient_for_desired_sum for x in summaries),
        skipped_insufficient_for_desired_boundary_sum=sum(
            x.skipped_insufficient_for_desired_boundary_sum for x in summaries
        ),
        skipped_insufficient_for_desired_supar_sum=sum(
            x.skipped_insufficient_for_desired_supar_sum for x in summaries
        ),
        skipped_insufficient_for_desired_whitespace_dp_sum=sum(
            x.skipped_insufficient_for_desired_whitespace_dp_sum for x in summaries
        ),
        skipped_insufficient_for_desired_other_sum=sum(
            x.skipped_insufficient_for_desired_other_sum for x in summaries
        ),
    )

    print("=== AGGREGATE ===")
    _print_seed_summary(agg)

    if args.out_csv:
        out = Path(str(args.out_csv))
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(summaries[0]).keys()) if summaries else list(asdict(agg).keys())
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in sorted(summaries, key=lambda x: x.seed):
                w.writerow(asdict(s))
            w.writerow(asdict(agg))
        print()
        print(f"wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
