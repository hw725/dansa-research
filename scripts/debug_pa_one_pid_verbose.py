"""Debug helper: run PA strict alignment for a single paragraph (pid) with verbose candidate scoring.

Usage (inside docker):
    python scripts/debug_pa_one_pid_verbose.py --pid 10 --input-pd datasets/pd/test_100.csv --threshold 0.72 --min-len 10 --device cuda

This prints:
- tgt sentence count
- candidate tags + scores (from process_paragraph_alignment_with_boundary_model verbose logs)
- final chosen src/tgt pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.boundary_model_loader import BoundaryModelLoader
from common.alignment_model_loader import AlignmentMatcher


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="문단식별자")
    ap.add_argument("--input-pd", required=True, help="PD input CSV (문단 단위)")
    ap.add_argument("--threshold", type=float, default=0.72, help="boundary threshold")
    ap.add_argument("--min-len", type=int, default=None, help="boundary min_len override")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    df = pd.read_csv(args.input_pd)
    row = df[df["문단식별자"] == args.pid]
    if row.empty:
        raise SystemExit(f"pid not found: {args.pid}")

    src_paragraph = str(row.iloc[0]["원문"])
    tgt_paragraph = str(row.iloc[0]["번역문"])

    models_root = Path(__file__).resolve().parents[1] / "models"
    boundary_path = models_root / "boundary_multitask.pt"
    align_path = models_root / "dual_encoder_alignment_pa.pt"

    boundary_model = BoundaryModelLoader(model_path=boundary_path, device=args.device)
    alignment_model = AlignmentMatcher(model_path=align_path, device=args.device)

    # strict 요구: 파서 초기화
    import common.new_parsers as new_parsers

    new_parsers.ensure_kanbun_pipeline()
    new_parsers.ensure_stanza_pipeline(lang="ko")

    from pa.processor import process_paragraph_alignment_with_boundary_model

    print(f"\n=== pid={args.pid} ===")
    print(f"src_len={len(src_paragraph)} tgt_len={len(tgt_paragraph)}")

    alignments = process_paragraph_alignment_with_boundary_model(
        src_paragraph=src_paragraph,
        tgt_paragraph=tgt_paragraph,
        boundary_model=boundary_model,
        alignment_model=alignment_model,
        threshold=args.threshold,
        boundary_min_len=args.min_len,
        verbose=True,
    )

    print("\n--- final pairs ---")
    for a in alignments:
        sid = a.get("문장식별자")
        src = a.get("원문", "")
        tgt = a.get("번역문", "")
        print(f"[{sid}] SRC({len(src)}): {src}")
        print(f"     TGT({len(tgt)}): {tgt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
