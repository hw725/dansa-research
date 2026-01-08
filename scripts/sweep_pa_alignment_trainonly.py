#!/usr/bin/env python3
"""Sweep alignment hyperparameters (TRAIN-ONLY) while keeping boundary fixed.

What it does per trial:
1) Train alignment model (datasets/pa/train.csv only)
2) Run PA(strict) on PD input paragraphs
3) Evaluate vs gold using integrity_report

Outputs:
- test_results/sweep_pa_alignment_<timestamp>.csv
- Per-trial PA outputs under test_results/sweep_runs/

Example (fast-ish):
  docker-compose run --rm csp python scripts/sweep_pa_alignment_trainonly.py \
    --enable-hard-neg \
    --temperatures 0.05 0.07 0.1 \
    --hard-neg-modes prefix_token full \
    --hard-neg-weights 0.25 0.5 \
    --hard-neg-margins 0.1 0.15 \
    --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Trial:
    temperature: float
    hard_neg_mode: str
    hard_neg_weight: float
    hard_neg_margin: float
    seed: int


@dataclass(frozen=True)
class ParsedEval:
    translation_exact: bool | None
    translation_exact_ok: int | None
    translation_exact_total: int | None
    micro_f1_all: float | None
    micro_f1_tgt_exact: float | None
    returncode: int | None


_F1_ALL_RE = re.compile(r"\(micro,\s*전체\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")
_F1_TGT_EXACT_RE = re.compile(
    r"\(micro,\s*tgt\s*완전일치\s*subset\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)"
)
_F1_LEGACY_RE = re.compile(r"\(micro\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")
_TRANSL_RE = re.compile(r"번역문 문장리스트 완전일치:\s*(\d+)\s*/\s*(\d+)")


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], allow_failure: bool = False) -> tuple[int, str]:
    print("\n$ " + " ".join(argv))
    proc = subprocess.run(argv, cwd=str(cwd), env=env, text=True, capture_output=True)
    if proc.returncode != 0 and not allow_failure:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"명령 실패(returncode={proc.returncode}): {' '.join(argv)}")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr)


def _parse_integrity_report(text: str, *, returncode: int | None = None) -> ParsedEval:
    translation_exact = None
    translation_exact_ok = None
    translation_exact_total = None

    m = _TRANSL_RE.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        translation_exact_ok = a
        translation_exact_total = b
        translation_exact = (a == b and b > 0)

    micro_f1_all = None
    m_all = _F1_ALL_RE.search(text)
    if m_all:
        micro_f1_all = float(m_all.group(3))
    else:
        # backward compatible with older integrity_report output
        m_legacy = _F1_LEGACY_RE.search(text)
        if m_legacy:
            micro_f1_all = float(m_legacy.group(3))

    micro_f1_tgt_exact = None
    m_ok = _F1_TGT_EXACT_RE.search(text)
    if m_ok:
        micro_f1_tgt_exact = float(m_ok.group(3))

    return ParsedEval(
        translation_exact=translation_exact,
        translation_exact_ok=translation_exact_ok,
        translation_exact_total=translation_exact_total,
        micro_f1_all=micro_f1_all,
        micro_f1_tgt_exact=micro_f1_tgt_exact,
        returncode=returncode,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep PA alignment hyperparameters (train-only).")

    # shared training args
    p.add_argument(
        "--train-csv",
        default=str(WORKSPACE_ROOT / "datasets" / "pa" / "train.csv"),
        help="PA alignment 학습용 train.csv",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=0)

    p.add_argument("--enable-hard-neg", action="store_true")
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.07])
    p.add_argument("--hard-neg-modes", nargs="+", default=["prefix_token"], choices=["prefix_token", "full"])
    p.add_argument("--hard-neg-weights", nargs="+", type=float, default=[0.5])
    p.add_argument("--hard-neg-margins", nargs="+", type=float, default=[0.15])
    p.add_argument("--seeds", nargs="+", type=int, default=[1])

    # eval inputs
    p.add_argument(
        "--pd-input",
        default=str(WORKSPACE_ROOT / "datasets" / "pd" / "test_10.csv"),
        help="PA 입력(문단병렬). PD는 학습이 아니라 입력용",
    )
    p.add_argument(
        "--gold",
        default=str(WORKSPACE_ROOT / "test_results" / "gold_subset_from_pd_test10.csv"),
        help="gold(문장 단위)",
    )
    p.add_argument("--pids", nargs="*", default=None, help="평가 pid 목록(미지정 시 전체 문단 평가)")

    # PA strict
    p.add_argument("--boundary-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # output
    p.add_argument("--out-dir", default=str(WORKSPACE_ROOT / "test_results"))

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "sweep_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"sweep_pa_alignment_{ts}.csv"

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    trials: list[Trial] = []
    for t, mode, w, m, seed in product(
        args.temperatures,
        args.hard_neg_modes,
        args.hard_neg_weights,
        args.hard_neg_margins,
        args.seeds,
    ):
        trials.append(Trial(temperature=float(t), hard_neg_mode=str(mode), hard_neg_weight=float(w), hard_neg_margin=float(m), seed=int(seed)))

    print(f"총 trial: {len(trials)}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "temperature",
                "hard_neg_mode",
                "hard_neg_weight",
                "hard_neg_margin",
                "seed",
                "translation_exact",
                "translation_exact_ok",
                "translation_exact_total",
                "micro_f1_all",
                "micro_f1_tgt_exact",
                "eval_returncode",
                "pa_output",
            ],
        )
        writer.writeheader()

        best = None

        for i, tr in enumerate(trials, 1):
            print("\n" + "=" * 80)
            print(f"[{i}/{len(trials)}] temp={tr.temperature} mode={tr.hard_neg_mode} w={tr.hard_neg_weight} m={tr.hard_neg_margin} seed={tr.seed}")
            print("=" * 80)

            # 1) Train alignment (overwrites models/dual_encoder_alignment_pa.pt)
            train_cmd = [
                sys.executable,
                str(WORKSPACE_ROOT / "scripts" / "train_alignment_dual_encoder_trainonly.py"),
                "--train-csv",
                str(args.train_csv),
                "--epochs",
                str(int(args.epochs)),
                "--batch",
                str(int(args.batch)),
                "--max-steps",
                str(int(args.max_steps)),
                "--temperature",
                str(float(tr.temperature)),
                "--seed",
                str(int(tr.seed)),
                "--hard-neg-mode",
                str(tr.hard_neg_mode),
                "--hard-neg-weight",
                str(float(tr.hard_neg_weight)),
                "--hard-neg-margin",
                str(float(tr.hard_neg_margin)),
            ]
            if args.enable_hard_neg:
                train_cmd.append("--enable-hard-neg")

            _run(train_cmd, cwd=WORKSPACE_ROOT, env=env)

            # 2) PA strict
            out_pa = runs_dir / (
                f"pa_strict_temp{tr.temperature}_mode{tr.hard_neg_mode}_w{tr.hard_neg_weight}_m{tr.hard_neg_margin}_seed{tr.seed}_{ts}.csv"
            )
            pa_cmd = [
                sys.executable,
                str(WORKSPACE_ROOT / "pa" / "main.py"),
                str(args.pd_input),
                str(out_pa),
                "--embedder",
                "bge",
                "--use-boundary-model",
                "--boundary-threshold",
                str(float(args.boundary_threshold)),
                "--device",
                str(args.device),
            ]
            _run(pa_cmd, cwd=WORKSPACE_ROOT, env=env)

            # 3) Eval
            eval_cmd = [
                sys.executable,
                str(WORKSPACE_ROOT / "integrity_report.py"),
                "--input",
                str(out_pa),
                "--gold",
                str(args.gold),
            ]
            if args.pids:
                eval_cmd.extend(["--pids", *[str(x) for x in args.pids]])
            eval_rc, report_text = _run(eval_cmd, cwd=WORKSPACE_ROOT, env=env, allow_failure=True)
            parsed = _parse_integrity_report(report_text, returncode=eval_rc)

            row = {
                "temperature": tr.temperature,
                "hard_neg_mode": tr.hard_neg_mode,
                "hard_neg_weight": tr.hard_neg_weight,
                "hard_neg_margin": tr.hard_neg_margin,
                "seed": tr.seed,
                "translation_exact": parsed.translation_exact,
                "translation_exact_ok": parsed.translation_exact_ok,
                "translation_exact_total": parsed.translation_exact_total,
                "micro_f1_all": parsed.micro_f1_all,
                "micro_f1_tgt_exact": parsed.micro_f1_tgt_exact,
                "eval_returncode": parsed.returncode,
                "pa_output": str(out_pa),
            }
            writer.writerow(row)
            f.flush()

            if parsed.micro_f1_tgt_exact is not None:
                if best is None or parsed.micro_f1_tgt_exact > best[0]:
                    best = (parsed.micro_f1_tgt_exact, row)
            elif parsed.micro_f1_all is not None:
                if best is None or parsed.micro_f1_all > best[0]:
                    best = (parsed.micro_f1_all, row)

        print("\n완료: " + str(out_csv))
        if best:
            print("Best micro_f1 (우선: tgt 완전일치 subset):")
            print(best[0])
            print(best[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
