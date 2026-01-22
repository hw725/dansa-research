#!/usr/bin/env python3
"""End-to-end (TRAIN-ONLY): P2S alignment 학습(train.csv만) → P2S(strict) 실행 → gold F1 평가.

원칙
- 학습: datasets/p2s/train.csv만 사용 (PD는 학습하지 않음)
- negative: 별도 파일 없이 in-batch contrastive로 train에서 자동 생성
- 평가 입력: PD는 "P2S 실행을 위한 입력 문단"으로만 사용

권장 실행(도커)
  docker-compose run --rm csp python scripts/train_p2s_boundary.py --enable-hard-neg

스모크(빠르게)
  docker-compose run --rm csp python scripts/train_p2s_boundary.py --enable-hard-neg --max-steps 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CmdResult:
    argv: Sequence[str]
    returncode: int


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> CmdResult:
    print("\n$ " + " ".join(argv))
    proc = subprocess.run(argv, cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"명령 실패(returncode={proc.returncode}): {' '.join(argv)}")
    return CmdResult(argv=argv, returncode=proc.returncode)


def _run_allow(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    ok_returncodes: set[int] | None = None,
) -> CmdResult:
    """특정 returncode를 허용하면서 명령을 실행한다.

    p2s_evaluator.py는 '불일치 항목 존재'를 returncode=2로 표현할 수 있는데,
    이 경우에도 리포트/점수 출력은 유효하므로 파이프라인을 중단하지 않기 위해 사용한다.
    """
    if ok_returncodes is None:
        ok_returncodes = {0}
    print("\n$ " + " ".join(argv))
    proc = subprocess.run(argv, cwd=str(cwd), env=env, check=False)
    if proc.returncode not in ok_returncodes:
        raise SystemExit(f"명령 실패(returncode={proc.returncode}): {' '.join(argv)}")
    if proc.returncode != 0:
        print(f"\n[warn] returncode={proc.returncode} 허용됨: {' '.join(argv)}")
    return CmdResult(argv=argv, returncode=proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-only P2S alignment + P2S strict + gold eval")

    # train (P2S sentence-parallel)
    parser.add_argument(
        "--train-csv",
        default=str(WORKSPACE_ROOT / "datasets" / "p2s" / "train.csv"),
        help="P2S 학습용 train.csv (기본: datasets/p2s/train.csv)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="재현성 시드(0이면 시드 고정 안 함). alignment 학습에 전달됨.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="가능한 범위에서 결정론적 실행을 강제(재현성 강화). P2S 실행에도 전달됨.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="alignment 학습을 건너뛰고 기존 models/dual_encoder_alignment_p2s.pt로 P2S+평가만 수행",
    )

    parser.add_argument("--enable-hard-neg", action="store_true")
    parser.add_argument("--hard-neg-mode", default="prefix_token", choices=["prefix_token", "full"])
    parser.add_argument("--hard-neg-weight", type=float, default=0.5)
    parser.add_argument("--hard-neg-margin", type=float, default=0.15)

    # eval inputs (PD paragraphs + gold sentences)
    parser.add_argument(
        "--pd-input",
        default=str(WORKSPACE_ROOT / "datasets" / "pd" / "test_10.csv"),
        help="P2S 입력(문단병렬). PD는 학습이 아니라 입력용 (기본: datasets/pd/test_10.csv)",
    )

    default_gold_100 = WORKSPACE_ROOT / "datasets" / "p2s" / "test_100.csv"
    default_gold_10 = WORKSPACE_ROOT / "test_results" / "gold_subset_from_pd_test10.csv"
    default_gold = default_gold_100 if default_gold_100.exists() else default_gold_10
    parser.add_argument(
        "--gold",
        default=str(default_gold),
        help="gold(문장 단위) (기본: datasets/p2s/test_100.csv가 있으면 우선 사용, 없으면 test_results/gold_subset_from_pd_test10.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(WORKSPACE_ROOT / "test_results"),
        help="출력 디렉토리 (기본: test_results/)",
    )
    parser.add_argument(
        "--pids",
        nargs="*",
        default=None,
        help="평가 pid 목록(미지정 시 전체 pid를 평가)",
    )
    parser.add_argument("--book-name", default=None, help="(선택) book_name 필터 (gold 비교용)")

    # P2S strict
    parser.add_argument("--boundary-threshold", type=float, default=0.72)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_p2s_csv = out_dir / f"p2s_strict_{ts}.csv"

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    # 1) Train alignment model (train-only)
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "train_p2s_alignment_dual_encoder.py"),
            "--train-csv",
            str(args.train_csv),
            "--epochs",
            str(int(args.epochs)),
            "--batch",
            str(int(args.batch)),
            "--max-steps",
            str(int(args.max_steps)),
            "--temperature",
            str(float(args.temperature)),
            "--seed",
            str(int(args.seed)),
            "--hard-neg-mode",
            str(args.hard_neg_mode),
            "--hard-neg-weight",
            str(float(args.hard_neg_weight)),
            "--hard-neg-margin",
            str(float(args.hard_neg_margin)),
        ]
        if args.enable_hard_neg:
            train_cmd.append("--enable-hard-neg")
        _run(train_cmd, cwd=WORKSPACE_ROOT, env=env)
    else:
        print("\n[skip] alignment 학습 생략 (--skip-train)")

    # 2) Run P2S strict on eval paragraphs
    # 혼선 방지: PD 입력(test_100_from_pd) 계열은 gold도 from_pd를 쓰는 게 일반적이다.
    try:
        pd_input_name = Path(str(args.pd_input)).name
        gold_name = Path(str(args.gold)).name
        if pd_input_name == "test_100.csv" and gold_name == "test_100.csv":
            from_pd_gold = WORKSPACE_ROOT / "datasets" / "p2s" / "test_100_from_pd.csv"
            if from_pd_gold.exists():
                print(
                    "\n[warn] 현재 입력은 datasets/pd/test_100.csv인데 gold를 datasets/p2s/test_100.csv로 지정했습니다."
                    "\n       PD→P2S 평가 기준 리포트와 점수가 다르게 나올 수 있습니다."
                    f"\n       (참고) 이 입력에 흔히 대응하는 gold: {from_pd_gold}"
                )
    except Exception:
        pass

    p2s_cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "p2s" / "main.py"),
        str(args.pd_input),
        str(out_p2s_csv),
        "--embedder",
        "bge",
        "--use-boundary-model",
        "--boundary-threshold",
        str(float(args.boundary_threshold)),
        "--device",
        str(args.device),
    ]

    # 재현성 옵션: P2S 실행에도 전달
    if int(args.seed) != 0:
        p2s_cmd.extend(["--seed", str(int(args.seed))])
    if args.deterministic:
        p2s_cmd.append("--deterministic")

    _run(p2s_cmd, cwd=WORKSPACE_ROOT, env=env)

    # 3) Eval vs gold
    eval_cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "accuracy" / "p2s_evaluator.py"),
        "--input",
        str(out_p2s_csv),
        "--gold",
        str(args.gold),
    ]
    if args.pids:
        eval_cmd.extend(["--pids", *[str(x) for x in args.pids]])
    if args.book_name:
        eval_cmd.extend(["--book-name", str(args.book_name)])

    _run_allow(eval_cmd, cwd=WORKSPACE_ROOT, env=env, ok_returncodes={0, 2})

    print("\n✅ 완료")
    print(f"- P2S output: {out_p2s_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

