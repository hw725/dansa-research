#!/usr/bin/env python3
"""Sweep PA boundary threshold while keeping alignment model fixed.

목적
- 번역문 문장리스트가 gold와 일치하는 subset에서("tgt exact subset")
  원문 경계 micro-F1(micro_f1_tgt_exact)을 최대화하는 boundary threshold를 찾는다.

이 스윕은 누수와 무관 (학습 없음). 이미 학습된 모델을 사용해 추론+평가만 한다.

예)
  docker-compose run --rm csp python scripts/sweep_pa_boundary_threshold.py \
    --pd-input datasets/sentenceragraph/test_100.csv \
    --gold datasets/p2s/test_100.csv \
    --thresholds 0.3 0.4 0.5 0.6 0.7 \
    --device cuda

출력
- test_results/sweep_pa_boundary_threshold_<timestamp>.csv
- trial별 PA 출력은 test_results/sweep_threshold_runs/ 아래에 저장
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
from pathlib import Path
from statistics import mean, median


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ParsedEval:
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
    ok = total = None
    m = _TRANSL_RE.search(text)
    if m:
        ok, total = int(m.group(1)), int(m.group(2))

    micro_f1_all = None
    m_all = _F1_ALL_RE.search(text)
    if m_all:
        micro_f1_all = float(m_all.group(3))
    else:
        m_legacy = _F1_LEGACY_RE.search(text)
        if m_legacy:
            micro_f1_all = float(m_legacy.group(3))

    micro_f1_tgt_exact = None
    m_ok = _F1_TGT_EXACT_RE.search(text)
    if m_ok:
        micro_f1_tgt_exact = float(m_ok.group(3))

    return ParsedEval(
        translation_exact_ok=ok,
        translation_exact_total=total,
        micro_f1_all=micro_f1_all,
        micro_f1_tgt_exact=micro_f1_tgt_exact,
        returncode=returncode,
    )


@dataclass(frozen=True)
class PaOutputGroup:
    src_sentences: list[str]
    tgt_sentences: list[str]


def _norm(s: str) -> str:
    # integrity_report.py와 동일한 정규화(공백/개행/탭 제거)
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


def _load_pa_output_groups(path: Path) -> dict[tuple[str, int], PaOutputGroup]:
    """PA 출력 CSV를 (book_name, pid) -> (src_sentence_list, tgt_sentence_list)로 로드."""

    groups: dict[tuple[str, int], list[tuple[int, str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"문단식별자", "book_name", "문장식별자", "원문", "번역문"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"PA 출력 CSV 스키마가 예상과 다릅니다. missing={sorted(missing)} path={path}")

        for row in reader:
            book = str(row["book_name"])
            pid = int(row["문단식별자"])
            sid = int(row["문장식별자"])
            src = str(row["원문"])
            tgt = str(row["번역문"])
            groups.setdefault((book, pid), []).append((sid, src, tgt))

    out: dict[tuple[str, int], PaOutputGroup] = {}
    for key, triples in groups.items():
        triples.sort(key=lambda x: x[0])
        out[key] = PaOutputGroup(
            src_sentences=[t[1] for t in triples],
            tgt_sentences=[t[2] for t in triples],
        )
    return out


def _boundary_set_from_src_sentences(src_sentences: list[str]) -> set[int]:
    """src sentence list로부터 boundary 위치(문자 오프셋) 집합을 만든다.

    - boundary는 '문장 i 끝'의 누적 길이로 정의
    - 마지막 문장 끝(전체 길이)은 boundary로 포함하지 않음
    """

    boundaries: set[int] = set()
    pos = 0
    for i, s in enumerate(src_sentences):
        pos += len(_norm(s))
        if i != len(src_sentences) - 1:
            boundaries.add(pos)
    return boundaries


def _compute_drift_vs_reference(
    *,
    pa_output: Path,
    ref_groups: dict[tuple[str, int], PaOutputGroup],
    cache: dict[Path, dict[str, float | int | str | None]],
) -> dict[str, float | int | str | None]:
    """pa_output를 reference와 비교해 boundary-only drift 통계를 계산."""

    if pa_output in cache:
        return cache[pa_output]

    groups = _load_pa_output_groups(pa_output)

    common_keys = set(groups.keys()) & set(ref_groups.keys())
    tgt_equal = 0
    src_concat_equal = 0
    boundary_equal = 0
    boundary_only_drift = 0
    boundary_symdiff_total = 0

    for k in common_keys:
        g = groups[k]
        r = ref_groups[k]

        # integrity_report 기준과 동일하게 비교는 정규화 기준으로 수행
        is_tgt_equal = [ _norm(x) for x in g.tgt_sentences ] == [ _norm(x) for x in r.tgt_sentences ]
        if is_tgt_equal:
            tgt_equal += 1

        g_src_concat = "".join(_norm(x) for x in g.src_sentences)
        r_src_concat = "".join(_norm(x) for x in r.src_sentences)
        is_src_concat_equal = g_src_concat == r_src_concat
        if is_src_concat_equal:
            src_concat_equal += 1

        g_b = _boundary_set_from_src_sentences(g.src_sentences)
        r_b = _boundary_set_from_src_sentences(r.src_sentences)
        is_boundary_equal = g_b == r_b
        if is_boundary_equal:
            boundary_equal += 1

        if is_tgt_equal and is_src_concat_equal and not is_boundary_equal:
            boundary_only_drift += 1
            boundary_symdiff_total += len(g_b.symmetric_difference(r_b))

    boundary_symdiff_mean = (
        (boundary_symdiff_total / boundary_only_drift) if boundary_only_drift > 0 else 0.0
    )

    stats: dict[str, float | int | str | None] = {
        "drift_ref": str(pa_output),  # placeholder, overwritten by caller
        "drift_common_pids": len(common_keys),
        "drift_tgt_equal_pids": tgt_equal,
        "drift_src_concat_equal_pids": src_concat_equal,
        "drift_boundary_equal_pids": boundary_equal,
        "drift_boundary_only_pids": boundary_only_drift,
        "drift_boundary_symdiff_total": boundary_symdiff_total,
        "drift_boundary_symdiff_mean": round(boundary_symdiff_mean, 6),
        "drift_missing_in_ref": len(set(groups.keys()) - set(ref_groups.keys())),
        "drift_missing_in_output": len(set(ref_groups.keys()) - set(groups.keys())),
    }
    cache[pa_output] = stats
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep PA boundary threshold (no training)")

    p.add_argument(
        "--pd-input",
        default=str(WORKSPACE_ROOT / "datasets" / "pd" / "test_100.csv"),
        help="PA 입력(문단병렬). 기본: datasets/sentenceragraph/test_100.csv",
    )
    p.add_argument(
        "--gold",
        default=str(WORKSPACE_ROOT / "datasets" / "pa" / "test_100.csv"),
        help="gold(문장 단위). 기본: datasets/p2s/test_100.csv",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="boundary threshold 목록",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="threshold 당 반복 실행 횟수(경계 drift가 있을 때 분포 확인용)",
    )
    p.add_argument(
        "--rank-by",
        default="median",
        choices=["median", "max", "mean"],
        help=(
            "threshold 간 우열을 정할 때 repeats 결과를 어떻게 집계할지. "
            "tgt_exact 점수가 있으면 tgt_exact를, 없으면 all을 사용"
        ),
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="PA max-length (p2s/main.py로 전달). 예: 기존 strict 재현용으로 10을 사용",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="PA max-workers (p2s/main.py로 전달)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PA batch-size (p2s/main.py로 전달)",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out-dir", default=str(WORKSPACE_ROOT / "test_results"))
    p.add_argument("--seed", type=int, default=None, help="PA 추론 재현성 seed (p2s/main.py로 전달)")
    p.add_argument(
        "--boundary-min-len",
        type=int,
        default=None,
        help="boundary 모델 디코딩 min_len 오버라이드(task=pa, 기본 20). p2s/main.py로 전달",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="PA 추론 deterministic 모드 사용 (p2s/main.py로 전달, 속도 저하 가능)",
    )

    default_ref = WORKSPACE_ROOT / "test_results" / "pa_strict_thr0p72_ml10_seed1_adjref_adaptive.csv"
    p.add_argument(
        "--reference-output",
        default=str(default_ref) if default_ref.exists() else None,
        help=(
            "boundary drift 비교용 reference PA 출력 CSV. "
            "지정 시 sweep CSV에 drift_* 컬럼이 추가됨"
        ),
    )

    args = p.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats는 1 이상이어야 합니다")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "sweep_threshold_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"sweep_pa_boundary_threshold_{ts}.csv"

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.seed is not None:
        # subprocess로 실행되는 Python의 해시 랜덤화를 고정 (set/dict 순서 비결정성 완화)
        env.setdefault("PYTHONHASHSEED", str(args.seed))

    ref_groups: dict[tuple[str, int], PaOutputGroup] | None = None
    drift_cache: dict[Path, dict[str, float | int | str | None]] = {}
    ref_path: Path | None = None
    if args.reference_output:
        ref_path = Path(args.reference_output)
        if not ref_path.is_absolute():
            ref_path = WORKSPACE_ROOT / ref_path
        if ref_path.exists():
            print(f"[drift] reference_output={ref_path}")
            ref_groups = _load_pa_output_groups(ref_path)
        else:
            print(f"[drift] reference_output not found: {ref_path} (skip drift metrics)")
            ref_path = None

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "boundary_threshold",
            "repeat_idx",
            "translation_exact_ok",
            "translation_exact_total",
            "micro_f1_all",
            "micro_f1_tgt_exact",
            "eval_returncode",
            "pa_output",
        ]
        if ref_groups is not None and ref_path is not None:
            fieldnames.extend(
                [
                    "reference_output",
                    "drift_common_pids",
                    "drift_tgt_equal_pids",
                    "drift_src_concat_equal_pids",
                    "drift_boundary_equal_pids",
                    "drift_boundary_only_pids",
                    "drift_boundary_symdiff_total",
                    "drift_boundary_symdiff_mean",
                    "drift_missing_in_ref",
                    "drift_missing_in_output",
                ]
            )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        best = None

        def _pick_score(parsed: ParsedEval) -> float | None:
            return parsed.micro_f1_tgt_exact if parsed.micro_f1_tgt_exact is not None else parsed.micro_f1_all

        def _aggregate(values: list[float]) -> float:
            if args.rank_by == "max":
                return max(values)
            if args.rank_by == "mean":
                return mean(values)
            return median(values)

        for thr in [float(x) for x in args.thresholds]:
            print("\n" + "=" * 80)
            print(f"threshold={thr}")
            print("=" * 80)

            scores: list[float] = []
            best_single: tuple[float, dict] | None = None

            for repeat_idx in range(1, int(args.repeats) + 1):
                out_pa = runs_dir / f"pa_strict_thr{thr}_{ts}_r{repeat_idx}.csv"
                pa_cmd = [
                    sys.executable,
                    str(WORKSPACE_ROOT / "pa" / "main.py"),
                    str(args.pd_input),
                    str(out_pa),
                    "--embedder",
                    "bge",
                    "--use-boundary-model",
                    "--boundary-threshold",
                    str(thr),
                    "--device",
                    str(args.device),
                ]
                if args.max_length is not None:
                    pa_cmd.extend(["--max-length", str(int(args.max_length))])
                if args.max_workers is not None:
                    pa_cmd.extend(["--max-workers", str(int(args.max_workers))])
                if args.batch_size is not None:
                    pa_cmd.extend(["--batch-size", str(int(args.batch_size))])
                if args.seed is not None:
                    pa_cmd.extend(["--seed", str(args.seed)])
                if args.boundary_min_len is not None:
                    pa_cmd.extend(["--boundary-min-len", str(args.boundary_min_len)])
                if args.deterministic:
                    pa_cmd.append("--deterministic")
                _run(pa_cmd, cwd=WORKSPACE_ROOT, env=env)

                eval_cmd = [
                    sys.executable,
                    str(WORKSPACE_ROOT / "integrity_report.py"),
                    "--input",
                    str(out_pa),
                    "--gold",
                    str(args.gold),
                ]
                eval_rc, report_text = _run(eval_cmd, cwd=WORKSPACE_ROOT, env=env, allow_failure=True)
                parsed = _parse_integrity_report(report_text, returncode=eval_rc)

                row: dict[str, float | int | str | None] = {
                    "boundary_threshold": thr,
                    "repeat_idx": repeat_idx,
                    "translation_exact_ok": parsed.translation_exact_ok,
                    "translation_exact_total": parsed.translation_exact_total,
                    "micro_f1_all": parsed.micro_f1_all,
                    "micro_f1_tgt_exact": parsed.micro_f1_tgt_exact,
                    "eval_returncode": parsed.returncode,
                    "pa_output": str(out_pa),
                }

                if ref_groups is not None and ref_path is not None:
                    drift = _compute_drift_vs_reference(
                        pa_output=out_pa,
                        ref_groups=ref_groups,
                        cache=drift_cache,
                    )
                    drift = dict(drift)
                    drift["drift_ref"] = str(ref_path)
                    row["reference_output"] = str(ref_path)
                    for k in [
                        "drift_common_pids",
                        "drift_tgt_equal_pids",
                        "drift_src_concat_equal_pids",
                        "drift_boundary_equal_pids",
                        "drift_boundary_only_pids",
                        "drift_boundary_symdiff_total",
                        "drift_boundary_symdiff_mean",
                        "drift_missing_in_ref",
                        "drift_missing_in_output",
                    ]:
                        row[k] = drift.get(k)

                writer.writerow(row)
                f.flush()

                score = _pick_score(parsed)
                if score is not None:
                    scores.append(score)
                    if best_single is None or score > best_single[0]:
                        best_single = (score, row)

            if scores:
                agg = _aggregate(scores)
                print(
                    f"[summary] thr={thr} repeats={len(scores)} "
                    f"min={min(scores):.4f} median={median(scores):.4f} max={max(scores):.4f} mean={mean(scores):.4f} "
                    f"rank_by={args.rank_by} => {agg:.4f}"
                )

                if best is None or agg > best[0]:
                    best = (agg, best_single[1] if best_single is not None else {"boundary_threshold": thr})

        print("\n완료: " + str(out_csv))
        if best:
            print("Best (우선: micro_f1_tgt_exact):")
            print(best[0])
            print(best[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
