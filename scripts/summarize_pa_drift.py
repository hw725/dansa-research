#!/usr/bin/env python3
"""Summarize boundary-only drift vs a reference PA output for existing sweep artifacts.

- PA 추론을 다시 돌리지 않고(=재실행 없이), 이미 생성된 PA 출력 CSV들을 대상으로
  reference 출력과의 boundary drift 통계를 계산해 요약 CSV로 저장한다.
- 옵션으로 gold를 주면 integrity_report.py를 실행해 micro_f1 값도 함께 기록한다.

예)
  docker compose exec -T csp bash -lc "python -u scripts/summarize_pa_drift.py \
    --reference-output test_results/pa_strict_thr0p72_ml10_seed1_adjref_adaptive.csv \
    --inputs-glob 'test_results/sweep_threshold_runs/pa_strict_thr*.csv' \
    --gold datasets/p2s/test_100_from_pd.csv"
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

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    # integrity_report.py와 동일 정규화
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


@dataclass(frozen=True)
class PaOutputGroup:
    src_sentences: list[str]
    tgt_sentences: list[str]


def _load_pa_output_groups(path: Path) -> dict[tuple[str, int], PaOutputGroup]:
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
    boundaries: set[int] = set()
    pos = 0
    for i, s in enumerate(src_sentences):
        pos += len(_norm(s))
        if i != len(src_sentences) - 1:
            boundaries.add(pos)
    return boundaries


def _compute_drift(output_groups: dict[tuple[str, int], PaOutputGroup], ref_groups: dict[tuple[str, int], PaOutputGroup]) -> dict[str, int | float]:
    common_keys = set(output_groups.keys()) & set(ref_groups.keys())

    tgt_equal = 0
    src_concat_equal = 0
    boundary_equal = 0
    boundary_only = 0
    symdiff_total = 0

    for k in common_keys:
        g = output_groups[k]
        r = ref_groups[k]

        is_tgt_equal = [_norm(x) for x in g.tgt_sentences] == [_norm(x) for x in r.tgt_sentences]
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
            boundary_only += 1
            symdiff_total += len(g_b.symmetric_difference(r_b))

    symdiff_mean = (symdiff_total / boundary_only) if boundary_only else 0.0

    return {
        "common_pids": len(common_keys),
        "tgt_equal_pids": tgt_equal,
        "src_concat_equal_pids": src_concat_equal,
        "boundary_equal_pids": boundary_equal,
        "boundary_only_pids": boundary_only,
        "boundary_symdiff_total": symdiff_total,
        "boundary_symdiff_mean": round(symdiff_mean, 6),
        "missing_in_ref": len(set(output_groups.keys()) - set(ref_groups.keys())),
        "missing_in_output": len(set(ref_groups.keys()) - set(output_groups.keys())),
    }


@dataclass(frozen=True)
class ParsedEval:
    translation_exact_ok: int | None
    translation_exact_total: int | None
    micro_f1_all: float | None
    micro_f1_tgt_exact: float | None
    returncode: int


_F1_ALL_RE = re.compile(r"\(micro,\s*전체\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")
_F1_TGT_EXACT_RE = re.compile(r"\(micro,\s*tgt\s*완전일치\s*subset\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")
_F1_LEGACY_RE = re.compile(r"\(micro\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")
_TRANSL_RE = re.compile(r"번역문 문장리스트 완전일치:\s*(\d+)\s*/\s*(\d+)")


def _run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=str(cwd), env=env, text=True, capture_output=True)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr)


def _parse_integrity_report(text: str, *, returncode: int) -> ParsedEval:
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


_FILENAME_THR_RE = re.compile(r"thr(?P<thr>[0-9]+(?:\.[0-9]+)?)")
_FILENAME_REPEAT_RE = re.compile(r"_r(?P<repeat>\d+)")


def _infer_threshold_from_name(name: str) -> str | None:
    m = _FILENAME_THR_RE.search(name)
    return m.group("thr") if m else None


def _infer_repeat_from_name(name: str) -> int | None:
    m = _FILENAME_REPEAT_RE.search(name)
    return int(m.group("repeat")) if m else None


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize PA boundary drift vs reference")
    p.add_argument(
        "--reference-output",
        required=True,
        help="reference PA 출력 CSV 경로 (container 기준, 예: test_results/pa_strict_thr0p72_ml10_seed1_adjref_adaptive.csv)",
    )
    p.add_argument(
        "--inputs-glob",
        default="test_results/sweep_threshold_runs/pa_strict_thr*.csv",
        help="요약할 PA 출력 파일 glob (container 기준)",
    )
    p.add_argument(
        "--gold",
        default=None,
        help="(옵션) gold 경로를 주면 integrity_report.py를 실행해 micro_f1도 같이 기록",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="출력 CSV 경로(미지정 시 test_results/pa_drift_summary_<ts>.csv)",
    )

    args = p.parse_args()

    ref_path = Path(args.reference_output)
    if not ref_path.is_absolute():
        ref_path = WORKSPACE_ROOT / ref_path
    if not ref_path.exists():
        raise SystemExit(f"reference-output not found: {ref_path}")

    inputs = sorted((WORKSPACE_ROOT / args.inputs_glob).parent.glob(Path(args.inputs_glob).name))
    # 위 glob은 마지막 패턴만 반영되므로, ** 같은 고급 glob은 지원하지 않음
    # 필요하면 inputs-glob을 폴더 + 단일 패턴으로 넘겨주세요.

    if not inputs:
        raise SystemExit(f"no inputs matched: {args.inputs_glob}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.out_csv) if args.out_csv else (WORKSPACE_ROOT / "test_results" / f"pa_drift_summary_{ts}.csv")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[drift] reference={ref_path}")
    print(f"[drift] inputs={len(inputs)} files")
    print(f"[drift] out={out_csv}")

    ref_groups = _load_pa_output_groups(ref_path)

    fieldnames = [
        "file",
        "threshold",
        "repeat_idx",
        "common_pids",
        "tgt_equal_pids",
        "src_concat_equal_pids",
        "boundary_equal_pids",
        "boundary_only_pids",
        "boundary_symdiff_total",
        "boundary_symdiff_mean",
        "missing_in_ref",
        "missing_in_output",
    ]
    if args.gold:
        fieldnames.extend(
            [
                "translation_exact_ok",
                "translation_exact_total",
                "micro_f1_all",
                "micro_f1_tgt_exact",
                "eval_returncode",
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, path in enumerate(inputs, start=1):
            print(f"[{i}/{len(inputs)}] {path}")
            groups = _load_pa_output_groups(path)
            drift = _compute_drift(groups, ref_groups)

            row: dict[str, object] = {
                "file": str(path),
                "threshold": _infer_threshold_from_name(path.name),
                "repeat_idx": _infer_repeat_from_name(path.name),
                **drift,
            }

            if args.gold:
                gold_path = Path(args.gold)
                if not gold_path.is_absolute():
                    gold_path = WORKSPACE_ROOT / gold_path
                eval_cmd = [
                    sys.executable,
                    str(WORKSPACE_ROOT / "integrity_report.py"),
                    "--input",
                    str(path),
                    "--gold",
                    str(gold_path),
                ]
                rc, text = _run(eval_cmd, cwd=WORKSPACE_ROOT, env=env)
                parsed = _parse_integrity_report(text, returncode=rc)
                row.update(
                    {
                        "translation_exact_ok": parsed.translation_exact_ok,
                        "translation_exact_total": parsed.translation_exact_total,
                        "micro_f1_all": parsed.micro_f1_all,
                        "micro_f1_tgt_exact": parsed.micro_f1_tgt_exact,
                        "eval_returncode": parsed.returncode,
                    }
                )

            w.writerow(row)
            f.flush()

    print(f"\n완료: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
