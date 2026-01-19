#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""여러 test_100을 만들고(또는 사용하고) PA(strict) 평가를 반복 실행.

핵심 목표(B):
- test_100 한 번의 점수에 의존하지 않고, seed별로 여러 번 돌려 평균/분산을 본다.

동작:
1) PD pool에서 (book_name, 문단식별자) 기준으로 N개 샘플링하여 pd_subset.csv 생성
2) PA gold(pool)에서 동일 키만 추출하여 gold_subset.csv 생성 (integrity_report.extract_gold_subset 사용)
3) docker compose로 p2s/main.py 실행하여 PA output 생성
4) integrity_report로 평가하고 (micro, tgt 완전일치 subset) F1을 파싱해 요약 CSV 저장

중요)
- 이 스크립트는 내부에서 `docker compose ...`를 호출하므로, 컨테이너 안에서 실행하면 안 됩니다.
    (즉, `docker compose run csp python scripts/pa_multitest_runner.py` 형태로 실행하지 마세요.)

사용 예)
    python scripts/pa_multitest_runner.py --seeds 1 2 3 4 5

  # 공식 재현 파라미터를 기본값으로 포함 (thr=0.70, min-len=20 등)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# scripts/ 아래에서 실행되면 sys.path[0]이 scripts로 잡혀 루트 모듈 import가 깨질 수 있어 보정
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _to_container_path(p: Path) -> str:
    rel = p.resolve().relative_to(WORKSPACE_ROOT.resolve())
    return "/workspace/" + rel.as_posix()


def _is_running_in_docker() -> bool:
    # 가장 신뢰할 수 있는 신호: /.dockerenv
    try:
        if Path("/.dockerenv").exists():
            return True
    except Exception:
        pass
    # 명시적 오버라이드(테스트/CI 용)
    if os.getenv("CSP_IN_DOCKER", "").strip() in ("1", "true", "True", "yes", "YES"):
        return True
    return False


def _to_run_path(p: Path) -> str:
    """현재 실행 환경에 맞는 경로 문자열을 반환.

    - 호스트 실행(Windows 등): docker compose run을 호출하므로 /workspace/... 경로 필요
    - 컨테이너 실행: 동일 컨테이너에서 python을 직접 실행하므로 로컬 경로(str(Path)) 사용
    """

    if _is_running_in_docker():
        return str(p)
    return _to_container_path(p)


_F1_TGT_EXACT_RE = re.compile(
    r"\(micro,\s*tgt\s*완전일치\s*subset\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)"
)
_F1_TGT_EXACT_NFC_RE = re.compile(
    r"\(micro,\s*tgt\s*완전일치\s*subset,\s*NFC\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)"
)
_TRANSL_RE = re.compile(r"번역문 문장리스트 완전일치:\s*(\d+)\s*/\s*(\d+)")
_TRANSL_NFC_RE = re.compile(r"번역문 문장리스트 완전일치\(NFC\):\s*(\d+)\s*/\s*(\d+)")
_TRANSL_SENT_RE = re.compile(r"번역문 문장 완전일치\(문장\):\s*(\d+)\s*/\s*(\d+)")
_TRANSL_SENT_NFC_RE = re.compile(r"번역문 문장 완전일치\(문장,NFC\):\s*(\d+)\s*/\s*(\d+)")
_SRC_SIM_OK_MEAN_RE = re.compile(r"원문 유사도\(SequenceMatcher,\s*tgt문장일치\s*subset\):\s*mean=([0-9.]+)")


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    allow_failure: bool = False,
    capture: bool = True,
    echo: bool = True,
    log_path: Path | None = None,
) -> tuple[int, str]:
    print("\n$ " + " ".join(argv))
    if not capture:
        proc = subprocess.run(argv, cwd=str(cwd), env=env)
        stdout = ""
        stderr = ""
    else:
        # Windows에서 기본 인코딩(cp949)으로 docker 출력 디코딩이 깨질 수 있어 bytes로 받아 안전 디코딩
        proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True)
        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        if echo:
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)

    merged = (stdout + "\n" + stderr).strip()
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(merged + "\n", encoding="utf-8")
        except Exception:
            pass
    if proc.returncode != 0 and not allow_failure:
        raise SystemExit(f"명령 실패(returncode={proc.returncode}): {' '.join(argv)}")
    return proc.returncode, merged


def _run_stream(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    allow_failure: bool = False,
    echo: bool = True,
    log_path: Path | None = None,
    tail_lines: int = 200,
) -> tuple[int, str]:
    """stdout/stderr를 실시간으로 스트리밍하며 실행한다.

    - docker compose run 같이 출력이 길고 오래 걸리는 작업에서 "멈춘 것처럼" 보이는 UX를 방지.
    - 반환 텍스트는 마지막 tail_lines 라인만(메모리 보호).
    """

    print("\n$ " + " ".join(argv))
    fp = None
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fp = log_path.open("w", encoding="utf-8")
        except Exception:
            fp = None

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        tails: list[str] = []
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            if raw is None:
                break
            line = raw.decode("utf-8", errors="replace")
            if fp is not None:
                try:
                    fp.write(line)
                    fp.flush()
                except Exception:
                    pass
            if echo:
                sys.stdout.write(line)

            tails.append(line.rstrip("\n"))
            if len(tails) > int(tail_lines):
                tails = tails[-int(tail_lines) :]

        rc = proc.wait()
        tail_text = "\n".join(tails).strip()
        if rc != 0 and not allow_failure:
            raise SystemExit(f"명령 실패(returncode={rc}): {' '.join(argv)}")
        return int(rc), tail_text
    finally:
        try:
            if fp is not None:
                fp.close()
        except Exception:
            pass


@dataclass
class RunnerConfig:
    pd_pool: str = "datasets/sentenceragraph/test.csv"
    pa_gold_pool: str = "datasets/p2s/test.csv"
    n: int = 100
    seeds: list[int] = None  # type: ignore[assignment]
    out_dir: str = "test_results/multitest"
    run_dir: str | None = None
    eval_only: bool = False
    enable_refine: bool = False
    enable_src_marker_bonus: bool = False
    enable_src_marker_whitespace_dp_bonus: bool = False
    device: str = "cuda"
    max_length: int = 200
    boundary_threshold: float = 0.70
    boundary_min_len: int = 20
    deterministic: bool = True
    pyhashseed: int = 1
    no_trace: bool = False
    stream: bool = False
    analyze_mismatch: bool = False

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [1, 2, 3, 4, 5]


def _strip_jsonc_comments(text: str) -> str:
    """JSONC 스타일 주석(//, /* */)을 제거한다.

    - 문자열 리터럴 내부의 //, /* */는 보존한다.
    - 사용자가 config에 한국어 주석을 달 수 있도록 하기 위한 유틸.
    """

    text = text.lstrip("\ufeff")
    out: list[str] = []
    i = 0
    in_str = False
    quote = ""
    n = len(text)

    while i < n:
        ch = text[i]

        if in_str:
            out.append(ch)
            if ch == "\\":
                # escape 다음 문자는 그대로
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
            elif ch == quote:
                in_str = False
                quote = ""
            i += 1
            continue

        # 문자열 시작
        if ch == '"':
            in_str = True
            quote = '"'
            out.append(ch)
            i += 1
            continue

        # // 라인 주석
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        # /* */ 블록 주석
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = i + 2 if i + 1 < n else n
            continue

        out.append(ch)
        i += 1

    cleaned = "".join(out)
    # trailing comma 허용: { "a": 1, } / [1,2,]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def _load_config(path: Path) -> RunnerConfig:
    try:
        raw = path.read_text(encoding="utf-8")
        raw = _strip_jsonc_comments(raw)
        data = json.loads(raw)
    except Exception as e:
        raise SystemExit(f"config 로드 실패(JSON/JSONC): {path} ({e})")

    if not isinstance(data, dict):
        raise SystemExit(f"config 형식 오류: 최상위는 object여야 합니다: {path}")

    cfg = RunnerConfig()
    for k, v in data.items():
        if not hasattr(cfg, k):
            raise SystemExit(f"config에 알 수 없는 키가 있습니다: {k} (path={path})")
        setattr(cfg, k, v)

    # 타입 정규화
    cfg.n = int(cfg.n)
    cfg.seeds = [int(x) for x in (cfg.seeds or [])]
    cfg.max_length = int(cfg.max_length)
    cfg.boundary_min_len = int(cfg.boundary_min_len)
    cfg.boundary_threshold = float(cfg.boundary_threshold)
    cfg.pyhashseed = int(cfg.pyhashseed)
    cfg.enable_refine = bool(cfg.enable_refine)
    cfg.enable_src_marker_bonus = bool(cfg.enable_src_marker_bonus)
    cfg.enable_src_marker_whitespace_dp_bonus = bool(cfg.enable_src_marker_whitespace_dp_bonus)
    cfg.deterministic = bool(cfg.deterministic)
    cfg.no_trace = bool(cfg.no_trace)
    cfg.stream = bool(cfg.stream)
    cfg.eval_only = bool(cfg.eval_only)
    cfg.run_dir = (str(cfg.run_dir) if cfg.run_dir is not None else None)
    return cfg


def _parse_eval(text: str) -> tuple[
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
    float | None,
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
]:
    f1 = None
    m = _F1_TGT_EXACT_RE.search(text)
    if m:
        f1 = float(m.group(3))

    f1_nfc = None
    m_nfc = _F1_TGT_EXACT_NFC_RE.search(text)
    if m_nfc:
        f1_nfc = float(m_nfc.group(3))

    ok = tot = None
    m2 = _TRANSL_RE.search(text)
    if m2:
        ok = int(m2.group(1))
        tot = int(m2.group(2))

    ok_nfc = tot_nfc = None
    m2n = _TRANSL_NFC_RE.search(text)
    if m2n:
        ok_nfc = int(m2n.group(1))
        tot_nfc = int(m2n.group(2))

    ok_s = tot_s = None
    m3 = _TRANSL_SENT_RE.search(text)
    if m3:
        ok_s = int(m3.group(1))
        tot_s = int(m3.group(2))

    ok_s_nfc = tot_s_nfc = None
    m3n = _TRANSL_SENT_NFC_RE.search(text)
    if m3n:
        ok_s_nfc = int(m3n.group(1))
        tot_s_nfc = int(m3n.group(2))

    sim_mean = None
    m4 = _SRC_SIM_OK_MEAN_RE.search(text)
    if m4:
        sim_mean = float(m4.group(1))
    return f1, ok, tot, ok_s, tot_s, sim_mean, f1_nfc, ok_nfc, tot_nfc, ok_s_nfc, tot_s_nfc


def _sample_pd_subset(pd_pool: Path, *, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(pd_pool).copy()
    required = {"문단식별자", "원문", "번역문", "book_name"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"PD pool에 필수 컬럼이 없습니다: {sorted(required - set(df.columns))}")
    df["문단식별자"] = df["문단식별자"].astype(int)
    df["book_name"] = df["book_name"].fillna("").astype(str)
    # 키 유일성 전제
    df = df.drop_duplicates(["book_name", "문단식별자"], keep="first")
    if len(df) < n:
        raise SystemExit(f"PD pool 크기({len(df)})가 n={n}보다 작습니다.")
    return df.sample(n=n, replace=False, random_state=int(seed)).sort_values(["book_name", "문단식별자"], kind="stable")


def main() -> int:
    p = argparse.ArgumentParser(description="Run PA strict on multiple random test splits")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="러너 설정 JSON/JSONC 경로(주석 //, /* */ 허용). CLI 인자는 config를 override합니다.",
    )

    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="기존 실행 디렉토리(예: test_results/.../YYYYmmdd_HHMMSS). --eval-only와 함께 쓰면 PA 재실행 없이 평가만 재계산합니다.",
    )

    p.add_argument(
        "--eval-only",
        action="store_true",
        help="PA 실행/샘플링 없이, 지정된 run-dir의 기존 pa_output/gold_subset으로 평가만 다시 계산합니다.",
    )

    # 아래 인자들은 config override 용도로 남겨둔다.
    p.add_argument("--pd-pool", default=None)
    p.add_argument("--pa-gold-pool", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--out-dir", default=None)

    p.add_argument(
        "--enable-refine",
        action="store_true",
        help="p2s/main.py의 --enable-refine를 켭니다(현재는 DP/인접 refine 이동폭 확장).",
    )

    p.add_argument(
        "--enable-src-marker-bonus",
        action="store_true",
        help=(
            "p2s/main.py의 --enable-src-marker-boundary-bonus를 켭니다. "
            "원문 내 현토(한글 marker) 패턴을 경계 선택 tie-break에 약하게 반영합니다."
        ),
    )

    p.add_argument(
        "--enable-src-marker-whitespace-dp-bonus",
        action="store_true",
        help=(
            "p2s/main.py의 --enable-src-marker-whitespace-dp-bonus를 켭니다. "
            "whitespace_dp(어절 DP 분할)에서 현토(한글 marker) 패턴을 후보 컷/DP 점수에 약하게 반영합니다."
        ),
    )

    p.add_argument(
        "--stream",
        action="store_true",
        help="PA 실행 로그를 실시간으로 출력합니다(긴 실행에서 '멈춘 것처럼' 보이는 문제 방지).",
    )

    # PA run params (official-ish defaults)
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--boundary-threshold", type=float, default=None)
    p.add_argument("--boundary-min-len", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--pyhashseed", type=int, default=None)
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="PA 실행 시 trace 파일 저장(--trace-stages-jsonl)을 비활성화합니다.",
    )
    p.add_argument(
        "--analyze-mismatch",
        action="store_true",
        help=(
            "모든 seed 실행/평가 후, 해당 run_dir에 대해 tgt mismatch 리포트를 추가 생성합니다. "
            "(scripts/collect_tgt_mismatch_cases.py + scripts/analyze_tgt_mismatch_subtypes.py)"
        ),
    )
    args = p.parse_args()

    # config 로드(선택)
    cfg = RunnerConfig()
    if args.config:
        cfg = _load_config((WORKSPACE_ROOT / str(args.config)).resolve() if not Path(str(args.config)).is_absolute() else Path(str(args.config)))
    else:
        # 기본 config 파일을 자동 로드(있을 때만)
        # - 팀에서 반복 실행할 때 매번 --config를 치지 않게 하기 위함
        default_cfg = WORKSPACE_ROOT / "scripts" / "pa_multitest_config.jsonc"
        if default_cfg.exists():
            cfg = _load_config(default_cfg)

    # CLI override 적용
    if args.pd_pool is not None:
        cfg.pd_pool = str(args.pd_pool)
    if args.pa_gold_pool is not None:
        cfg.pa_gold_pool = str(args.pa_gold_pool)
    if args.n is not None:
        cfg.n = int(args.n)
    if args.seeds is not None:
        cfg.seeds = [int(x) for x in args.seeds]
    if args.out_dir is not None:
        cfg.out_dir = str(args.out_dir)

    if args.run_dir is not None:
        cfg.run_dir = str(args.run_dir)
    if args.eval_only:
        cfg.eval_only = True

    # enable_refine/stream/no_trace/deterministic는 "켜기" 플래그만 제공
    if args.enable_refine:
        cfg.enable_refine = True
    if args.enable_src_marker_bonus:
        cfg.enable_src_marker_bonus = True
    if args.enable_src_marker_whitespace_dp_bonus:
        cfg.enable_src_marker_whitespace_dp_bonus = True
    if args.stream:
        cfg.stream = True
    if args.no_trace:
        cfg.no_trace = True
    if args.deterministic:
        cfg.deterministic = True
    if args.analyze_mismatch:
        cfg.analyze_mismatch = True

    if args.device is not None:
        cfg.device = str(args.device)
    if args.max_length is not None:
        cfg.max_length = int(args.max_length)
    if args.boundary_threshold is not None:
        cfg.boundary_threshold = float(args.boundary_threshold)
    if args.boundary_min_len is not None:
        cfg.boundary_min_len = int(args.boundary_min_len)
    if args.pyhashseed is not None:
        cfg.pyhashseed = int(args.pyhashseed)

    out_root = WORKSPACE_ROOT / str(cfg.out_dir)
    if cfg.run_dir:
        run_dir = (WORKSPACE_ROOT / str(cfg.run_dir)).resolve() if not Path(str(cfg.run_dir)).is_absolute() else Path(str(cfg.run_dir)).resolve()
        if not run_dir.exists():
            raise SystemExit(f"run_dir이 존재하지 않습니다: {run_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / ts
        run_dir.mkdir(parents=True, exist_ok=True)

    pd_pool = WORKSPACE_ROOT / str(cfg.pd_pool)
    gold_pool = WORKSPACE_ROOT / str(cfg.pa_gold_pool)

    # import here so the runner can be used both on host and in container
    if not cfg.eval_only:
        from integrity_report import extract_gold_subset

    summary_name = "summary_recalc.csv" if cfg.eval_only else "summary.csv"
    summary_path = run_dir / summary_name
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "pd_subset",
                "gold_subset",
                "pa_output",
                "tgt_exact_ok",
                "tgt_exact_total",
                "tgt_exact_nfc_ok",
                "tgt_exact_nfc_total",
                "tgt_exact_sent_ok",
                "tgt_exact_sent_total",
                "tgt_exact_sent_nfc_ok",
                "tgt_exact_sent_nfc_total",
                "src_sim_mean_tgt_sent_ok",
                "micro_f1_tgt_exact",
                "micro_f1_tgt_exact_nfc",
                "returncode",
            ],
        )
        w.writeheader()

        for seed in cfg.seeds:
            pd_subset_path = run_dir / f"pd_subset_n{cfg.n}_seed{seed}.csv"
            gold_subset_path = run_dir / f"pa_gold_subset_n{cfg.n}_seed{seed}.csv"
            pa_output_path = run_dir / f"pa_output_n{cfg.n}_seed{seed}.csv"

            if not cfg.eval_only:
                pd_subset_df = _sample_pd_subset(pd_pool, n=int(cfg.n), seed=int(seed))
                pd_subset_df.to_csv(pd_subset_path, index=False, encoding="utf-8-sig")

                # keys_from는 (문단식별자, book_name) 컬럼이 있어야 함 → pd_subset을 그대로 사용
                extract_gold_subset(
                    gold_path=gold_pool,
                    out_path=gold_subset_path,
                    keys_from=pd_subset_path,
                )
            pa_log_path = run_dir / f"pa_run_seed{seed}.log"
            pa_trace_path = run_dir / f"pa_trace_seed{seed}.jsonl"

            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(int(cfg.pyhashseed))

            if not cfg.eval_only:
                # 1) PA 실행
                # - 호스트: docker compose run 호출
                # - 컨테이너: 같은 컨테이너에서 python을 직접 실행(중첩 docker 금지)
                if _is_running_in_docker():
                    pa_cmd = [
                        "python",
                        "p2s/main.py",
                        _to_run_path(pd_subset_path),
                        _to_run_path(pa_output_path),
                        "--embedder",
                        "bge",
                        "--use-boundary-model",
                        "--boundary-threshold",
                        str(float(cfg.boundary_threshold)),
                        "--boundary-min-len",
                        str(int(cfg.boundary_min_len)),
                        "--max-length",
                        str(int(cfg.max_length)),
                        "--seed",
                        str(int(seed)),
                    ]
                else:
                    pa_cmd = [
                        "docker",
                        "compose",
                        "run",
                        "--rm",
                        "csp",
                        "python",
                        "p2s/main.py",
                        _to_run_path(pd_subset_path),
                        _to_run_path(pa_output_path),
                        "--embedder",
                        "bge",
                        "--use-boundary-model",
                        "--boundary-threshold",
                        str(float(cfg.boundary_threshold)),
                        "--boundary-min-len",
                        str(int(cfg.boundary_min_len)),
                        "--max-length",
                        str(int(cfg.max_length)),
                        "--seed",
                        str(int(seed)),
                    ]

                if not cfg.no_trace:
                    pa_cmd += ["--trace-stages-jsonl", _to_run_path(pa_trace_path)]
                if cfg.enable_refine:
                    pa_cmd.append("--enable-refine")
                if cfg.enable_src_marker_bonus:
                    pa_cmd.append("--enable-src-marker-boundary-bonus")
                if cfg.enable_src_marker_whitespace_dp_bonus:
                    pa_cmd.append("--enable-src-marker-whitespace-dp-bonus")
                if cfg.deterministic:
                    pa_cmd.append("--deterministic")
                pa_cmd += ["--device", str(cfg.device)]

                if cfg.stream:
                    rc_pa, pa_text = _run_stream(
                        pa_cmd,
                        cwd=WORKSPACE_ROOT,
                        env=env,
                        allow_failure=True,
                        echo=True,
                        log_path=pa_log_path,
                    )
                else:
                    rc_pa, pa_text = _run(
                        pa_cmd,
                        cwd=WORKSPACE_ROOT,
                        env=env,
                        allow_failure=True,
                        capture=True,
                        echo=False,
                        log_path=pa_log_path,
                    )
                if rc_pa != 0:
                    w.writerow(
                        {
                            "seed": seed,
                            "pd_subset": str(pd_subset_path.relative_to(WORKSPACE_ROOT)) if pd_subset_path.exists() else None,
                            "gold_subset": str(gold_subset_path.relative_to(WORKSPACE_ROOT)) if gold_subset_path.exists() else None,
                            "pa_output": str(pa_output_path.relative_to(WORKSPACE_ROOT)),
                            "tgt_exact_ok": None,
                            "tgt_exact_total": None,
                            "tgt_exact_nfc_ok": None,
                            "tgt_exact_nfc_total": None,
                            "tgt_exact_sent_ok": None,
                            "tgt_exact_sent_total": None,
                            "tgt_exact_sent_nfc_ok": None,
                            "tgt_exact_sent_nfc_total": None,
                            "src_sim_mean_tgt_sent_ok": None,
                            "micro_f1_tgt_exact": None,
                            "micro_f1_tgt_exact_nfc": None,
                            "returncode": rc_pa,
                        }
                    )
                    tail = "\n".join((pa_text or "").splitlines()[-30:])
                    print(f"\n→ seed={seed} PA failed (returncode={rc_pa}), skipping eval")
                    if tail.strip():
                        print("--- PA log tail ---")
                        print(tail)
                        print("-------------------")
                    continue
            else:
                # eval-only: 필요한 파일이 없으면 스킵
                if not pa_output_path.exists():
                    w.writerow(
                        {
                            "seed": seed,
                            "pd_subset": str(pd_subset_path.relative_to(WORKSPACE_ROOT)) if pd_subset_path.exists() else None,
                            "gold_subset": str(gold_subset_path.relative_to(WORKSPACE_ROOT)) if gold_subset_path.exists() else None,
                            "pa_output": str(pa_output_path.relative_to(WORKSPACE_ROOT)),
                            "tgt_exact_ok": None,
                            "tgt_exact_total": None,
                            "tgt_exact_nfc_ok": None,
                            "tgt_exact_nfc_total": None,
                            "tgt_exact_sent_ok": None,
                            "tgt_exact_sent_total": None,
                            "tgt_exact_sent_nfc_ok": None,
                            "tgt_exact_sent_nfc_total": None,
                            "src_sim_mean_tgt_sent_ok": None,
                            "micro_f1_tgt_exact": None,
                            "micro_f1_tgt_exact_nfc": None,
                            "returncode": 1,
                        }
                    )
                    print(f"\n→ seed={seed} eval-only: pa_output missing, skipping")
                    continue
                if not gold_subset_path.exists():
                    w.writerow(
                        {
                            "seed": seed,
                            "pd_subset": str(pd_subset_path.relative_to(WORKSPACE_ROOT)) if pd_subset_path.exists() else None,
                            "gold_subset": str(gold_subset_path.relative_to(WORKSPACE_ROOT)),
                            "pa_output": str(pa_output_path.relative_to(WORKSPACE_ROOT)),
                            "tgt_exact_ok": None,
                            "tgt_exact_total": None,
                            "tgt_exact_nfc_ok": None,
                            "tgt_exact_nfc_total": None,
                            "tgt_exact_sent_ok": None,
                            "tgt_exact_sent_total": None,
                            "tgt_exact_sent_nfc_ok": None,
                            "tgt_exact_sent_nfc_total": None,
                            "src_sim_mean_tgt_sent_ok": None,
                            "micro_f1_tgt_exact": None,
                            "micro_f1_tgt_exact_nfc": None,
                            "returncode": 1,
                        }
                    )
                    print(f"\n→ seed={seed} eval-only: gold_subset missing, skipping")
                    continue

            # 2) 평가
            if _is_running_in_docker():
                eval_cmd = [
                    "python",
                    "integrity_report.py",
                    "--input",
                    _to_run_path(pa_output_path),
                    "--gold",
                    _to_run_path(gold_subset_path),
                ]
            else:
                eval_cmd = [
                    "docker",
                    "compose",
                    "run",
                    "--rm",
                    "csp",
                    "python",
                    "integrity_report.py",
                    "--input",
                    _to_run_path(pa_output_path),
                    "--gold",
                    _to_run_path(gold_subset_path),
                ]
            rc_eval, text = _run(eval_cmd, cwd=WORKSPACE_ROOT, env=env, allow_failure=True, capture=True)
            (
                f1,
                ok,
                tot,
                ok_s,
                tot_s,
                sim_mean,
                f1_nfc,
                ok_nfc,
                tot_nfc,
                ok_s_nfc,
                tot_s_nfc,
            ) = _parse_eval(text)

            w.writerow(
                {
                    "seed": seed,
                    "pd_subset": str(pd_subset_path.relative_to(WORKSPACE_ROOT)),
                    "gold_subset": str(gold_subset_path.relative_to(WORKSPACE_ROOT)),
                    "pa_output": str(pa_output_path.relative_to(WORKSPACE_ROOT)),
                    "tgt_exact_ok": ok,
                    "tgt_exact_total": tot,
                    "tgt_exact_nfc_ok": ok_nfc,
                    "tgt_exact_nfc_total": tot_nfc,
                    "tgt_exact_sent_ok": ok_s,
                    "tgt_exact_sent_total": tot_s,
                    "tgt_exact_sent_nfc_ok": ok_s_nfc,
                    "tgt_exact_sent_nfc_total": tot_s_nfc,
                    "src_sim_mean_tgt_sent_ok": sim_mean,
                    "micro_f1_tgt_exact": f1,
                    "micro_f1_tgt_exact_nfc": f1_nfc,
                    "returncode": rc_eval,
                }
            )
            extra = ""
            if ok_s is not None and tot_s is not None:
                extra += f", tgt_exact_sent={ok_s}/{tot_s}"
            if sim_mean is not None:
                extra += f", src_sim_mean={sim_mean:.4f}"
            print(f"\n→ seed={seed} micro_f1_tgt_exact={f1} (tgt_exact={ok}/{tot}{extra})")

    print("\n✅ multitest summary saved:")
    print(summary_path)

    if cfg.analyze_mismatch:
        try:
            # mismatch cases (전체 케이스 + 대표 예시)
            _run(
                [
                    sys.executable,
                    "scripts/collect_tgt_mismatch_cases.py",
                    "--run-dir",
                    str(run_dir.relative_to(WORKSPACE_ROOT)),
                    "--seeds",
                    *[str(int(x)) for x in cfg.seeds],
                    "--out-md",
                    str((run_dir / "tgt_mismatch_seed1_10.md").relative_to(WORKSPACE_ROOT)),
                    "--out-csv",
                    str((run_dir / "tgt_mismatch_seed1_10.csv").relative_to(WORKSPACE_ROOT)),
                ],
                cwd=WORKSPACE_ROOT,
                env=os.environ.copy(),
                allow_failure=True,
                capture=True,
                echo=True,
            )

            # mismatch subtypes (경계 주변 컨텍스트 라벨)
            _run(
                [
                    sys.executable,
                    "scripts/analyze_tgt_mismatch_subtypes.py",
                    "--run-dir",
                    str(run_dir.relative_to(WORKSPACE_ROOT)),
                    "--seeds",
                    *[str(int(x)) for x in cfg.seeds],
                    "--out-md",
                    str((run_dir / "tgt_mismatch_subtypes_seed1_10.md").relative_to(WORKSPACE_ROOT)),
                ],
                cwd=WORKSPACE_ROOT,
                env=os.environ.copy(),
                allow_failure=True,
                capture=True,
                echo=True,
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
