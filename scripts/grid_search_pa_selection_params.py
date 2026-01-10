#!/usr/bin/env python3
"""PA selection params Grid Search 러너

왜 새 스크립트인가?
- 기존 scripts/grid_search_pa_weights.py 가 바꾸던 pa.sentence_splitter.{prior_bonus,length_penalty_coef,supar_bonus}는
  현재 PA 선택/정렬 점수 로직(특히 pa/processor.py의 후보 선택 score)에 직접 연결되어 있지 않습니다.
- 반면 pa_selection_params는 후보 선택 점수에 직접 들어갑니다:
  - candidate_prior_bonus_by_prefix ("supar(", "boundary(")
  - boundary_style_prior weights
  - boundary_aware_weight

이 스크립트는 위 '실제 레버'를 grid로 튜닝합니다.

출력 구조는 summarize_grid_search.py가 읽을 수 있도록 root summary.json에 results=[...] 형태로 저장합니다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import hashlib
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_float_values(spec: str) -> List[float]:
    """float 리스트 파서.

    지원 형식:
    - 콤마 리스트: "0,0.01,0.02"
    - 구간 스텝:  "start:end:step" (end 포함; 부동소수 오차는 소수 12자리 반올림)
      예) "0:0.3:0.05" -> 0.00, 0.05, ..., 0.30
    """
    s = (spec or "").strip()
    if not s:
        return []
    if ":" in s and "," not in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid range spec (expected start:end:step): {spec}")
        start, end, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise ValueError(f"step must be > 0: {spec}")
        vals: List[float] = []
        cur = start
        # end 포함 (부동소수 오차 고려)
        while cur <= end + (step / 1_000_000):
            vals.append(round(cur, 12))
            cur += step
        return vals
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_path(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur: Dict[str, Any] = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def _run_pa(
    seed: int,
    boundary_threshold: float,
    input_xlsx: Path,
    output_xlsx: Path,
) -> tuple[bool, str]:
    project_root = Path.cwd()
    rel_input = input_xlsx.relative_to(project_root)
    rel_output = output_xlsx.relative_to(project_root)

    docker_input = f"/workspace/{rel_input.as_posix()}"
    docker_output = f"/workspace/{rel_output.as_posix()}"

    # NOTE:
    # - `docker-compose exec`는 csp 서비스 컨테이너가 "running"이어야만 동작합니다.
    # - 대규모/반복 실행 중 컨테이너가 재시작/종료되면 `service "csp" is not running`으로
    #   전체 실험이 연쇄 실패할 수 있습니다.
    # - 여기서는 1회 실행용 컨테이너를 띄우는 `docker compose run --rm --no-deps`로
    #   서비스 상태 의존을 제거합니다.
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "csp",
        "python",
        "pa/main.py",
        docker_input,
        docker_output,
        "--embedder",
        "bge",
        "--use-boundary-model",
        "--boundary-threshold",
        str(boundary_threshold),
        "--enable-src-marker-boundary-bonus",
        "--seed",
        str(seed),
    ]

    result = subprocess.run(cmd, capture_output=True, cwd=project_root)
    stdout = result.stdout.decode("utf-8", errors="ignore") if isinstance(result.stdout, (bytes, bytearray)) else str(result.stdout)
    stderr = result.stderr.decode("utf-8", errors="ignore") if isinstance(result.stderr, (bytes, bytearray)) else str(result.stderr)

    if result.returncode != 0:
        return False, (stderr or stdout)

    if not output_xlsx.exists():
        return False, "PA output file not created"

    return True, ""


def _check_docker_available() -> tuple[bool, str]:
    """Docker 엔진/daemon이 살아있는지 빠르게 확인.

    Windows에서 Docker Desktop이 꺼져 있거나 Linux 엔진 파이프가 없으면
    `open //./pipe/dockerDesktopLinuxEngine` 류 에러가 나며 모든 run이 즉시 실패한다.
    하루짜리 테스트를 막기 위해, 한 번이라도 이런 상태면 즉시 중단한다.
    """
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if r.returncode == 0:
            return True, ""
        msg = (r.stderr or r.stdout or "").strip()
        return False, msg
    except Exception as e:
        return False, str(e)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _evaluate_paragraph_based(pred_xlsx: Path, gt_csv: Path, sample_keys_file: Path) -> tuple[float, float]:
    """
    번역문 exact match한 문장들만 대상으로, 원문 경계의 boundary-based F1(조화평균) 산출.
    이것은 integrity_report.py의 'tgt 완전일치 subset' F1과 동일한 방식.

    Returns:
        (src_f1_on_tgt_exact, mean_src_similarity_on_tgt_exact)
        - src_f1_on_tgt_exact: 번역문 일치 문장들의 원문 경계 F1 (조화평균)
        - mean_src_similarity_on_tgt_exact: 번역문 일치 문장에서 원문 유사도 평균
    """
    import pandas as pd
    from difflib import SequenceMatcher

    def _norm(x: object) -> str:
        return str(x).strip() if x is not None else ""

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _boundary_positions_normed(segments: list[str]) -> set[int]:
        """정규화 문자열 기준 문장 경계 위치(누적 길이) 집합."""
        positions: set[int] = set()
        cursor = 0
        for i, seg in enumerate(segments):
            seg_norm = _norm(seg)
            cursor += len(seg_norm)
            if i < len(segments) - 1:
                positions.add(cursor)
        return positions

    def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        """Precision, Recall, F1 조화평균 계산."""
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0, 1.0, 1.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return p, r, f1

    pred_df = pd.read_excel(pred_xlsx)
    gt_df_full = pd.read_csv(gt_csv)

    if sample_keys_file.exists():
        sampled_keys = json.loads(sample_keys_file.read_text(encoding="utf-8"))
        sampled_keys_df = pd.DataFrame(sampled_keys, columns=["book_name", "문단식별자"])
        gt_df = gt_df_full.merge(sampled_keys_df, on=["book_name", "문단식별자"], how="inner").reset_index(drop=True)
    else:
        gt_df = gt_df_full

    use_book = ("book_name" in pred_df.columns) and ("book_name" in gt_df.columns)
    group_cols = ["book_name", "문단식별자"] if use_book else ["문단식별자"]

    pred_groups = pred_df.groupby(group_cols, sort=False)
    gt_groups = gt_df.groupby(group_cols, sort=False)

    common_keys = sorted(set(pred_groups.groups.keys()) & set(gt_groups.groups.keys()))

    # 번역문 일치 문장 subset에서의 원문 경계 TP/FP/FN
    tp_ok = 0
    fp_ok = 0
    fn_ok = 0
    src_sim_on_tgt_exact: List[float] = []

    for key in common_keys:
        pred_g = pred_groups.get_group(key)
        gt_g = gt_groups.get_group(key)

        if "문장식별자" in pred_g.columns:
            pred_g = pred_g.sort_values("문장식별자", kind="mergesort")
        if "문장식별자" in gt_g.columns:
            gt_g = gt_g.sort_values("문장식별자", kind="mergesort")

        pred_tgt = [_norm(x) for x in pred_g.get("번역문", [])]
        gt_tgt = [_norm(x) for x in gt_g.get("번역문", [])]
        pred_src = [_norm(x) for x in pred_g.get("원문", [])]
        gt_src = [_norm(x) for x in gt_g.get("원문", [])]

        # 번역문 리스트 완전일치 여부
        tgt_match = (pred_tgt == gt_tgt)

        # 문장 단위 번역문 일치 체크 (유사도 계산용)
        for ps, pt, gs, gt in zip(pred_src, pred_tgt, gt_src, gt_tgt):
            if pt == gt:  # 번역문 문장이 일치하는 경우만
                src_sim_on_tgt_exact.append(_sim(ps, gs))

        # 번역문 리스트가 일치하는 경우에만 원문 경계 F1 집계
        if tgt_match:
            pred_b = _boundary_positions_normed(pred_src)
            gold_b = _boundary_positions_normed(gt_src)
            inter = pred_b & gold_b

            tp_i = len(inter)
            fp_i = len(pred_b - gold_b)
            fn_i = len(gold_b - pred_b)

            tp_ok += tp_i
            fp_ok += fp_i
            fn_ok += fn_i

    # F1: 번역문 일치 문장들의 원문 경계 F1 (조화평균)
    _, _, src_f1_on_tgt_exact = _prf1(tp_ok, fp_ok, fn_ok)
    mean_similarity = (sum(src_sim_on_tgt_exact) / len(src_sim_on_tgt_exact)) if src_sim_on_tgt_exact else 0.0

    return src_f1_on_tgt_exact, mean_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="PA selection params Grid Search")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--base-config", default="csp_config.json", type=str)
    parser.add_argument("--sample-size", default=1000, type=int)
    parser.add_argument("--seeds", default="1,2,3", type=str)

    # 실제 레버들
    # boundary-threshold: 그리드로 여러 값 시도 가능 (콤마/구간)
    parser.add_argument(
        "--boundary-threshold",
        default="0.70",
        type=str,
        help="pa boundary model threshold 후보 (예: 0.5,0.7,0.9 또는 0.3:0.9:0.1)",
    )

    # 후보 세트 선택적 제외(강한 레버)
    parser.add_argument("--disable-supar", action="store_true", help="supar 후보 제외")
    parser.add_argument("--disable-boundary", action="store_true", help="boundary 후보 제외")
    parser.add_argument("--disable-whitespace-dp", action="store_true", help="whitespace_dp 후보 제외")

    # A) prior 범위 확장(콤마 리스트 또는 start:end:step)
    parser.add_argument(
        "--prior-boundary",
        default="0.00,0.01,0.02",
        type=str,
        help="candidate_prior_bonus_by_prefix['boundary('] 후보 (예: 0,0.01,0.02 또는 0:0.3:0.05)",
    )
    parser.add_argument(
        "--prior-supar",
        default="0.00,0.015,0.03",
        type=str,
        help="candidate_prior_bonus_by_prefix['supar('] 후보 (예: 0,0.015,0.03 또는 0:0.3:0.05)",
    )

    # B) 추가 레버들(필요할 때만 지정; 미지정 시 base-config 값을 그대로 사용)
    parser.add_argument(
        "--boundary-aware-weight",
        default="",
        type=str,
        help="pa_selection_params.boundary_aware_weight 후보 (예: 0.1,0.3,0.6 또는 0.1:1.0:0.1)",
    )
    parser.add_argument(
        "--style-weight-terminal",
        default="",
        type=str,
        help="pa_selection_params.boundary_style_prior.weight_terminal 후보",
    )
    parser.add_argument(
        "--style-weight-continuation",
        default="",
        type=str,
        help="pa_selection_params.boundary_style_prior.weight_continuation 후보(보통 음수)",
    )
    parser.add_argument(
        "--max-candidates-multiplier",
        default="",
        type=str,
        help="pa_selection_params.max_candidates_multiplier 후보(예: 8,12,16)",
    )
    parser.add_argument(
        "--penalty-empty-src",
        default="",
        type=str,
        help="pa_selection_params.penalty_empty_src 후보(예: 0.3,0.5,0.8)",
    )
    parser.add_argument(
        "--penalty-short-per-pair",
        default="",
        type=str,
        help="pa_selection_params.penalty_short_pairs.penalty_per_pair 후보(예: 0.01,0.015,0.02)",
    )

    # 하루짜리 전체 실행을 피하기 위한 기본 staged 실행
    staged_group = parser.add_mutually_exclusive_group()
    staged_group.add_argument("--staged", action="store_true", default=True, help="(기본) stage1 소샘플로 top-k를 고른 뒤 stage2만 실행")
    staged_group.add_argument("--no-staged", dest="staged", action="store_false", help="staged 실행 비활성화(모든 config를 sample-size로 실행)")
    parser.add_argument("--stage1-size", default=50, type=int, help="staged 실행 시 stage1 샘플 문단 수")
    parser.add_argument("--top-k", default=4, type=int, help="staged 실행 시 stage2로 올릴 상위 config 개수")
    parser.add_argument("--stage1-seeds", default="", type=str, help="stage1에만 사용할 seed 목록(기본: seeds의 첫 seed 1개)")
    parser.add_argument("--force", action="store_true", help="프리플라이트(노브 영향 검증) 실패해도 강제 진행")

    parser.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "초소형 프리셋(런타임 단축): 기본값을 stage1-size=5, top-k=1, sample-size=30으로 축소하고 "
            "--stage1-seeds 를 명시하지 않았으면 1로 설정합니다. 명시적으로 준 값(--sample-size 등)은 덮어쓰지 않습니다."
        ),
    )

    parser.add_argument("--yes", action="store_true", help="확인 없이 실행")
    args = parser.parse_args()

    # NOTE: argparse만으로는 "기본값을 쓴 것"과 "사용자가 기본값과 동일한 값을 명시"한 것을
    # 구분하기 어렵습니다. 런타임 단축을 위해, 사용자가 --stage1-seeds 를 명시했는데
    # --seeds 를 명시하지 않은 경우에는 stage2도 stage1 seed만 사용하도록 해석합니다.
    seeds_was_explicit = "--seeds" in sys.argv
    sample_size_was_explicit = "--sample-size" in sys.argv
    stage1_size_was_explicit = "--stage1-size" in sys.argv
    top_k_was_explicit = "--top-k" in sys.argv
    stage1_seeds_was_explicit = "--stage1-seeds" in sys.argv

    if args.tiny:
        if not sample_size_was_explicit:
            args.sample_size = 30
        if not stage1_size_was_explicit:
            args.stage1_size = 5
        if not top_k_was_explicit:
            args.top_k = 1
        if not stage1_seeds_was_explicit:
            # stage1_seeds 파싱 로직이 args.stage1_seeds를 보므로 문자열로 주입
            args.stage1_seeds = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds 가 비어있습니다")

    if args.stage1_seeds.strip():
        stage1_seeds = [int(s.strip()) for s in args.stage1_seeds.split(",") if s.strip()]
    else:
        stage1_seeds = [seeds[0]]

    # stage1-seeds를 명시했는데 stage2 seeds(--seeds)를 안 줬다면, stage2도 stage1 seed로 축소
    if (not seeds_was_explicit) and args.stage1_seeds.strip():
        seeds = list(stage1_seeds)

    boundary_threshold_vals = _parse_float_values(args.boundary_threshold)
    if not boundary_threshold_vals:
        boundary_threshold_vals = [0.70]

    prior_boundary_vals = _parse_float_values(args.prior_boundary)
    prior_supar_vals = _parse_float_values(args.prior_supar)

    boundary_aware_vals = _parse_float_values(args.boundary_aware_weight)
    style_w_term_vals = _parse_float_values(args.style_weight_terminal)
    style_w_cont_vals = _parse_float_values(args.style_weight_continuation)
    max_cand_mult_vals = [int(round(x)) for x in _parse_float_values(args.max_candidates_multiplier)]
    penalty_empty_src_vals = _parse_float_values(args.penalty_empty_src)
    penalty_short_per_pair_vals = _parse_float_values(args.penalty_short_per_pair)

    # 지정하지 않은 축은 base-config 값을 그대로 쓰므로, 그리드 축에서는 'None(=no patch)'로 둔다.
    boundary_aware_axis: List[Optional[float]] = boundary_aware_vals if boundary_aware_vals else [None]
    style_w_term_axis: List[Optional[float]] = style_w_term_vals if style_w_term_vals else [None]
    style_w_cont_axis: List[Optional[float]] = style_w_cont_vals if style_w_cont_vals else [None]
    max_cand_mult_axis: List[Optional[int]] = max_cand_mult_vals if max_cand_mult_vals else [None]
    penalty_empty_src_axis: List[Optional[float]] = penalty_empty_src_vals if penalty_empty_src_vals else [None]
    penalty_short_per_pair_axis: List[Optional[float]] = penalty_short_per_pair_vals if penalty_short_per_pair_vals else [None]

    # 후보 세트 제외 플래그 (CLI에서 한 번 고정; 그리드가 아니라 '모드')
    disable_supar = bool(args.disable_supar)
    disable_boundary = bool(args.disable_boundary)
    disable_whitespace_dp = bool(args.disable_whitespace_dp)

    configs: List[Dict[str, Any]] = []
    for bt, pb, ps, b_w, w_t, w_c, m_mult, p_empty, p_short in itertools.product(
        boundary_threshold_vals,
        prior_boundary_vals,
        prior_supar_vals,
        boundary_aware_axis,
        style_w_term_axis,
        style_w_cont_axis,
        max_cand_mult_axis,
        penalty_empty_src_axis,
        penalty_short_per_pair_axis,
    ):
        configs.append(
            {
                "boundary_threshold": float(bt),
                "prior_boundary": float(pb),
                "prior_supar": float(ps),
                "boundary_aware_weight": b_w,
                "style_weight_terminal": w_t,
                "style_weight_continuation": w_c,
                "max_candidates_multiplier": m_mult,
                "penalty_empty_src": p_empty,
                "penalty_short_per_pair": p_short,
                "disable_supar": disable_supar,
                "disable_boundary": disable_boundary,
                "disable_whitespace_dp": disable_whitespace_dp,
            }
        )

    stage1_size = int(max(1, min(args.stage1_size, args.sample_size)))
    top_k = int(max(1, min(args.top_k, len(configs))))

    if args.staged:
        # configs가 top_k 이하이면 stage1에서 걸러낼 게 없으므로 stage1을 생략(실험량/시간 절약)
        if len(configs) <= top_k:
            total_experiments = top_k * len(seeds)
        else:
            total_experiments = (len(configs) * len(stage1_seeds)) + (top_k * len(seeds))
    else:
        total_experiments = len(configs) * len(seeds)
    if not args.yes:
        print("=" * 80)
        print("PA selection params Grid Search")
        print("=" * 80)
        print(f"configs: {len(configs)}  seeds: {len(seeds)}  total runs: {total_experiments}")
        print("주의: 이 스크립트는 pa_selection_params를 실제로 수정합니다 (각 run 후 복구).")
        print("계속하려면 --yes 를 붙여주세요.")
        return

    # Docker 사전 점검: 엔진이 안 떠있으면 전체 실험은 의미가 없으니 즉시 중단
    ok_docker, docker_msg = _check_docker_available()
    if not ok_docker:
        raise SystemExit(
            "[ABORT] Docker 엔진에 연결할 수 없습니다. "
            "Docker Desktop 실행/엔진 전환(Linux containers) 상태를 확인하세요.\n" + docker_msg
        )

    start = time.time()

    def _run_grid(selected_configs: List[Dict[str, Any]], run_seeds: List[int], effective_sample_size: int, run_output_dir: Path) -> tuple[List[Dict[str, Any]], int, int]:
        import pandas as pd

        run_output_dir.mkdir(parents=True, exist_ok=True)
        grid_results: List[Dict[str, Any]] = []
        grid_success = 0
        grid_failed = 0

        for cfg in selected_configs:
            bt = cfg.get("boundary_threshold", 0.70)
            parts = [f"bt{float(bt):.2f}", f"pB{cfg['prior_boundary']:.3f}", f"pS{cfg['prior_supar']:.3f}"]
            if cfg.get("boundary_aware_weight") is not None:
                parts.append(f"bW{float(cfg['boundary_aware_weight']):.3f}")
            if cfg.get("style_weight_terminal") is not None:
                parts.append(f"sT{float(cfg['style_weight_terminal']):.3f}")
            if cfg.get("style_weight_continuation") is not None:
                parts.append(f"sC{float(cfg['style_weight_continuation']):.3f}")
            if cfg.get("max_candidates_multiplier") is not None:
                parts.append(f"mC{int(cfg['max_candidates_multiplier'])}")
            if cfg.get("penalty_empty_src") is not None:
                parts.append(f"pE{float(cfg['penalty_empty_src']):.3f}")
            if cfg.get("penalty_short_per_pair") is not None:
                parts.append(f"pSP{float(cfg['penalty_short_per_pair']):.3f}")
            if cfg.get("disable_supar"):
                parts.append("noSup")
            if cfg.get("disable_boundary"):
                parts.append("noBnd")
            if cfg.get("disable_whitespace_dp"):
                parts.append("noWsDP")
            config_name = "_".join(parts)
            config_entry: Dict[str, Any] = {
                "config": {
                    "prior_bonus": cfg["prior_boundary"],
                    "length_penalty": cfg["prior_supar"],
                    "boundary_threshold": float(bt),
                    "supar_bonus": 0.0,
                    **(
                        {"boundary_aware_weight": float(cfg["boundary_aware_weight"])}
                        if cfg.get("boundary_aware_weight") is not None
                        else {}
                    ),
                    "disable_supar": bool(cfg.get("disable_supar")),
                    "disable_boundary": bool(cfg.get("disable_boundary")),
                    "disable_whitespace_dp": bool(cfg.get("disable_whitespace_dp")),
                    "_tuned": {
                        "pa_selection_params": {
                            "candidate_prior_bonus_by_prefix": {
                                "boundary(": cfg["prior_boundary"],
                                "supar(": cfg["prior_supar"],
                            },
                            **(
                                {"boundary_aware_weight": float(cfg["boundary_aware_weight"])}
                                if cfg.get("boundary_aware_weight") is not None
                                else {}
                            ),
                            **(
                                {
                                    "boundary_style_prior": {
                                        "weight_terminal": float(cfg["style_weight_terminal"])
                                        if cfg.get("style_weight_terminal") is not None
                                        else None,
                                        "weight_continuation": float(cfg["style_weight_continuation"])
                                        if cfg.get("style_weight_continuation") is not None
                                        else None,
                                    }
                                }
                                if (cfg.get("style_weight_terminal") is not None or cfg.get("style_weight_continuation") is not None)
                                else {}
                            ),
                            **(
                                {"max_candidates_multiplier": int(cfg["max_candidates_multiplier"])}
                                if cfg.get("max_candidates_multiplier") is not None
                                else {}
                            ),
                            **(
                                {"penalty_empty_src": float(cfg["penalty_empty_src"])}
                                if cfg.get("penalty_empty_src") is not None
                                else {}
                            ),
                            **(
                                {"penalty_short_pairs": {"penalty_per_pair": float(cfg["penalty_short_per_pair"])}}
                                if cfg.get("penalty_short_per_pair") is not None
                                else {}
                            ),
                        },
                    },
                },
                "seed_results": [],
            }

            for seed in run_seeds:
                run_dir = run_output_dir / config_name / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dir_abs = run_dir.resolve()

                backup_path = base_config_path.with_suffix(".backup.json")
                shutil.copy(base_config_path, backup_path)

                try:
                    # 1) config patch (실제 레버)
                    cfg_json = _load_json(base_config_path)
                    pa_sel = _ensure_path(cfg_json, ["pa_selection_params"])
                    priors = _ensure_path(pa_sel, ["candidate_prior_bonus_by_prefix"])
                    priors["boundary("] = float(cfg["prior_boundary"])
                    priors["supar("] = float(cfg["prior_supar"])

                    if cfg.get("boundary_aware_weight") is not None:
                        pa_sel["boundary_aware_weight"] = float(cfg["boundary_aware_weight"])

                    if cfg.get("style_weight_terminal") is not None or cfg.get("style_weight_continuation") is not None:
                        style = _ensure_path(pa_sel, ["boundary_style_prior"])
                        # enabled는 명시된 경우에만 True로 켠다
                        style["enabled"] = True
                        if cfg.get("style_weight_terminal") is not None:
                            style["weight_terminal"] = float(cfg["style_weight_terminal"])
                        if cfg.get("style_weight_continuation") is not None:
                            style["weight_continuation"] = float(cfg["style_weight_continuation"])

                    if cfg.get("max_candidates_multiplier") is not None:
                        pa_sel["max_candidates_multiplier"] = int(cfg["max_candidates_multiplier"])

                    if cfg.get("penalty_empty_src") is not None:
                        pa_sel["penalty_empty_src"] = float(cfg["penalty_empty_src"])

                    if cfg.get("penalty_short_per_pair") is not None:
                        psp = _ensure_path(pa_sel, ["penalty_short_pairs"])
                        psp["penalty_per_pair"] = float(cfg["penalty_short_per_pair"])

                    # boundary threshold도 명시(옵션)
                    pa_cfg = _ensure_path(cfg_json, ["pa"])
                    pa_cfg["boundary_threshold"] = float(bt)

                    # 후보 세트 제외 플래그 저장 (PA processor가 읽음)
                    pa_sel["disable_supar"] = bool(cfg.get("disable_supar"))
                    pa_sel["disable_boundary"] = bool(cfg.get("disable_boundary"))
                    pa_sel["disable_whitespace_dp"] = bool(cfg.get("disable_whitespace_dp"))

                    _save_json(base_config_path, cfg_json)

                    # 2) sample keys (stage2 sample을 먼저 뽑고 stage1은 prefix로 사용 → nesting)
                    sample_keys_file = run_dir_abs / f"sample_keys_seed{seed}.json"
                    if effective_sample_size and effective_sample_size < len(key_df):
                        import random

                        rnd = random.Random(seed)
                        sampled_idx = rnd.sample(range(len(key_df)), min(effective_sample_size, len(key_df)))
                        sampled_keys_df = key_df.iloc[sampled_idx].reset_index(drop=True)
                    else:
                        sampled_keys_df = key_df

                    sampled_keys = sampled_keys_df[["book_name", "문단식별자"]].values.tolist()
                    sample_keys_file.write_text(json.dumps(sampled_keys, ensure_ascii=False), encoding="utf-8")

                    # 3) input xlsx (PD)
                    pd_df_full = pd.read_csv(pd_csv)
                    pd_df_sample = pd_df_full.merge(sampled_keys_df, on=["book_name", "문단식별자"], how="inner").reset_index(drop=True)

                    input_xlsx = run_dir_abs / f"pa_test_input_seed{seed}.xlsx"
                    output_xlsx = run_dir_abs / f"pa_test_output_seed{seed}.xlsx"
                    pd_df_sample.to_excel(input_xlsx, index=False)

                    # 4) run PA (config별 boundary_threshold 사용)
                    ok, err = _run_pa(seed=seed, boundary_threshold=float(bt), input_xlsx=input_xlsx, output_xlsx=output_xlsx)
                    if not ok:
                        grid_failed += 1
                        config_entry["seed_results"].append(
                            {
                                "seed": seed,
                                "micro_f1_tgt_exact": 0.0,
                                "mean_similarity": 0.0,
                                "success": False,
                                "error": err[:4000],
                            }
                        )
                        continue

                    # 5) evaluate
                    f1, sim = _evaluate_paragraph_based(output_xlsx, gt_csv, sample_keys_file)
                    grid_success += 1
                    config_entry["seed_results"].append(
                        {
                            "seed": seed,
                            "micro_f1_tgt_exact": float(f1),
                            "mean_similarity": float(sim),
                            "success": True,
                        }
                    )

                finally:
                    # restore config
                    try:
                        if backup_path.exists():
                            shutil.move(str(backup_path), str(base_config_path))
                    except Exception:
                        pass

            grid_results.append(config_entry)

        return grid_results, grid_success, grid_failed

    # 공통 데이터 경로
    gt_csv = Path("datasets/pa/test.csv")
    pd_csv = Path("datasets/pd/test.csv")

    if not gt_csv.exists() or not pd_csv.exists():
        raise FileNotFoundError("datasets/pa/test.csv 또는 datasets/pd/test.csv 를 찾을 수 없습니다")

    import pandas as pd
    test_df = pd.read_csv(gt_csv)
    key_df = test_df[["book_name", "문단식별자"]].drop_duplicates().reset_index(drop=True)

    # 프리플라이트: 첫/끝 config를 stage1 크기로 1 seed만 돌려 결과 파일 해시가 같으면 즉시 중단
    if len(configs) >= 2 and stage1_seeds:
        pf_dir = output_dir / "_preflight"
        pf_results, pf_ok, pf_fail = _run_grid([configs[0], configs[-1]], [stage1_seeds[0]], stage1_size, pf_dir)
        try:
            out1 = (pf_dir / f"pB{configs[0]['prior_boundary']:.3f}_pS{configs[0]['prior_supar']:.3f}" / f"seed{stage1_seeds[0]}" / f"pa_test_output_seed{stage1_seeds[0]}.xlsx")
            out2 = (pf_dir / f"pB{configs[-1]['prior_boundary']:.3f}_pS{configs[-1]['prior_supar']:.3f}" / f"seed{stage1_seeds[0]}" / f"pa_test_output_seed{stage1_seeds[0]}.xlsx")
            if out1.exists() and out2.exists():
                h1 = _sha256_file(out1)
                h2 = _sha256_file(out2)
                if h1 == h2 and not args.force:
                    raise SystemExit(
                        "[ABORT] 프리플라이트: 서로 다른 설정 2개가 동일 출력(해시 동일)입니다. "
                        "현재 튜닝 노브가 결과에 영향을 못 주는 상태이거나, 샘플이 너무 작아 구분이 안 됩니다. "
                        "강제로 진행하려면 --force 를 붙이세요."
                    )
        except Exception as e:
            # 프리플라이트 자체가 실패하면(도커/런타임 문제) 일단 그대로 진행(실패는 seed_results에 기록됨)
            if args.force:
                pass
            else:
                raise

    stage1_meta: Dict[str, Any] | None = None
    stage2_meta: Dict[str, Any] | None = None

    if args.staged and len(configs) <= top_k:
        # stage1을 생략: configs를 그대로 stage2로 실행
        results, success, failed = _run_grid(configs, seeds, int(args.sample_size), output_dir)
        stage2_meta = {
            "dir": str(output_dir),
            "success": int(success),
            "failed": int(failed),
            "total_experiments": int(len(configs) * len(seeds)),
            "sample_size": int(args.sample_size),
            "seeds": list(seeds),
            "selected_configs": configs,
            "stage1_skipped": True,
        }
    elif args.staged:
        # stage1: 모든 config를 작은 샘플로 빠르게 스코어링
        stage1_dir = output_dir / "_stage1"
        stage1_results, s1_ok, s1_fail = _run_grid(configs, stage1_seeds, stage1_size, stage1_dir)

        # stage1 요약 저장
        stage1_summary = {
            "timestamp": datetime.now().isoformat(),
            "stage": "stage1",
            "stage1_size": stage1_size,
            "seeds": stage1_seeds,
            "total_configs": len(configs),
            "total_experiments": len(configs) * len(stage1_seeds),
            "success": s1_ok,
            "failed": s1_fail,
            "results": stage1_results,
        }
        _save_json(stage1_dir / "summary.json", stage1_summary)
        stage1_meta = {
            "dir": str(stage1_dir),
            "success": int(s1_ok),
            "failed": int(s1_fail),
            "total_experiments": int(len(configs) * len(stage1_seeds)),
            "stage1_size": int(stage1_size),
            "seeds": list(stage1_seeds),
        }

        # top-k 선정(성공 seed만 평균)
        def _mean_f1(entry: Dict[str, Any]) -> float:
            vals = [r.get("micro_f1_tgt_exact", 0.0) for r in entry.get("seed_results", []) if bool(r.get("success", True))]
            return sum(vals) / len(vals) if vals else -1.0

        ranked = sorted(stage1_results, key=_mean_f1, reverse=True)
        selected_names = set()
        selected_cfgs: List[Dict[str, Any]] = []
        for r in ranked:
            tuned = (((r.get("config") or {}).get("_tuned") or {}).get("pa_selection_params") or {})
            pri = (tuned.get("candidate_prior_bonus_by_prefix") or {})
            pb = pri.get("boundary(")
            ps = pri.get("supar(")
            if pb is None or ps is None:
                continue
            # top-k는 seed 성능 기준으로 고르되, stage2에서는 추가 레버가 지정되면 함께 유지한다.
            key_parts = [f"{float(pb):.6f}", f"{float(ps):.6f}"]
            if "boundary_aware_weight" in tuned:
                key_parts.append(f"bw={float(tuned['boundary_aware_weight']):.6f}")
            style = tuned.get("boundary_style_prior") or {}
            if isinstance(style, dict):
                if style.get("weight_terminal") is not None:
                    key_parts.append(f"st={float(style['weight_terminal']):.6f}")
                if style.get("weight_continuation") is not None:
                    key_parts.append(f"sc={float(style['weight_continuation']):.6f}")
            if "max_candidates_multiplier" in tuned:
                key_parts.append(f"mc={int(tuned['max_candidates_multiplier'])}")
            if "penalty_empty_src" in tuned:
                key_parts.append(f"pe={float(tuned['penalty_empty_src']):.6f}")
            psp = tuned.get("penalty_short_pairs") or {}
            if isinstance(psp, dict) and psp.get("penalty_per_pair") is not None:
                key_parts.append(f"psp={float(psp['penalty_per_pair']):.6f}")
            name = "/".join(key_parts)
            if name in selected_names:
                continue
            selected_names.add(name)
            selected_cfgs.append(
                {
                    "prior_boundary": float(pb),
                    "prior_supar": float(ps),
                    "boundary_aware_weight": float(tuned["boundary_aware_weight"]) if "boundary_aware_weight" in tuned else None,
                    "style_weight_terminal": float(style.get("weight_terminal")) if isinstance(style, dict) and style.get("weight_terminal") is not None else None,
                    "style_weight_continuation": float(style.get("weight_continuation")) if isinstance(style, dict) and style.get("weight_continuation") is not None else None,
                    "max_candidates_multiplier": int(tuned["max_candidates_multiplier"]) if "max_candidates_multiplier" in tuned else None,
                    "penalty_empty_src": float(tuned["penalty_empty_src"]) if "penalty_empty_src" in tuned else None,
                    "penalty_short_per_pair": float(psp.get("penalty_per_pair")) if isinstance(psp, dict) and psp.get("penalty_per_pair") is not None else None,
                }
            )
            if len(selected_cfgs) >= top_k:
                break

        # stage2: 상위 config만 원래 sample-size로 실행
        results, success, failed = _run_grid(selected_cfgs, seeds, int(args.sample_size), output_dir)
        stage2_meta = {
            "dir": str(output_dir),
            "success": int(success),
            "failed": int(failed),
            "total_experiments": int(len(selected_cfgs) * len(seeds)),
            "sample_size": int(args.sample_size),
            "seeds": list(seeds),
            "selected_configs": selected_cfgs,
        }
    else:
        results, success, failed = _run_grid(configs, seeds, int(args.sample_size), output_dir)
        stage2_meta = {
            "dir": str(output_dir),
            "success": int(success),
            "failed": int(failed),
            "total_experiments": int(len(configs) * len(seeds)),
            "sample_size": int(args.sample_size),
            "seeds": list(seeds),
        }

    elapsed = time.time() - start

    # staged일 때는 stage1+stage2 합산으로 표시
    total_success = int(success) + int(stage1_meta["success"] if stage1_meta else 0)
    total_failed = int(failed) + int(stage1_meta["failed"] if stage1_meta else 0)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(configs),
        "total_experiments": total_experiments,
        "success": total_success,
        "failed": total_failed,
        "elapsed_seconds": elapsed,
        "staged": bool(args.staged),
        "stage1": stage1_meta,
        "stage2": stage2_meta,
        "results": results,
    }

    out_path = output_dir / "summary.json"
    _save_json(out_path, summary)

    print("=" * 80)
    print("Grid Search 완료")
    print("=" * 80)
    print(f"총 실험 횟수: {total_experiments}")
    print(f"성공: {total_success}/{total_experiments}")
    print(f"실패: {total_failed}/{total_experiments}")
    print(f"소요 시간: {elapsed/60:.1f}분")
    print("=" * 80)
    print(f"\n[OK] 결과 저장: {out_path}")
    print("\n다음 단계:")
    print(f"  python scripts/summarize_grid_search.py {output_dir}")

    # 전부 실패면 CI/배치에서 바로 감지되도록 비정상 종료
    if total_success == 0 and total_failed > 0:
        raise SystemExit("[ERROR] 모든 실험이 실패했습니다. Docker/환경 로그를 확인하세요.")


if __name__ == "__main__":
    main()
