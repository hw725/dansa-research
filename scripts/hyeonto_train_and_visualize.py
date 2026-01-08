#!/usr/bin/env python3
"""hyeonto 전용: (PA 데이터셋 기반) 경계 클러스터링 → 재클러스터(child) → 현토 임베딩 시각화

산출물은 모두 hyeonto/reports/ 아래에 생성한다.

파이프라인(간소화, 비지도 기반):
1) hyeonto/datasets/pa/train.csv로부터 boundary 기능 클러스터링(k=16)
2) parent cluster 내부를 재클러스터링하여 child_cluster_id 부여
3) parent_cluster_id/child_cluster_id + 원문 컨텍스트를 CSV(reclustered.csv)로 저장
4) visualize_hyeonto_semantic_roles.py로 marker_semantic_embedding.* 생성

주의:
- 기존 "정답 parent/child"가 있는 학습 파이프라인이 아니라, 현토 데이터만으로 재구성하는 최소 E2E이다.
- 필요 시 이후 label/학습(train_pa_* 등)로 확장 가능.

실행 예:
  python scripts/hyeonto_train_and_visualize.py --device-id 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

HYEONTO_DIR = WORKSPACE_ROOT / "hyeonto"
DATASETS_PA = HYEONTO_DIR / "datasets" / "pa" / "train.csv"
REPORTS_DIR = HYEONTO_DIR / "reports"

TRASH_SCRIPTS = WORKSPACE_ROOT / "trash" / "pre_revert_20260105_233346" / "untracked" / "scripts"


def _ensure_script_available(name: str) -> Path:
    # prefer workspace scripts/ if present, else fallback to trash backup
    p1 = WORKSPACE_ROOT / "scripts" / name
    if p1.exists():
        return p1
    p2 = TRASH_SCRIPTS / name
    if p2.exists():
        return p2
    raise SystemExit(f"필요 스크립트를 찾지 못했습니다: {name}")


def _run_py(cmd: str) -> None:
    import subprocess

    print(f"\n$ {cmd}")
    subprocess.check_call(cmd, cwd=str(WORKSPACE_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="hyeonto: cluster boundaries and visualize marker roles")
    ap.add_argument("--input", type=str, default=str(DATASETS_PA), help="입력 CSV 경로 (기본: hyeonto/datasets/pa/train.csv)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--child-k", type=int, default=16)
    ap.add_argument("--parent-min-size", type=int, default=50, help="재클러스터 대상 parent 최소 크기")
    ap.add_argument("--min-per-child", type=int, default=50, help="child K 자동 축소용 (n//min_per_child)")
    ap.add_argument("--max-boundaries", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=None)
    ap.add_argument("--clean", action="store_true", help="hyeonto/reports 를 비우고 다시 생성")
    args = ap.parse_args()

    input_csv = Path(args.input)
    if not input_csv.exists():
        raise SystemExit(f"입력 데이터셋이 없습니다: {input_csv}")

    if bool(args.clean) and REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) parent clustering
    cluster_py = _ensure_script_available("cluster_pa_boundary_functions.py")
    out_parent = REPORTS_DIR / "boundary_function_clusters"
    out_parent.mkdir(parents=True, exist_ok=True)

    cmd_parent = [
        sys.executable,
        str(cluster_py),
        "--input",
        str(input_csv),
        "--out-dir",
        str(out_parent),
        "--k",
        str(int(args.k)),
        "--seed",
        str(int(args.seed)),
        "--max-boundaries",
        str(int(args.max_boundaries)),
    ]
    if args.device_id is not None:
        cmd_parent += ["--device-id", str(int(args.device_id))]
    _run_py(cmd_parent)

    parent_csv = out_parent / "boundary_clusters.csv"
    if not parent_csv.exists():
        # non-baseline naming fallback (shouldn't happen with k=64 baseline; but we use k=args.k)
        # so find latest boundary_clusters_*.csv
        candidates = sorted(out_parent.glob("boundary_clusters_*.csv"))
        if not candidates:
            raise SystemExit(f"parent boundary csv를 찾지 못했습니다: {out_parent}")
        parent_csv = candidates[-1]

    # 2) recluster within parent clusters
    recluster_py = _ensure_script_available("recluster_within_boundary_clusters.py")
    out_recluster = REPORTS_DIR / "recluster_k16_child"
    out_recluster.mkdir(parents=True, exist_ok=True)

    # The recluster script expects --csv input and writes a reclustered.csv
    cmd_child = [
        sys.executable,
        str(recluster_py),
        "--csv",
        str(parent_csv),
        "--out-dir",
        str(out_recluster),
        "--child-k",
        str(int(args.child_k)),
        "--parent-min-size",
        str(int(args.parent_min_size)),
        "--min-per-child",
        str(int(args.min_per_child)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.device_id is not None:
        cmd_child += ["--device-id", str(int(args.device_id))]
    _run_py(cmd_child)

    reclustered_csv = out_recluster / "reclustered.csv"
    if not reclustered_csv.exists():
        # fallback: any reclustered*.csv
        candidates = sorted(out_recluster.glob("reclustered*.csv"))
        if not candidates:
            raise SystemExit(f"reclustered.csv를 찾지 못했습니다: {out_recluster}")
        reclustered_csv = candidates[-1]

    # 3) visualize marker semantic roles
    viz_py = WORKSPACE_ROOT / "scripts" / "visualize_hyeonto_semantic_roles.py"
    if not viz_py.exists():
        # fallback to trash
        viz_py = _ensure_script_available("visualize_hyeonto_semantic_roles.py")

    out_viz = REPORTS_DIR / "k16_analysis"
    out_viz.mkdir(parents=True, exist_ok=True)

    cmd_viz = [
        sys.executable,
        str(viz_py),
        "--csv",
        str(reclustered_csv),
        "--out-dir",
        str(out_viz),
        "--save-html",
        "--save-parent-child-comparison",
    ]
    _run_py(cmd_viz)

    print("\n✅ done")
    print(f"- reclustered: {reclustered_csv}")
    print(f"- viz out: {out_viz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
