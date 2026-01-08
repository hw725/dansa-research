#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""상위 boundary 클러스터(k=16 등) 내부를 2차로 재클러스터링한다.

- 입력: boundary_clusters.csv (cluster_id + 컨텍스트)
- 출력: out-dir/reclustered.csv (parent_cluster_id, child_cluster_id + 컨텍스트)

이 파일은 trash 백업에 있던 스크립트를 workspace scripts/로 옮겨온 것입니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import sys

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass
class RowInstance:
    book_name: str
    paragraph_id: int
    left_sentence_id: int
    right_sentence_id: int
    src_left: str
    src_right: str
    tgt_left: str
    tgt_right: str

    def to_embed_text(self, use_src: bool, use_tgt: bool) -> str:
        parts: list[str] = []
        if use_src:
            parts.append("[SRC_L] " + (self.src_left or ""))
            parts.append("[SRC_R] " + (self.src_right or ""))
        if use_tgt:
            parts.append("[TGT_L] " + (self.tgt_left or ""))
            parts.append("[TGT_R] " + (self.tgt_right or ""))
        return "\n".join(parts)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def compute_embeddings(texts: List[str], device_id: Optional[int]) -> np.ndarray:
    from common.embedders.bge import get_embed_func

    embed_fn = get_embed_func(device_id=device_id)
    embs = embed_fn(texts, use_multi_vector=False)
    if isinstance(embs, list):
        embs = np.array(embs)
    return np.asarray(embs, dtype=np.float32)


def compute_embeddings_batched(texts: List[str], batch_size: int, device_id: Optional[int]) -> np.ndarray:
    all_embs: List[np.ndarray] = []
    for b in batch_iter(texts, int(batch_size)):
        all_embs.append(compute_embeddings(list(b), device_id=device_id))
    return np.vstack(all_embs)


def _safe_int(x: object, default: int = -1) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _choose_child_k(n: int, child_k: int, min_per_child: int) -> int:
    if n < 4:
        return 0
    if min_per_child <= 0:
        return min(int(child_k), int(n))
    k_by_size = max(2, int(n // int(min_per_child)))
    k_eff = min(int(child_k), k_by_size)
    k_eff = max(2, min(k_eff, n))
    if k_eff >= n:
        k_eff = max(2, n // 2)
    return int(k_eff)


def main() -> int:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise SystemExit(
            "scikit-learn이 필요합니다.\n"
            "- 도커: `docker compose build csp` 후 재실행 (또는 컨테이너에서 `pip install scikit-learn`)\n"
            "- 로컬: `.venv`에 `pip install scikit-learn`\n"
            f"원인: {e}"
        )

    p = argparse.ArgumentParser(description="상위 boundary cluster 내부 2차 재클러스터링")
    p.add_argument("--csv", type=Path, required=True, help="입력 boundary_clusters*.csv")
    p.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    p.add_argument("--child-k", type=int, default=16, help="parent 내부 child K")
    p.add_argument(
        "--parent-min-size",
        type=int,
        default=0,
        help="parent 클러스터 크기가 이 값보다 작으면 재클러스터링을 건너뛰고 child=0으로 둔다",
    )
    p.add_argument("--min-per-child", type=int, default=50, help="child 최소 샘플 수 (크면 K 자동 감소)")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--use-src", action="store_true")
    p.add_argument("--use-tgt", action="store_true")

    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--batch", type=int, default=128)

    args = p.parse_args()

    in_csv = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_src = bool(args.use_src)
    use_tgt = bool(args.use_tgt)
    if not (use_src or use_tgt):
        use_src = True
        use_tgt = True

    df = pd.read_csv(in_csv)

    required_cols = [
        "cluster_id",
        "book_name",
        "paragraph_id",
        "left_sentence_id",
        "right_sentence_id",
        "src_left",
        "src_right",
        "tgt_left",
        "tgt_right",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"입력 CSV에 {c} 컬럼이 없습니다. 사용 가능: {sorted(df.columns)}")

    # instances grouped by parent cluster
    rows_out: list[dict] = []

    parent_min_size = int(args.parent_min_size)

    for parent_id, gdf in df.groupby("cluster_id"):
        parent_id_int = _safe_int(parent_id, default=-1)
        if parent_id_int < 0:
            continue

        inst: list[RowInstance] = []
        for _, r in gdf.iterrows():
            inst.append(
                RowInstance(
                    book_name=str(r.get("book_name", "")),
                    paragraph_id=_safe_int(r.get("paragraph_id"), default=-1),
                    left_sentence_id=_safe_int(r.get("left_sentence_id"), default=-1),
                    right_sentence_id=_safe_int(r.get("right_sentence_id"), default=-1),
                    src_left=str(r.get("src_left", "")),
                    src_right=str(r.get("src_right", "")),
                    tgt_left=str(r.get("tgt_left", "")),
                    tgt_right=str(r.get("tgt_right", "")),
                )
            )

        n = len(inst)
        if parent_min_size > 0 and n < parent_min_size:
            for x in inst:
                rows_out.append(
                    {
                        "parent_cluster_id": parent_id_int,
                        "child_cluster_id": 0,
                        "book_name": x.book_name,
                        "paragraph_id": x.paragraph_id,
                        "left_sentence_id": x.left_sentence_id,
                        "right_sentence_id": x.right_sentence_id,
                        "src_left": x.src_left,
                        "src_right": x.src_right,
                        "tgt_left": x.tgt_left,
                        "tgt_right": x.tgt_right,
                    }
                )
            continue
        k_eff = _choose_child_k(n, child_k=int(args.child_k), min_per_child=int(args.min_per_child))
        if k_eff <= 0:
            # too small; make single child=0
            for x in inst:
                rows_out.append(
                    {
                        "parent_cluster_id": parent_id_int,
                        "child_cluster_id": 0,
                        "book_name": x.book_name,
                        "paragraph_id": x.paragraph_id,
                        "left_sentence_id": x.left_sentence_id,
                        "right_sentence_id": x.right_sentence_id,
                        "src_left": x.src_left,
                        "src_right": x.src_right,
                        "tgt_left": x.tgt_left,
                        "tgt_right": x.tgt_right,
                    }
                )
            continue

        texts = [x.to_embed_text(use_src=use_src, use_tgt=use_tgt) for x in inst]
        X = compute_embeddings_batched(texts, batch_size=int(args.batch), device_id=args.device_id)
        X = _l2_normalize(X)

        km = MiniBatchKMeans(n_clusters=int(k_eff), random_state=int(args.seed), batch_size=1024, n_init="auto")
        child_ids = km.fit_predict(X)

        for x, c_id in zip(inst, child_ids):
            rows_out.append(
                {
                    "parent_cluster_id": parent_id_int,
                    "child_cluster_id": int(c_id),
                    "book_name": x.book_name,
                    "paragraph_id": x.paragraph_id,
                    "left_sentence_id": x.left_sentence_id,
                    "right_sentence_id": x.right_sentence_id,
                    "src_left": x.src_left,
                    "src_right": x.src_right,
                    "tgt_left": x.tgt_left,
                    "tgt_right": x.tgt_right,
                }
            )

    out_df = pd.DataFrame(rows_out)
    out_path = out_dir / "reclustered.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
