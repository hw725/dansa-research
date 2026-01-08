#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA 경계(문장↔문장 사이) '기능'을 비지도적으로 군집화한다.

- 입력: datasets/pa/train.csv (문단식별자, 문장식별자, 원문, 번역문, book_name)
- 출력: out_dir/boundary_clusters.csv, out_dir/boundary_clusters.md

이 파일은 trash 백업에 있던 스크립트를 workspace scripts/로 옮겨온 것입니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass
class BoundaryInstance:
    book_name: str
    paragraph_id: int
    left_sentence_id: int
    right_sentence_id: int
    src_left: str
    src_right: str
    tgt_left: str
    tgt_right: str

    def to_embed_text(self, use_src: bool = True, use_tgt: bool = True) -> str:
        parts: list[str] = []
        if use_src:
            parts.append("[SRC_L] " + (self.src_left or ""))
            parts.append("[SRC_R] " + (self.src_right or ""))
        if use_tgt:
            parts.append("[TGT_L] " + (self.tgt_left or ""))
            parts.append("[TGT_R] " + (self.tgt_right or ""))
        return "\n".join(parts)

    def to_cue_text(self, use_src: bool = True, use_tgt: bool = True, left_chars: int = 40, right_chars: int = 40) -> str:
        parts: list[str] = []
        if use_src:
            parts.append("[SRC_L_TAIL] " + _tail(self.src_left, left_chars))
            parts.append("[SRC_R_HEAD] " + _head(self.src_right, right_chars))
        if use_tgt:
            parts.append("[TGT_L_TAIL] " + _tail(self.tgt_left, left_chars))
            parts.append("[TGT_R_HEAD] " + _head(self.tgt_right, right_chars))
        return "\n".join(parts)

    def to_hybrid_text(
        self,
        use_src: bool = True,
        use_tgt: bool = True,
        left_chars: int = 40,
        right_chars: int = 40,
        mid_chars: int = 160,
    ) -> str:
        parts: list[str] = [self.to_cue_text(use_src=use_src, use_tgt=use_tgt, left_chars=left_chars, right_chars=right_chars)]
        if use_src:
            parts.append("[SRC_L] " + _clip(self.src_left, mid_chars))
            parts.append("[SRC_R] " + _clip(self.src_right, mid_chars))
        if use_tgt:
            parts.append("[TGT_L] " + _clip(self.tgt_left, mid_chars))
            parts.append("[TGT_R] " + _clip(self.tgt_right, mid_chars))
        return "\n".join([p for p in parts if p])


def _clip(s: str, n: int) -> str:
    s2 = (s or "").strip()
    return s2 if len(s2) <= n else s2[:n] + "…"


def _head(s: str, n: int) -> str:
    s2 = (s or "").lstrip()
    return s2 if len(s2) <= n else s2[:n] + "…"


def _tail(s: str, n: int) -> str:
    s2 = (s or "").rstrip()
    if len(s2) <= n:
        return s2
    return "…" + s2[-n:]


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


def load_pa_sentence_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["문단식별자", "문장식별자", "원문", "번역문", "book_name"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"입력 CSV에 {c} 컬럼이 없습니다. 사용 가능: {sorted(df.columns)}")
    return df


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


def iter_boundary_instances(df: pd.DataFrame) -> Iterable[BoundaryInstance]:
    # group by (book_name, paragraph)
    gcols = ["book_name", "문단식별자"]
    for (book, para), gdf in df.groupby(gcols):
        # sentence_id numeric sort
        gdf2 = gdf.copy()
        gdf2["_sid"] = gdf2["문장식별자"].apply(lambda v: _safe_int(v, default=-1))
        gdf2 = gdf2.sort_values(by=["_sid"], kind="mergesort")
        rows = list(gdf2.itertuples(index=False))
        for i in range(0, max(0, len(rows) - 1)):
            left = rows[i]
            right = rows[i + 1]
            yield BoundaryInstance(
                book_name=str(getattr(left, "book_name")),
                paragraph_id=_safe_int(getattr(left, "문단식별자"), default=-1),
                left_sentence_id=_safe_int(getattr(left, "문장식별자"), default=-1),
                right_sentence_id=_safe_int(getattr(right, "문장식별자"), default=-1),
                src_left=str(getattr(left, "원문", "")),
                src_right=str(getattr(right, "원문", "")),
                tgt_left=str(getattr(left, "번역문", "")),
                tgt_right=str(getattr(right, "번역문", "")),
            )


def write_markdown_report(out_path: Path, df: pd.DataFrame, top_k: int, seed: int) -> None:
    lines: list[str] = []
    lines.append(f"# boundary clusters (seed={seed})")
    lines.append("")

    for cid, gdf in df.groupby("cluster_id"):
        lines.append(f"## cluster {cid} (n={len(gdf)})")
        lines.append("")
        for _, r in gdf.head(int(top_k)).iterrows():
            lines.append(f"- book={r.get('book_name','')}, para={r.get('paragraph_id','')}")
            lines.append(f"  - src_L: {r.get('src_left','')}")
            lines.append(f"  - src_R: {r.get('src_right','')}")
            lines.append(f"  - tgt_L: {r.get('tgt_left','')}")
            lines.append(f"  - tgt_R: {r.get('tgt_right','')}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise SystemExit(
            "scikit-learn이 필요합니다.\n"
            "- 도커: `docker compose build csp` 후 재실행 (또는 컨테이너에서 `pip install scikit-learn`)\n"
            "- 로컬: `.venv`에 `pip install scikit-learn`\n"
            f"원인: {e}"
        )

    p = argparse.ArgumentParser(description="PA 경계 기능 비지도 클러스터링 (PA-only)")
    p.add_argument(
        "--input",
        type=str,
        default=str(WORKSPACE_ROOT / "datasets" / "pa" / "train.csv"),
        help="입력 CSV (기본: datasets/pa/train.csv)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "reports" / "boundary_function_clusters"),
        help="출력 디렉토리",
    )
    p.add_argument("--max-boundaries", type=int, default=20000, help="최대 경계 샘플 수")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--use-src", action="store_true", help="임베딩 입력에 src 포함")
    p.add_argument("--use-tgt", action="store_true", help="임베딩 입력에 tgt 포함")

    p.add_argument("--device-id", type=int, default=None, help="BGE 임베딩에 사용할 GPU id")
    p.add_argument("--batch", type=int, default=64, help="임베딩 배치")

    p.add_argument("--k", type=int, default=64, help="클러스터 수(KMeans K)")
    p.add_argument("--top-k", type=int, default=3, help="리포트 예시 개수")

    args = p.parse_args()

    use_src = bool(args.use_src)
    use_tgt = bool(args.use_tgt)
    if not (use_src or use_tgt):
        use_src = True
        use_tgt = True

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pa_sentence_pairs(in_path)
    instances: List[BoundaryInstance] = []
    for inst in iter_boundary_instances(df):
        instances.append(inst)
        if len(instances) >= int(args.max_boundaries):
            break

    if not instances:
        raise SystemExit("경계 인스턴스가 0개입니다. 입력/정렬 상태를 확인하세요.")

    texts = [x.to_embed_text(use_src=use_src, use_tgt=use_tgt) for x in instances]
    X = compute_embeddings_batched(texts, batch_size=int(args.batch), device_id=args.device_id)
    X = _l2_normalize(X)

    k = max(2, min(int(args.k), len(instances)))
    km = MiniBatchKMeans(n_clusters=k, random_state=int(args.seed), batch_size=1024, n_init="auto")
    cluster_id = km.fit_predict(X)

    out_dict = {
        "cluster_id": cluster_id,
        "book_name": [x.book_name for x in instances],
        "paragraph_id": [x.paragraph_id for x in instances],
        "left_sentence_id": [x.left_sentence_id for x in instances],
        "right_sentence_id": [x.right_sentence_id for x in instances],
        "src_left": [x.src_left for x in instances],
        "src_right": [x.src_right for x in instances],
        "tgt_left": [x.tgt_left for x in instances],
        "tgt_right": [x.tgt_right for x in instances],
    }

    df_out = pd.DataFrame(out_dict)

    counts = df_out["cluster_id"].value_counts().to_dict()
    df_out["cluster_size"] = df_out["cluster_id"].map(lambda c: int(counts.get(int(c), 0)))
    df_out = df_out.sort_values(by=["cluster_size", "cluster_id"], ascending=[False, True], kind="mergesort")

    out_csv = out_dir / "boundary_clusters.csv"
    out_md = out_dir / "boundary_clusters.md"

    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_markdown_report(out_md, df_out, top_k=int(args.top_k), seed=int(args.seed))

    print(f"✅ wrote: {out_csv}")
    print(f"✅ wrote: {out_md}")


if __name__ == "__main__":
    main()
