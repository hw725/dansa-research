#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SA 경계(구↔구 사이) '기능'을 비지도적으로 군집화한다.

PA와 달리 SA는 구(phrase) 단위이므로:
- 입력: hyeonto/datasets/sa_merged_v2.csv (문장식별자, 구식별자, 원문, 번역문, book_name)
- 출력: out_dir/sa_boundary_clusters.csv, out_dir/sa_boundary_clusters.md
- 경계: 같은 문장 내에서 구n ↔ 구n+1 사이
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
    """SA 구↔구 경계"""
    book_name: str
    sentence_id: int
    left_phrase_id: int
    right_phrase_id: int
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


def load_sa_boundaries(csv_path: str, max_boundaries: int) -> List[BoundaryInstance]:
    """SA CSV에서 구↔구 경계 생성"""
    df = pd.read_csv(csv_path)
    print(f"SA CSV 로드: {len(df):,}행")

    required_cols = ['book_name', '문장식별자', '구식별자', '원문', '번역문']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}. 사용 가능: {df.columns.tolist()}")

    boundaries = []
    grouped = df.groupby(['book_name', '문장식별자'])

    print(f"고유 (book_name, 문장식별자) 조합: {len(grouped):,}개")

    for (book, sent_id), group in grouped:
        group = group.sort_values('구식별자')
        phrases = group.to_dict('records')

        for i in range(len(phrases) - 1):
            left = phrases[i]
            right = phrases[i + 1]

            boundaries.append(BoundaryInstance(
                book_name=str(book),
                sentence_id=int(sent_id),
                left_phrase_id=int(left['구식별자']),
                right_phrase_id=int(right['구식별자']),
                src_left=str(left['원문']),
                src_right=str(right['원문']),
                tgt_left=str(left['번역문']),
                tgt_right=str(right['번역문'])
            ))
            if len(boundaries) >= max_boundaries:
                break
        if len(boundaries) >= max_boundaries:
            break

    print(f"구↔구 경계 생성: {len(boundaries):,}개")
    return boundaries


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


def cluster_kmeans(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    print(f"MiniBatchKMeans 클러스터링: k={k}, seed={seed}")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels


def save_results(instances: List[BoundaryInstance], labels: np.ndarray, out_dir: Path, seed: int):
    import regex as re
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 마커 추출 함수
    _CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")
    HYEONTO_REPLACE_MAP = {"은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가"}
    
    def normalize_marker(m):
        if not m: return m
        if m in HYEONTO_REPLACE_MAP: return HYEONTO_REPLACE_MAP[m]
        if len(m) > 1 and (m.startswith("이") or m.startswith("으")): return m[1:]
        return m
    
    def extract_markers(text):
        if pd.isna(text) or not text: return ""
        markers = [normalize_marker(m.group("marker")) for m in _CJK_MARKER_RE.finditer(str(text)) if m.group("marker")]
        return ",".join(markers) if markers else ""
    
    records = []
    for inst, label in zip(instances, labels):
        marker_left = extract_markers(inst.src_left)
        marker_right = extract_markers(inst.src_right)
        
        # 대표 마커 (left의 마지막 또는 right의 첫 번째)
        left_markers = marker_left.split(",") if marker_left else []
        right_markers = marker_right.split(",") if marker_right else []
        if left_markers and left_markers[-1]:
            marker_normalized = left_markers[-1]
        elif right_markers and right_markers[0]:
            marker_normalized = right_markers[0]
        else:
            marker_normalized = ""
        
        records.append({
            'cluster_id': int(label),
            'book_name': inst.book_name,
            'sentence_id': inst.sentence_id,
            'left_phrase_id': inst.left_phrase_id,
            'right_phrase_id': inst.right_phrase_id,
            'src_left': inst.src_left,
            'src_right': inst.src_right,
            'tgt_left': inst.tgt_left,
            'tgt_right': inst.tgt_right,
            'marker_left': marker_left,
            'marker_right': marker_right,
            'marker_normalized': marker_normalized,
        })
    df = pd.DataFrame(records)
    cluster_sizes = df['cluster_id'].value_counts().to_dict()
    df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
    df.to_csv(out_dir / 'sa_boundary_clusters.csv', index=False, encoding='utf-8-sig')
    
    with open(out_dir / 'sa_boundary_clusters.md', 'w', encoding='utf-8') as f:
        f.write(f"# SA boundary clusters (seed={seed})\n\n")
        for cid in sorted(cluster_sizes.keys()):
            f.write(f"## cluster {cid} (n={cluster_sizes[cid]})\n\n")
            samples = df[df['cluster_id'] == cid].head(3)
            for _, r in samples.iterrows():
                f.write(f"- book={r['book_name']}, sent={r['sentence_id']}\n")
                f.write(f"  - src_L: {r['src_left']}\n  - src_R: {r['src_right']}\n")
                f.write(f"  - tgt_L: {r['tgt_left']}\n  - tgt_R: {r['tgt_right']}\n")
                f.write(f"  - marker: {r['marker_normalized']}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="SA 구↔구 경계 클러스터링 (BGE-M3)")
    parser.add_argument('--input', required=True, help='입력 CSV 경로')
    parser.add_argument('--out-dir', required=True, help='출력 디렉토리')
    parser.add_argument('--k', type=int, default=16, help='클러스터 개수')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument("--max-boundaries", type=int, default=300000)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--use-src", action="store_true", default=True)
    parser.add_argument("--use-tgt", action="store_true", default=True)

    # 임베딩 캐시 옵션
    parser.add_argument("--load-embeddings", type=str, default=None, help="임베딩 캐시 로드 경로 (.npy)")
    parser.add_argument("--save-embeddings", type=str, default=None, help="임베딩 캐시 저장 경로 (.npy)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries = load_sa_boundaries(args.input, args.max_boundaries)
    if not boundaries:
        print("경계가 없습니다.")
        return

    # 임베딩 캐시 로드 또는 계산
    if args.load_embeddings and Path(args.load_embeddings).exists():
        print(f"✅ 임베딩 캐시 로드: {args.load_embeddings}")
        X = np.load(args.load_embeddings)
        X = _l2_normalize(X)
    else:
        texts = [inst.to_embed_text(use_src=args.use_src, use_tgt=args.use_tgt) for inst in boundaries]
        X = compute_embeddings_batched(texts, batch_size=args.batch, device_id=args.device_id)
        X = _l2_normalize(X)
        
        # 임베딩 캐시 저장
        if args.save_embeddings:
            save_path = Path(args.save_embeddings)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), X)
            print(f"✅ 임베딩 캐시 저장: {save_path}")

    labels = cluster_kmeans(X, k=args.k, seed=args.seed)
    save_results(boundaries, labels, out_dir, seed=args.seed)
    print("\n✅ 완료!")


if __name__ == '__main__':
    main()
