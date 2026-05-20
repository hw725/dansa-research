"""기존 정제 TSV에서 바로 임베딩만 수행.

Input:  analysis/parallel_data_v2_cleaned.tsv (11,327건, 정제 완료)
Output: analysis/emb_openai_v2.npy
Checkpoint: analysis/emb_checkpoint.npz (자동 재개)
"""
import os, sys, time
import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072
BATCH_SIZE = 200
CHECKPOINT = os.path.join(ROOT, "emb_checkpoint.npz")

client = OpenAI()


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=EMB_MODEL)
    return [d.embedding for d in resp.data]


def main():
    tsv_path = os.path.join(ROOT, "parallel_data_v2_cleaned.tsv")
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")

    # valid cells only
    VALID = {"I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"}
    df = df[df["cell"].isin(VALID)].reset_index(drop=True)
    print(f"Loaded: {len(df)} valid rows from {tsv_path}")
    print(df["cell"].value_counts().sort_index().to_string())

    texts = df["parallel"].tolist()
    n = len(texts)
    emb = np.zeros((n, EMB_DIM), dtype=np.float32)

    t0 = time.time()
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        vectors = embed_batch(texts[batch_start:batch_end])
        for i, vec in enumerate(vectors):
            emb[batch_start + i] = vec
        elapsed = time.time() - t0
        print(f"  Embedded {batch_end}/{n} ({elapsed:.1f}s)", flush=True)

    out_npy = os.path.join(ROOT, "emb_openai_v2.npy")
    np.save(out_npy, emb)
    print(f"\nSaved {out_npy} shape={emb.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
