"""L2 embedding pipeline: merge 3-model consensus → embed → save.

Input:  l2_analysis/{gpt5mini,gemini,claude_sonnet}_l2.csv
Output: parallel_data_v2.tsv, emb_openai_v2.npy
Checkpoint: emb_checkpoint.npz (auto-resume on restart)
"""
import os, sys, time
import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = r"C:/Users/junto/Downloads/analysis_v8"
L2_DIR = os.path.join(ROOT, "l2_analysis")
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072
BATCH_SIZE = 200
CHECKPOINT = os.path.join(ROOT, "emb_checkpoint.npz")

MERGE_KEYS = ["book", "문단식별자", "문장식별자", "marker_type"]

client = OpenAI()


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=EMB_MODEL)
    return [d.embedding for d in resp.data]


def build_consensus() -> pd.DataFrame:
    gpt = pd.read_csv(os.path.join(L2_DIR, "gpt5mini_l2.csv"))
    gem = pd.read_csv(os.path.join(L2_DIR, "gemini_l2.csv"))
    cla = pd.read_csv(os.path.join(L2_DIR, "claude_sonnet_l2.csv"))
    print(f"Loaded: gpt={len(gpt)}, gemini={len(gem)}, claude={len(cla)}")

    base = gpt[MERGE_KEYS + ["원문", "번역문", "marker_raw", "marker_normalized", "dansa_category"]].copy()
    base["B_gpt"] = gpt["llm_judgment"].values
    base["B_gem"] = gem.set_index(MERGE_KEYS).loc[
        base.set_index(MERGE_KEYS).index, "llm_judgment"
    ].values
    base["B_cla"] = cla.set_index(MERGE_KEYS).loc[
        base.set_index(MERGE_KEYS).index, "llm_judgment"
    ].values

    base["unanimous_O"] = base["B_gpt"] & base["B_gem"] & base["B_cla"]
    base["unanimous_X"] = ~base["B_gpt"] & ~base["B_gem"] & ~base["B_cla"]
    consensus = base[base["unanimous_O"] | base["unanimous_X"]].copy()

    def assign_cell(row):
        mt, ox = row["marker_type"], "O" if row["unanimous_O"] else "X"
        prefix = {"니라": {"O": "I", "X": "II"}, "라": {"O": "III", "X": "IV"}}
        return f"{prefix[mt][ox]}_{mt}_{ox}"

    consensus["cell"] = consensus.apply(assign_cell, axis=1)
    consensus["parallel"] = consensus["원문"].astype(str) + "\n" + consensus["번역문"].astype(str)
    consensus = consensus.reset_index(drop=True)
    return consensus


def run_embedding(consensus: pd.DataFrame):
    n = len(consensus)
    texts = consensus["parallel"].tolist()

    if os.path.exists(CHECKPOINT):
        ckpt = np.load(CHECKPOINT)
        emb = ckpt["emb"]
        done = int(ckpt["done"])
        print(f"Resuming from checkpoint: {done}/{n}")
    else:
        emb = np.zeros((n, EMB_DIM), dtype=np.float32)
        done = 0

    if done >= n:
        print("Embedding already complete.")
        return emb

    t0 = time.time()
    for batch_start in range(done, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        vectors = embed_batch(texts[batch_start:batch_end])
        for i, vec in enumerate(vectors):
            emb[batch_start + i] = vec

        if (batch_end // BATCH_SIZE) % 5 == 0 or batch_end == n:
            np.savez(CHECKPOINT, emb=emb, done=np.array(batch_end))

        elapsed = time.time() - t0
        print(f"  Embedded {batch_end}/{n} ({elapsed:.1f}s)", flush=True)

    return emb


def main():
    # 1. Build consensus
    consensus = build_consensus()
    print(f"\nConsensus: {len(consensus)} rows")
    print(consensus["cell"].value_counts().sort_index().to_string())

    # 2. Save parallel_data first (no API dependency)
    out_cols = ["cell", "book", "문단식별자", "문장식별자", "marker_raw", "marker_type",
                "marker_normalized", "dansa_category", "원문", "번역문", "parallel"]
    out_tsv = os.path.join(ROOT, "parallel_data_v2.tsv")
    consensus[out_cols].to_csv(out_tsv, sep="\t", index=False, encoding="utf-8")
    print(f"Saved {out_tsv}")

    # 3. Embed
    print(f"\nStarting embedding ({EMB_MODEL}, {EMB_DIM}d)...")
    emb = run_embedding(consensus)

    # 4. Save embeddings
    out_npy = os.path.join(ROOT, "emb_openai_v2.npy")
    np.save(out_npy, emb)
    print(f"Saved {out_npy} shape={emb.shape}")

    # 5. Cleanup checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print("\n=== Summary ===")
    for cell in sorted(consensus["cell"].unique()):
        print(f"  {cell}: {(consensus['cell'] == cell).sum()}")
    print(f"  Total: {len(consensus)}")


if __name__ == "__main__":
    main()
