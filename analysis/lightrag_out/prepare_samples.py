"""Dump ALL sentences per cluster (full-data ingestion).

Sentences within each cluster are ordered by cosine-similarity to the cluster
centroid (most-representative first) so that LightRAG's chunk windows surface
core meanings before edge cases.
"""
import json, csv, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lightrag_out"

# 1. Load parallel data
rows = []
with open(ROOT / "parallel_data_v2_cleaned.tsv", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)
print(f"parallel_data: {len(rows)} rows")
print(f"columns: {list(rows[0].keys())}")

# 2. Load cluster info
with open(ROOT / "clusters_v2.json", encoding="utf-8") as f:
    clusters = json.load(f)

# 3. Load embeddings + summaries
emb = np.load(ROOT / "emb_openai_v2.npy")
print(f"embeddings shape: {emb.shape}")

with open(ROOT / "cluster_summaries_v2.json", encoding="utf-8") as f:
    summaries = json.load(f)

# 4. For each (category, cluster), pick top-K sentences nearest to centroid
samples = {}  # {category: {cluster_id: {"size": n, "summary": str, "sentences": [{idx, 원문, 번역문, book}, ...]}}}
rng = np.random.default_rng(42)

for cat, info in clusters.items():
    labels = np.array(info["labels"])
    global_idx = np.array(info["global_idx"])
    centroids = np.array(info["centroids"])
    samples[cat] = {}

    for cid in range(info["best_k"]):
        in_cluster = np.where(labels == cid)[0]
        cluster_global = global_idx[in_cluster]
        cluster_emb = emb[cluster_global]
        # Cosine similarity to centroid
        c = centroids[cid]
        c_norm = c / (np.linalg.norm(c) + 1e-9)
        e_norm = cluster_emb / (np.linalg.norm(cluster_emb, axis=1, keepdims=True) + 1e-9)
        sims = e_norm @ c_norm
        # Use ALL sentences in this cluster, ordered by similarity to centroid (descending)
        order = np.argsort(-sims)
        chosen = cluster_global[order]
        sentences = []
        for gi in chosen:
            r = rows[int(gi)]
            sentences.append({
                "idx": int(gi),
                "book": r.get("book", ""),
                "원문": r.get("원문", "").strip(),
                "번역문": r.get("번역문", "").strip(),
                "marker_raw": r.get("marker_raw", ""),
            })
        cluster_summary = summaries.get(cat, {}).get(str(cid), {}).get("summary", "")
        samples[cat][str(cid)] = {
            "size": int((labels == cid).sum()),
            "summary": cluster_summary,
            "sentences": sentences,
        }

# 5. Save
out_path = os.path.join(OUT, "cluster_full.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

# 6. Print summary
print("\n=== Full-data summary ===")
total = 0
for cat, cl in samples.items():
    n_cl = len(cl)
    n_sent = sum(len(c["sentences"]) for c in cl.values())
    total += n_sent
    print(f"  {cat}: {n_cl} clusters, {n_sent} sentences")
    for cid, c in cl.items():
        print(f"    [{cid}] size={c['size']:>5} (==sentences {len(c['sentences'])})")
print(f"\nTotal sentences: {total}")
print(f"Saved: {out_path}")
