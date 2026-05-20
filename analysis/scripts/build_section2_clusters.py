"""Rebuild Section 2 clustering + summaries from current embeddings.

Input:  parallel_data_v2_cleaned.tsv, emb_openai_v2.npy
Output: clusters_v2.json, cluster_summaries_v2.json, lightrag_out/cluster_full_v2.json
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # dansa-research/analysis
OUT = os.path.join(ROOT, "lightrag_out")
CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]
K_RANGE = range(2, 13)

client = OpenAI()


def find_best_k(emb_subset, k_range):
    if len(emb_subset) < max(k_range):
        k_range = range(2, min(len(emb_subset), 13))
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(emb_subset)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(emb_subset, labels, sample_size=min(5000, len(emb_subset)))
    best_k = max(scores, key=scores.get)
    print(f"    K scores: {', '.join(f'{k}={s:.3f}' for k, s in sorted(scores.items()))}")
    print(f"    Best K={best_k} (silhouette={scores[best_k]:.3f})")
    return best_k


def summarize_cluster(cat, cid, sentences, marker_type, ox):
    sample = sentences[:20]
    text_block = "\n".join(f"- {s['번역문']}" for s in sample)
    decision = "행동·태도 결정(O)" if ox == "O" else "행동·태도 비결정(X)"
    prompt = (
        f"다음은 한문 번역문 클러스터({cat}, 종결어미 '{marker_type}', {decision})의 "
        f"대표 문장 {len(sample)}개입니다.\n\n{text_block}\n\n"
        f"이 클러스터의 주제·내용적 특징을 3-5문장으로 요약하세요."
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def main():
    df = pd.read_csv(os.path.join(ROOT, "parallel_data_v2_cleaned.tsv"), sep="\t", encoding="utf-8")
    emb = np.load(os.path.join(ROOT, "emb_openai_v2.npy"))
    print(f"Loaded: {len(df)} rows, emb {emb.shape}")

    clusters_out = {}
    summaries_out = {}
    cluster_full = {}

    for cat in CATS:
        mask = df["cell"] == cat
        cat_idx = np.where(mask.values)[0]
        cat_emb = emb[cat_idx]
        mt = "니라" if "니라" in cat else "라"
        ox = "O" if cat.endswith("_O") else "X"
        print(f"\n{'='*60}\n  {cat}: {len(cat_idx)} sentences\n{'='*60}")

        # 1. Find optimal K
        best_k = find_best_k(cat_emb, K_RANGE)

        # 2. Run final K-means
        km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
        labels = km.fit_predict(cat_emb)
        centroids = km.cluster_centers_

        clusters_out[cat] = {
            "labels": labels.tolist(),
            "global_idx": cat_idx.tolist(),
            "centroids": centroids.tolist(),
            "best_k": best_k,
        }

        # 3. Build per-cluster data + summaries
        summaries_out[cat] = {}
        cluster_full[cat] = {}

        for cid in range(best_k):
            in_cluster = np.where(labels == cid)[0]
            cluster_global = cat_idx[in_cluster]
            cluster_emb = cat_emb[in_cluster]

            # Sort by similarity to centroid
            c = centroids[cid]
            c_norm = c / (np.linalg.norm(c) + 1e-9)
            e_norm = cluster_emb / (np.linalg.norm(cluster_emb, axis=1, keepdims=True) + 1e-9)
            sims = e_norm @ c_norm
            order = np.argsort(-sims)
            chosen = cluster_global[order]

            sentences = []
            for gi in chosen:
                r = df.iloc[int(gi)]
                sentences.append({
                    "idx": int(gi),
                    "book": r.get("book", ""),
                    "원문": str(r.get("원문", "")).strip(),
                    "번역문": str(r.get("번역문", "")).strip(),
                    "marker_raw": r.get("marker_raw", ""),
                })

            # Generate summary
            print(f"  Cluster {cid}: {len(sentences)} sentences, summarizing...", flush=True)
            t0 = time.time()
            summary = summarize_cluster(cat, cid, sentences, mt, ox)
            print(f"    Summary done ({time.time()-t0:.1f}s)", flush=True)

            summaries_out[cat][str(cid)] = {"summary": summary}
            cluster_full[cat][str(cid)] = {
                "size": len(sentences),
                "summary": summary,
                "sentences": sentences,
            }

    # Save
    with open(os.path.join(ROOT, "clusters_v2.json"), "w", encoding="utf-8") as f:
        json.dump(clusters_out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved clusters_v2.json")

    with open(os.path.join(ROOT, "cluster_summaries_v2.json"), "w", encoding="utf-8") as f:
        json.dump(summaries_out, f, ensure_ascii=False, indent=2)
    print(f"Saved cluster_summaries_v2.json")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cluster_full_v2.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_full, f, ensure_ascii=False, indent=2)
    print(f"Saved lightrag_out/cluster_full_v2.json")

    # Final summary
    print("\n=== Final Summary ===")
    total = 0
    for cat in CATS:
        n_cl = len(cluster_full[cat])
        n_sent = sum(c["size"] for c in cluster_full[cat].values())
        total += n_sent
        print(f"  {cat}: {n_cl} clusters, {n_sent} sentences")
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
