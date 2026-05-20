"""Build per-category LightRAG knowledge graphs and run analytical queries.

Categories:
  I_니라_O   (3-model consensus 결단=O for 니라)
  II_니라_X  (3-model consensus 결단=X for 니라)
  III_라_O   (3-model consensus 결단=O for 라)
  IV_라_X   (3-model consensus 결단=X for 라)

For each category we:
  1. Build documents per cluster (summary + 30 representative sentences).
  2. Insert into a dedicated LightRAG working_dir (idempotent — resumes on re-run).
  3. Run a fixed battery of 7 analytical queries (mix mode).
  4. Save query responses + entity counts.

Outputs land under lightrag_out/<category>/ and lightrag_out/results/.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lightrag_out"
RESULTS = OUT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LLM_MODEL = "gpt-5-mini"
LLM_FALLBACK = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072

# Phase-controlled reasoning effort. Set globally before insert vs query phases.
# - "minimal": fast, used for entity/relation extraction (highly structured)
# - "medium" : default reasoning, used for analytical query answers
REASONING_EFFORT = "minimal"

setup_logger("lightrag", level="WARNING")  # quiet


# ---- LLM / Embedding wrappers --------------------------------------------------
async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    history_messages = history_messages or []
    # Strip params that gpt-5-mini reasoning model rejects
    kwargs.pop("temperature", None)
    kwargs.pop("max_tokens", None)
    # Inject phase-controlled reasoning effort
    kwargs.setdefault("reasoning_effort", REASONING_EFFORT)
    try:
        return await openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    except Exception as e:
        msg = str(e)
        # Fallback if model not available or param rejected
        if any(s in msg.lower() for s in ("model", "not found", "unsupported", "reasoning_effort")):
            kwargs.pop("reasoning_effort", None)
            print(f"[llm_func] {LLM_MODEL} failed ({msg[:120]}), falling back to {LLM_FALLBACK} without reasoning_effort", flush=True)
            return await openai_complete_if_cache(
                LLM_FALLBACK,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
        raise


async def embed_func(texts):
    import openai as _openai
    client = _openai.AsyncOpenAI()
    resp = await client.embeddings.create(input=texts, model=EMB_MODEL)
    return np.array([item.embedding for item in resp.data])


SUB_DOC_SIZE = 50  # sentences per sub-document to prevent OOM

def build_documents(cat: str, clusters: dict) -> list[tuple[str, str]]:
    """Return [(doc_id, doc_text)]. Large clusters split into sub-docs of SUB_DOC_SIZE."""
    docs = []
    for cid, c in clusters.items():
        sents = c["sentences"]
        n_parts = max(1, (len(sents) + SUB_DOC_SIZE - 1) // SUB_DOC_SIZE)

        for part_idx in range(n_parts):
            start = part_idx * SUB_DOC_SIZE
            end = min(start + SUB_DOC_SIZE, len(sents))
            chunk = sents[start:end]

            lines = [
                f"# Cluster {cat} #{cid} (part {part_idx+1}/{n_parts})",
                f"Category: {cat}",
                f"Cluster ID: {cid}",
                f"Cluster size (total): {c['size']}",
            ]
            if part_idx == 0:
                lines += [
                    "",
                    "## Cluster summary (LLM-generated)",
                    c["summary"].strip(),
                ]
            lines += [
                "",
                f"## Sentences {start+1}-{end} of {len(sents)}",
            ]
            for i, s in enumerate(chunk, start + 1):
                lines.append(f"{i}. [{s['book']}] marker={s['marker_raw']}")
                lines.append(f"   原文: {s['원문']}")
                lines.append(f"   번역: {s['번역문']}")

            doc_id = f"{cat}__cluster_{cid}__p{part_idx}"
            docs.append((doc_id, "\n".join(lines)))
    return docs


# ---- Query battery -------------------------------------------------------------
def build_queries(cat: str) -> list[tuple[str, str]]:
    """Return [(qid, question)]. Same battery for all 4 categories so we can compare."""
    if cat.endswith("_O"):
        decision_label = "행동·태도 결정(O)"
    else:
        decision_label = "행동·태도 비결정(X)"
    if "니라" in cat:
        marker = "'니라'"
    else:
        marker = "'라'"

    # Prefix every query with the category tag so cache keys differ per cat,
    # protecting against any cross-instance cache pollution.
    tag = f"[분석 대상: {cat} ({marker}, {decision_label})]\n\n"
    return [
        ("Q1_themes",
         tag + f"이 카테고리({cat})는 종결어미 {marker}이며 3개 LLM 일치 판정 {decision_label}로 분류된 한문 번역문들이다. "
         f"이 카테고리 전체에 걸쳐 가장 두드러지는 주제·내용 패턴 5가지를 근거 인용과 함께 요약하라."),
        ("Q2_cluster_diff",
         tag + f"{cat} 카테고리 안의 클러스터들 간에 의미적·내용적 차이는 무엇인가? "
         f"각 클러스터를 한 문장으로 특징짓고, 클러스터들 사이의 핵심 대비점을 정리하라."),
        ("Q3_decision_features",
         tag + f"{cat} 카테고리 종결문의 행동·태도 결정 표지(예: 是, 非, 可, 不可, 故, 當, 必 등 한문 / "
         f"옳다, 그르다, 마땅하다, ~해야 한다 등 번역)가 얼마나 자주, 어떤 패턴으로 나타나는지 분석하라. "
         f"근거 문장을 인용하라."),
        ("Q4_logical_markers",
         tag + f"{cat} 카테고리에 빈번한 인과·결론·조건 접속표지(故, 是以, 然이나, 若…則…, ~이니, ~이므로 등)와 "
         "그 사용 양상의 특징을 정리하라."),
        ("Q5_tense_mood",
         tag + f"{cat} 카테고리의 시제·서법 분포(당위/규범, 정의/현재, 과거사건, 인용중계 등) 중 어느 것이 가장 우세하며, "
         "그것이 종결어미 선택과 어떤 관계가 있는가?"),
        ("Q6_sources",
         tag + f"{cat} 카테고리에서 가장 많이 등장하는 출처 문헌(book) 또는 인물·개념을 5~10개 나열하고, "
         "그 분포가 종결어미 선택과 의미하는 바를 해석하라."),
        # Q7 제거: 단일 범주 KG로는 횡단 비교 불가. 정량 비교는 pandas/scipy로 별도 수행.
    ]


# ---- Main ----------------------------------------------------------------------
async def run_category(cat: str, clusters: dict, only_query: bool = False):
    workdir = OUT / cat.replace("/", "_")
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  [{cat}]  workdir={workdir}\n{'='*70}", flush=True)

    rag = LightRAG(
        working_dir=str(workdir),
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=8192,
            func=embed_func,
        ),
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        chunk_token_size=2400,           # bigger chunks → fewer LLM calls on full-data
        chunk_overlap_token_size=200,
        llm_model_max_async=2,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    global REASONING_EFFORT

    if not only_query:
        # Insert per-cluster documents — fast structured extraction
        REASONING_EFFORT = "minimal"
        docs = build_documents(cat, clusters)
        print(f"  inserting {len(docs)} cluster documents (reasoning_effort=minimal) …", flush=True)
        ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        t0 = time.time()
        await rag.ainsert(texts, ids=ids, file_paths=[f"{cat}/cluster_{i}" for i in ids])
        print(f"  insert done in {time.time()-t0:.1f}s", flush=True)

    # Run queries — analytical, use medium reasoning
    REASONING_EFFORT = "medium"
    queries = build_queries(cat)
    answers = {}
    for qid, q in queries:
        out_file = RESULTS / f"{cat}__{qid}.md"
        if out_file.exists() and out_file.stat().st_size > 200:
            print(f"  [skip] {qid} already exists", flush=True)
            answers[qid] = out_file.read_text(encoding="utf-8")
            continue
        print(f"  query {qid} …", flush=True)
        t0 = time.time()
        try:
            ans = await rag.aquery(q, param=QueryParam(mode="mix", top_k=20, response_type="Multiple Paragraphs"))
        except Exception as e:
            ans = f"ERROR: {e}\n\n{traceback.format_exc()}"
        elapsed = time.time() - t0
        answers[qid] = ans
        out_file.write_text(f"# {cat} — {qid}\n\n## Question\n{q}\n\n## Answer\n{ans}\n\n_(elapsed: {elapsed:.1f}s)_\n", encoding="utf-8")
        print(f"    -> {len(ans)} chars in {elapsed:.1f}s", flush=True)

    await rag.finalize_storages()
    return answers


async def main():
    # Optional CLI: python run_category.py [cat1 cat2 ...] [--only-query]
    only_query = "--only-query" in sys.argv
    cli_cats = [a for a in sys.argv[1:] if not a.startswith("--")]

    samples_path = OUT / "cluster_full_v2.json"
    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)

    cats = cli_cats if cli_cats else list(samples.keys())
    print(f"Running categories: {cats}  (only_query={only_query})", flush=True)

    all_answers = {}
    for cat in cats:
        try:
            ans = await run_category(cat, samples[cat], only_query=only_query)
            all_answers[cat] = ans
        except Exception:
            print(f"  [FAIL] {cat}", flush=True)
            traceback.print_exc()
            continue

    # Save consolidated
    consolidated = RESULTS / "all_answers.json"
    with open(consolidated, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    print(f"\nDONE. Consolidated -> {consolidated}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
