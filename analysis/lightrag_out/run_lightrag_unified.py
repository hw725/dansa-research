"""Build a UNIFIED LightRAG knowledge graph from all 4 categories
and run cross-category comparison queries.

The unified KG allows direct横断 comparison that per-category KGs cannot.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status

_aclient = AsyncOpenAI()

ROOT = Path(__file__).resolve().parent.parent  # dansa-research/analysis
OUT = ROOT / "lightrag_out"
RESULTS = OUT / "results_unified"
RESULTS.mkdir(parents=True, exist_ok=True)

LLM_MODEL = "gpt-5-mini"
LLM_FALLBACK = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072

REASONING_EFFORT = "minimal"

setup_logger("lightrag", level="WARNING")

CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]


# ---- LLM / Embedding wrappers (same as run_lightrag.py) ----------------------
async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    history_messages = history_messages or []
    kwargs.pop("temperature", None)
    kwargs.pop("max_tokens", None)
    kwargs.setdefault("reasoning_effort", REASONING_EFFORT)
    try:
        return await openai_complete_if_cache(
            LLM_MODEL, prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("model", "not found", "unsupported", "reasoning_effort")):
            kwargs.pop("reasoning_effort", None)
            print(f"[llm_func] fallback to {LLM_FALLBACK}", flush=True)
            return await openai_complete_if_cache(
                LLM_FALLBACK, prompt, system_prompt=system_prompt,
                history_messages=history_messages, **kwargs,
            )
        raise


async def embed_func(texts):
    """Direct OpenAI call — bypasses LightRAG's openai_embed which doubles vectors."""
    texts = [t[:8191] if t else " " for t in texts]
    resp = await _aclient.embeddings.create(input=texts, model=EMB_MODEL)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# ---- Document builder (reuses per-category format) ----------------------------
def build_all_documents(samples: dict) -> list[tuple[str, str]]:
    docs = []
    for cat in CATS:
        clusters = samples[cat]
        for cid, c in clusters.items():
            if cat.endswith("_O"):
                decision = "행동·태도 결정(O)"
            else:
                decision = "행동·태도 비결정(X)"
            if "니라" in cat:
                marker = "니라"
            else:
                marker = "라"

            lines = [
                f"# Cluster {cat} #{cid}",
                f"Category: {cat}",
                f"종결어미: {marker}",
                f"판정: {decision}",
                f"Cluster ID: {cid}",
                f"Cluster size (3-model consensus rows): {c['size']}",
                "",
                "## Cluster summary (LLM-generated)",
                c["summary"].strip(),
                "",
                f"## All {len(c['sentences'])} sentences (ordered by similarity to centroid)",
            ]
            for i, s in enumerate(c["sentences"], 1):
                lines.append(f"{i}. [{s['book']}] marker={s['marker_raw']}")
                lines.append(f"   原文: {s['원문']}")
                lines.append(f"   번역: {s['번역문']}")
            doc_id = f"{cat}__cluster_{cid}"
            docs.append((doc_id, "\n".join(lines)))
    return docs


# ---- Cross-category query battery --------------------------------------------
def build_cross_queries() -> list[tuple[str, str]]:
    preamble = (
        "[통합 분석: 4범주 전수 11,327건 — I_니라_O(5,085건), II_니라_X(1,715건), "
        "III_라_O(3,082건), IV_라_X(1,445건)]\n\n"
        "판정 기준: '행동·태도를 분명하게 결정하며 종결하는 형태' 여부 (O/X)\n"
        "任圭直 夬絶 가설: 종결어미 '니라'가 '라'보다 행동·태도 결정 종결을 더 자주 표지한다.\n\n"
    )
    return [
        ("CQ1_nira_O_vs_ra_O",
         preamble +
         "I_니라_O 클러스터들과 III_라_O 클러스터들을 직접 대비하라. "
         "두 범주 모두 '행동·태도 결정(O)' 판정이지만 종결어미가 다르다(니라 vs 라). "
         "주제·내용 패턴, 행동·태도 결정 표지의 강도·빈도, 인과·결론 접속표지, 시제·서법에서 "
         "어떤 체계적 차이가 관찰되는가? 근거 문장을 인용하라."),

        ("CQ2_nira_X_vs_ra_X",
         preamble +
         "II_니라_X 클러스터들과 IV_라_X 클러스터들을 직접 대비하라. "
         "두 범주 모두 '행동·태도 비결정(X)' 판정이지만 종결어미가 다르다(니라 vs 라). "
         "비-결단 문장들 사이에서도 니라와 라의 사용 맥락이 다른가? 차이가 있다면 구체적 근거를 제시하라."),

        ("CQ3_nira_O_vs_nira_X",
         preamble +
         "I_니라_O 클러스터들과 II_니라_X 클러스터들을 직접 대비하라. "
         "같은 종결어미 '니라'이지만 행동·태도 결정 여부가 다르다(O vs X). "
         "O 판정 문장과 X 판정 문장의 주제·내용·서법·논리구조에서 어떤 체계적 차이가 나타나는가?"),

        ("CQ4_ra_O_vs_ra_X",
         preamble +
         "III_라_O 클러스터들과 IV_라_X 클러스터들을 직접 대비하라. "
         "같은 종결어미 '라'이지만 행동·태도 결정 여부가 다르다(O vs X). "
         "O 판정 문장과 X 판정 문장의 주제·내용·서법·논리구조에서 어떤 체계적 차이가 나타나는가?"),

        ("CQ5_O_all_vs_X_all",
         preamble +
         "행동·태도 결정(O) 범주(I + III) 전체와 비표명(X) 범주(II + IV) 전체를 대비하라. "
         "종결어미와 무관하게, O 판정 문장과 X 판정 문장은 어떤 언어적·내용적 특징으로 구별되는가? "
         "행동·태도 결정 표지, 인과 접속표지, 시제·서법 분포의 차이를 정리하라."),

        ("CQ6_nira_all_vs_ra_all",
         preamble +
         "종결어미 '니라' 범주(I + II) 전체와 '라' 범주(III + IV) 전체를 대비하라. "
         "판정 여부와 무관하게, 니라 종결 문장과 라 종결 문장은 어떤 주제·서법·논리구조적 차이를 보이는가? "
         "夬絶 가설과의 관련성을 평가하라."),

        ("CQ7_hypothesis_verdict",
         preamble +
         "종합 판정: 통합 지식그래프의 4범주 횡단 비교를 바탕으로, "
         "任圭直의 夬絶 가설 — '니라'가 '라'보다 행동·태도 결정을 명시적으로 표명하며 "
         "종결하는 형태를 더 자주 표지한다 — 을 평가하라. "
         "가설을 지지하는 근거, 가설에 부합하지 않는 근거를 각각 정리하고, "
         "최종적으로 데이터가 가설을 어느 정도 지지하는지 결론을 내려라."),
    ]


# ---- Main --------------------------------------------------------------------
async def main():
    only_query = "--only-query" in sys.argv

    samples_path = OUT / "cluster_full_v2.json"
    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)

    workdir = OUT / "ALL_unified"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Unified workdir: {workdir}", flush=True)

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
        chunk_token_size=2400,
        chunk_overlap_token_size=200,
        llm_model_max_async=4,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    global REASONING_EFFORT

    if not only_query:
        REASONING_EFFORT = "minimal"
        docs = build_all_documents(samples)
        print(f"Inserting {len(docs)} cluster documents from all 4 categories …", flush=True)
        ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        t0 = time.time()
        await rag.ainsert(texts, ids=ids,
                          file_paths=[f"unified/{did}" for did in ids])
        print(f"Insert done in {time.time()-t0:.1f}s", flush=True)

    REASONING_EFFORT = "medium"
    queries = build_cross_queries()
    answers = {}
    for qid, q in queries:
        out_file = RESULTS / f"unified__{qid}.md"
        if out_file.exists() and out_file.stat().st_size > 200:
            print(f"  [skip] {qid} already exists", flush=True)
            answers[qid] = out_file.read_text(encoding="utf-8")
            continue
        print(f"  query {qid} …", flush=True)
        t0 = time.time()
        try:
            ans = await rag.aquery(q, param=QueryParam(mode="mix", top_k=60, response_type="Multiple Paragraphs"))
        except Exception as e:
            ans = f"ERROR: {e}\n\n{traceback.format_exc()}"
        elapsed = time.time() - t0
        answers[qid] = ans
        out_file.write_text(
            f"# Unified — {qid}\n\n## Question\n{q}\n\n## Answer\n{ans}\n\n_(elapsed: {elapsed:.1f}s)_\n",
            encoding="utf-8")
        print(f"    -> {len(ans) if ans else 0} chars in {elapsed:.1f}s", flush=True)

    with open(RESULTS / "unified_answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    await rag.finalize_storages()
    print(f"\nDONE. Results in {RESULTS}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
