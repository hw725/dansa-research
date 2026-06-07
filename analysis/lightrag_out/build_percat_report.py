# -*- coding: utf-8 -*-
"""LightRAG per-category report builder (v4 successor).
Reads results/{cat}__{qid}.md (24 files) -> REPORT_v4.md + all_answers_v2.json.
Updated for cleaned data (11,327 rows).
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parent
RES = OUT / "results"

CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]
CAT_COUNTS = {"I_니라_O": 5085, "II_니라_X": 1715, "III_라_O": 3082, "IV_라_X": 1445}
CAT_CLUSTERS = {"I_니라_O": 2, "II_니라_X": 2, "III_라_O": 2, "IV_라_X": 12}
CAT_MARKERS = {"I_니라_O": "nira", "II_니라_X": "nira", "III_라_O": "ra", "IV_라_X": "ra"}
CAT_DECISION = {"I_니라_O": "O", "II_니라_X": "X", "III_라_O": "O", "IV_라_X": "X"}

QIDS = ["Q1_themes", "Q2_cluster_diff", "Q3_decision_features",
        "Q4_logical_markers", "Q5_tense_mood", "Q6_sources"]
QTITLES = {
    "Q1_themes":            "Q1. Main topic patterns",
    "Q2_cluster_diff":      "Q2. Inter-cluster differences",
    "Q3_decision_features": "Q3. Decision markers",
    "Q4_logical_markers":   "Q4. Causal/conclusion/conditional connectives",
    "Q5_tense_mood":        "Q5. Tense/mood distribution",
    "Q6_sources":           "Q6. Source texts/persons/concepts",
}


def load_answer(cat: str, q: str) -> str:
    p = RES / f"{cat}__{q}.md"
    if not p.exists():
        return "(no result)"
    txt = p.read_text(encoding="utf-8")
    if "## Answer" in txt:
        body = txt.split("## Answer", 1)[-1]
        for sep in ["_(elapsed", "## Metadata"]:
            if sep in body:
                body = body.split(sep, 1)[0]
        return body.strip()
    return txt.strip()


def build_consolidated_json():
    all_answers = {}
    for c in CATS:
        all_answers[c] = {}
        for q in QIDS:
            all_answers[c][q] = load_answer(c, q)
    out = RES / "all_answers_v2.json"
    out.write_text(json.dumps(all_answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_markdown_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# LightRAG Per-Category Analysis Report v4: hypothesis verification")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("> **v4 notes**: (1) Cleaned data -- removed contaminated Jachi tonggam gangmok vols 8-20. "
                 "(2) text-embedding-3-large (3072d) unified. "
                 "(3) Per-category KG x 6 queries = 24 analyses. "
                 "(4) Cross-category comparison in separate unified report (REPORT_v5.md).")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("- **Corpus**: parallel_data_v2_cleaned.tsv (11,327 rows, 3-model consensus)")
    lines.append("- **Method**: LightRAG 1.4.15, per-category KG build + 6 analytical queries")
    lines.append("- **LLM**: gpt-5-mini (insert: reasoning_effort=minimal, query: medium)")
    lines.append("- **Embedding**: text-embedding-3-large (3072d)")
    lines.append("- **Isolation**: separate Python subprocess per category (singleton KV pollution prevention)")
    lines.append("- **chunk_size**: 2400 tokens, concurrency 2")
    lines.append("")
    lines.append("### 4-Category Data")
    lines.append("")
    lines.append("| Category | Ending | Decision | Sentences | Clusters |")
    lines.append("|---|---|---|---:|---:|")
    for c in CATS:
        lines.append(f"| {c} | {CAT_MARKERS[c]} | {CAT_DECISION[c]} | {CAT_COUNTS[c]:,} | {CAT_CLUSTERS[c]} |")
    lines.append(f"| **Total** | | | **{sum(CAT_COUNTS.values()):,}** | **{sum(CAT_CLUSTERS.values())}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-category section
    for cat in CATS:
        n = CAT_COUNTS[cat]
        k = CAT_CLUSTERS[cat]
        lines.append(f"## {cat} ({n:,} sentences, {k} clusters)")
        lines.append("")
        for q in QIDS:
            lines.append(f"### {QTITLES[q]}")
            lines.append("")
            ans = load_answer(cat, q)
            lines.append(ans)
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"*LightRAG Per-Category Report v4 -- {now}*")

    text = "\n".join(lines)
    out_path = OUT / "REPORT_v4.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path, len(text)


if __name__ == "__main__":
    print("[1/2] all_answers_v2.json ...")
    jp = build_consolidated_json()
    print(f"  -> {jp}")

    print("[2/2] REPORT_v4.md ...")
    rp, n = build_markdown_report()
    print(f"  -> {rp}  ({n:,} chars)")

    # Check completeness
    missing = []
    for c in CATS:
        for q in QIDS:
            p = RES / f"{c}__{q}.md"
            if not p.exists() or p.stat().st_size < 200:
                missing.append(f"{c}/{q}")
    if missing:
        print(f"\nWARNING: {len(missing)} missing/incomplete:")
        for m in missing:
            print(f"  - {m}")
    else:
        print(f"\nAll {len(CATS) * len(QIDS)} query results present.")

    print("DONE")
