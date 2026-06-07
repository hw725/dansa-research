# -*- coding: utf-8 -*-
"""DCI vs LightRAG comparison + cross-category synthesis reports.
Cleaned data (11,327 rows).
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DCI_RES = ROOT / "dci_out" / "results"
DCI_CROSS = ROOT / "dci_out" / "cross"
DCI_LOG = ROOT / "dci_out" / "dci_experiment_log.json"
DCI_CROSS_LOG = ROOT / "dci_out" / "dci_cross_log.json"
LRAG_UNIFIED = ROOT / "lightrag_out" / "results_unified"

LQ = "“"  # left curly quote
RQ = "”"  # right curly quote


def load_answer(path: Path) -> str:
    if not path.exists():
        return "(no result)"
    txt = path.read_text(encoding="utf-8")
    if "## Answer" in txt:
        body = txt.split("## Answer", 1)[-1]
        for sep in ["## DCI Metadata", "## Metadata", "## LightRAG Metadata", "_(elapsed"]:
            if sep in body:
                body = body.split(sep, 1)[0]
        return body.strip()
    return txt.strip()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def q(s):
    """Wrap s in curly quotes."""
    return f"{LQ}{s}{RQ}"


def build_comparison_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dci_log = load_json(DCI_LOG)
    dci_clog = load_json(DCI_CROSS_LOG)

    dci_tok = dci_log.get("total_tokens", 0) + dci_clog.get("total_tokens", 0)
    dci_per = dci_log.get("results", [])
    dci_cq = dci_clog.get("results", [])
    dci_time = sum(r.get("total_s", 0) for r in dci_per) + sum(r.get("total_s", 0) for r in dci_cq)

    L = []
    L.append("# DCI vs LightRAG comparison: hypothesis verification")
    L.append("")
    L.append(f"Generated: {now}")
    L.append("")
    L.append("> Cleaned data (11,327 rows). DCI (full-corpus grep) + LightRAG (KG search)")
    L.append("> independent analysis comparison. Convergent findings listed as core results.")
    L.append("")

    L.append("## 1. Method Comparison")
    L.append("")
    L.append("| Axis | DCI | LightRAG |")
    L.append("|---|---|---|")
    L.append("| Approach | Raw corpus full grep/awk -> LLM synthesis | KG build -> entity/relation search |")
    L.append("| Corpus | parallel_data_v2_cleaned.tsv (11,327) | 18 cluster summaries -> unified KG |")
    L.append("| LLM | gpt-5-mini | gpt-5-mini |")
    L.append("| Embedding | N/A | text-embedding-3-large (3072d) |")
    L.append("| Coverage | 100% (full access) | ~56% (top_k=60, mix mode) |")
    L.append("| Preprocessing cost | $0 | embedding + KG build |")
    L.append(f"| Total tokens | {dci_tok:,} | (see unified_answers.json) |")
    L.append(f"| Total time | {dci_time:.0f}s | (separate measurement) |")
    L.append("| Analysis units | per-cat Q1-Q6 (24) + cross CQ1-CQ6 (6) | cross CQ1-CQ7 (7) |")
    L.append("| Strength | Exact freq/ratio, tail patterns, source dist | Conceptual relations, semantic clustering |")
    L.append("| Weakness | No structural relation inference | Imprecise quantification, category omission |")
    L.append("")

    L.append("## 2. Convergent Core Findings (DCI + LightRAG agree)")
    L.append("")
    L.append("Below: identical conclusions reached independently by both methods. Highest confidence.")
    L.append("")

    L.append("### 2.1 I(nira+decision): Four-Books moral argumentation")
    L.append("")
    L.append(f"- **DCI**: Mengzi jizhu(662), Lunyu jizhu(608) top. "
             f"Gu(829, 16.3%), Ke(742, 14.6%) dominant. "
             f"{q('RuCi hanira')}(66), {q('ErYiYi nira')}(56) argumentative/limiting closers.")
    L.append("- **LightRAG**: Confucian normative themes dominant. Junzi/xiaoren contrast, li/renyi/xiao moral-cultivation center.")
    L.append(f"- **Convergence**: Four-Books-based moral argumentation as {q('decision')}.")
    L.append("")

    L.append("### 2.2 II(nira+non-decision): Historical narrative records")
    L.append("")
    L.append("- **DCI**: Tangsung paldeaga muncho Suchol2(585) top. Gu(86, 5.0%) decision markers very low.")
    L.append("- **LightRAG**: Office transfers, military campaigns, ritual/funerary fact records.")
    L.append("- **Convergence**: Literary-collection narrative. Decision markers appear but in non-normative functions.")
    L.append("")

    L.append("### 2.3 III(ra+decision): Five-Classics adjudication/norms")
    L.append("")
    L.append(f"- **DCI**: Zhouyi jeonui(698) top. Ke(447, 14.5%) dominant, Ze(613, 19.9%) high ratio. "
             f"{q('ShiYe ra')}(35), {q('LiYe ra')}(24) definitional/adjudicative closers. Ruo...Ze 71 cases.")
    L.append("- **LightRAG**: Divination auspicious/inauspicious, balance principles, ritual norms.")
    L.append(f"- **Convergence**: Five-Classics yixue/ritual adjudication. Distinct from nira's argumentative {q('decision')}.")
    L.append("")

    L.append("### 2.4 IV(ra+non-decision): Heterogeneous function set")
    L.append("")
    L.append("- **DCI**: Scattered across Shijing/Tangsung. Decision markers very low(Gu 43, 3.0%). 12 clusters(silhouette 0.063).")
    L.append("- **LightRAG**: Commentarial definitions, lexical glosses, exclamations, biographical narratives.")
    L.append(f"- **Convergence**: Not a single category but a mix of heterogeneous functions. Reflects {q('ra')}'s versatility.")
    L.append("")

    L.append(f"### 2.5 Core insight: Same {q('decision')}, different character")
    L.append("")
    L.append("I(nira+decision) and III(ra+decision) both normative-dominant, but:")
    L.append("")
    L.append("| Dimension | I nira+decision | III ra+decision |")
    L.append("|---|---|---|")
    L.append("| Sources | Four Books (Mengzi/Lunyu) | Five Classics (Zhouyi/Shijing/Shujing) |")
    L.append("| Decision character | Moral argumentation conclusion | Yixue/ritual adjudication |")
    L.append(f"| Top marker | Gu(829, 16.3%) {q('therefore')} | Ke(447, 14.5%) {q('permissible')} |")
    L.append("| Logic structure | Gu->conclusion (causal) | Ruo...Ze->adjudication (conditional) |")
    L.append(f"| Tail pattern | {q('RuCi hanira')} {q('ErYiYi nira')} | {q('ShiYe ra')} {q('LiYe ra')} {q('RuCi ra')} |")
    L.append("")
    L.append(f"-> **Both methods confirm**: {q('decision')} content differs by ending marker. "
             "Gwoljeol/mijeol difference is multidimensional, not a single-axis intensity.")
    L.append("")

    L.append("## 3. Method-specific unique contributions")
    L.append("")
    L.append("### 3.1 DCI unique (quantitative precision)")
    L.append("")
    L.append("- Exact frequencies: Gu 829(16.3%), Ke 742(14.6%)")
    L.append(f"- Tail pattern full census: {q('RuCi hanira')} 66, {q('ShiYe ra')} 35")
    L.append(f"- Translation {q('geureumeuro')} freq: I=556, II=23, III=255, IV=9")
    L.append("- Ruo...Ze compound: 71 in III, confirming conditional->adjudication chain")
    L.append("- IV 12-cluster heterogeneity measured (silhouette 0.063)")
    L.append("")

    L.append("### 3.2 LightRAG unique (semantic connections)")
    L.append("")
    L.append("- Entity-relation inference: junzi<->xiaoren, tianming<->renshi")
    L.append("- Semantic clustering: moral-cultivation vs ritual-institutional")
    L.append("- Scholarly context: Zhuxi interpretive tradition and ending marker selection")
    L.append("- CQ7 hypothesis verdict from KG perspective")
    L.append("")

    L.append("## 4. Hypothesis verification implications")
    L.append("")
    L.append("### 4.1 DCI + LightRAG convergent conclusions")
    L.append("")
    L.append("1. **nira O-rate > ra O-rate** (74.8% vs 68.1%, +6.7pp): statistically significant, small effect")
    L.append(f"2. **{q('Decision')} content differs by ending marker**: nira=moral argumentation, ra=yixue adjudication")
    L.append(f"3. **{q('ra+non-decision')} is heterogeneous**: {q('ra')}'s versatility makes control group impure")
    L.append("4. **Gwoljeol/mijeol difference is multidimensional**: topic/style/source entangled, "
             "single binary prompt has inherent limitations")
    L.append("")

    L.append("### 4.2 Methodological lessons")
    L.append("")
    L.append("- DCI+LightRAG convergence cancels individual method bias")
    L.append("- Quantitative(DCI) + qualitative(LightRAG) complementary use effective")
    L.append("- LightRAG-alone quantification unreliable; DCI verification required")
    L.append("")

    L.append("---")
    L.append(f"*DCI vs LightRAG Comparison Report -- {now}*")

    text = "\n".join(L)
    out = ROOT / "COMPARISON_REPORT.md"
    out.write_text(text, encoding="utf-8")
    return out, len(text)


def build_cross_category_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("# Cross-Category Synthesis Report: hypothesis verification")
    L.append("")
    L.append(f"Generated: {now}")
    L.append("")
    L.append("> DCI cross-comparison (CQ1-CQ6, full census) + LightRAG cross-comparison (CQ1-CQ7, KG search) synthesis.")
    L.append("> Convergent results as core, method-specific as supplementary.")
    L.append("")

    # DCI Cross-Category
    L.append("## Part A: DCI Cross-Category (CQ1-CQ6)")
    L.append("")
    L.append("> Single TSV simultaneous access for direct 4-category comparison.")
    L.append("> Quantitative cross-comparison impossible in LightRAG due to per-category KG isolation.")
    L.append("")

    dci_cq_ids = ["CQ1_nira_vs_ra", "CQ2_O_vs_X", "CQ3_interaction",
                  "CQ4_logical_markers", "CQ5_sources_cross", "CQ6_hypothesis_test"]
    dci_cq_titles = {
        "CQ1_nira_vs_ra": "CQ1. nira vs ra -- topic/source distribution",
        "CQ2_O_vs_X": "CQ2. decision(O) vs non-decision(X) -- decision markers",
        "CQ3_interaction": "CQ3. ending x decision interaction",
        "CQ4_logical_markers": "CQ4. connective markers 4-category cross",
        "CQ5_sources_cross": "CQ5. source x ending distribution",
        "CQ6_hypothesis_test": "CQ6. direct hypothesis test (chi2, OR, RR)",
    }

    for cqid in dci_cq_ids:
        L.append(f"### {dci_cq_titles[cqid]}")
        L.append("")
        ans = load_answer(DCI_CROSS / f"{cqid}.md")
        L.append(ans)
        L.append("")

    L.append("---")
    L.append("")

    # LightRAG Cross-Category
    L.append("## Part B: LightRAG Cross-Category (CQ1-CQ7)")
    L.append("")
    L.append("> unified KG (18 cluster summaries) mix mode (top_k=60) search.")
    L.append("> Qualitative cross-comparison via entity/relation inference.")
    L.append("")

    lrag_cq_ids = [
        "CQ1_nira_O_vs_ra_O", "CQ2_nira_X_vs_ra_X",
        "CQ3_nira_O_vs_nira_X", "CQ4_ra_O_vs_ra_X",
        "CQ5_O_all_vs_X_all", "CQ6_nira_all_vs_ra_all",
        "CQ7_hypothesis_verdict",
    ]
    lrag_cq_titles = {
        "CQ1_nira_O_vs_ra_O": "CQ1. nira+decision(I) vs ra+decision(III)",
        "CQ2_nira_X_vs_ra_X": "CQ2. nira+non-decision(II) vs ra+non-decision(IV)",
        "CQ3_nira_O_vs_nira_X": "CQ3. nira+decision(I) vs nira+non-decision(II)",
        "CQ4_ra_O_vs_ra_X": "CQ4. ra+decision(III) vs ra+non-decision(IV)",
        "CQ5_O_all_vs_X_all": "CQ5. decision(O) all vs non-decision(X) all",
        "CQ6_nira_all_vs_ra_all": "CQ6. nira all vs ra all",
        "CQ7_hypothesis_verdict": "CQ7. hypothesis final verdict",
    }

    for cqid in lrag_cq_ids:
        L.append(f"### {lrag_cq_titles[cqid]}")
        L.append("")
        ans = load_answer(LRAG_UNIFIED / f"unified__{cqid}.md")
        L.append(ans)
        L.append("")

    L.append("---")
    L.append(f"*Cross-Category Synthesis Report -- {now}*")

    text = "\n".join(L)
    out = ROOT / "CROSS_CATEGORY_REPORT.md"
    out.write_text(text, encoding="utf-8")
    return out, len(text)


if __name__ == "__main__":
    print("[1/2] COMPARISON_REPORT.md ...")
    cp, cn = build_comparison_report()
    print(f"  -> {cp}  ({cn:,} chars)")

    print("[2/2] CROSS_CATEGORY_REPORT.md ...")
    xp, xn = build_cross_category_report()
    print(f"  -> {xp}  ({xn:,} chars)")

    print("DONE")
