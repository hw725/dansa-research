# DCI vs LightRAG comparison: hypothesis verification

Generated: 2026-05-20 19:36

> Cleaned data (11,327 rows). DCI (full-corpus grep) + LightRAG (KG search)
> independent analysis comparison. Convergent findings listed as core results.

## 1. Method Comparison

| Axis | DCI | LightRAG |
|---|---|---|
| Approach | Raw corpus full grep/awk -> LLM synthesis | KG build -> entity/relation search |
| Corpus | parallel_data_v2_cleaned.tsv (11,327) | 18 cluster summaries -> unified KG |
| LLM | gpt-5-mini | gpt-5-mini |
| Embedding | N/A | text-embedding-3-large (3072d) |
| Coverage | 100% (full access) | ~56% (top_k=60, mix mode) |
| Preprocessing cost | $0 | embedding + KG build |
| Total tokens | 377,243 | (see unified_answers.json) |
| Total time | 1410s | (separate measurement) |
| Analysis units | per-cat Q1-Q6 (24) + cross CQ1-CQ6 (6) | cross CQ1-CQ7 (7) |
| Strength | Exact freq/ratio, tail patterns, source dist | Conceptual relations, semantic clustering |
| Weakness | No structural relation inference | Imprecise quantification, category omission |

## 2. Convergent Core Findings (DCI + LightRAG agree)

Below: identical conclusions reached independently by both methods. Highest confidence.

### 2.1 I(nira+decision): Four-Books moral argumentation

- **DCI**: Mengzi jizhu(662), Lunyu jizhu(608) top. Gu(829, 16.3%), Ke(742, 14.6%) dominant. “RuCi hanira”(66), “ErYiYi nira”(56) argumentative/limiting closers.
- **LightRAG**: Confucian normative themes dominant. Junzi/xiaoren contrast, li/renyi/xiao moral-cultivation center.
- **Convergence**: Four-Books-based moral argumentation as “decision”.

### 2.2 II(nira+non-decision): Historical narrative records

- **DCI**: Tangsung paldeaga muncho Suchol2(585) top. Gu(86, 5.0%) decision markers very low.
- **LightRAG**: Office transfers, military campaigns, ritual/funerary fact records.
- **Convergence**: Literary-collection narrative. Decision markers appear but in non-normative functions.

### 2.3 III(ra+decision): Five-Classics adjudication/norms

- **DCI**: Zhouyi jeonui(698) top. Ke(447, 14.5%) dominant, Ze(613, 19.9%) high ratio. “ShiYe ra”(35), “LiYe ra”(24) definitional/adjudicative closers. Ruo...Ze 71 cases.
- **LightRAG**: Divination auspicious/inauspicious, balance principles, ritual norms.
- **Convergence**: Five-Classics yixue/ritual adjudication. Distinct from nira's argumentative “decision”.

### 2.4 IV(ra+non-decision): Heterogeneous function set

- **DCI**: Scattered across Shijing/Tangsung. Decision markers very low(Gu 43, 3.0%). 12 clusters(silhouette 0.063).
- **LightRAG**: Commentarial definitions, lexical glosses, exclamations, biographical narratives.
- **Convergence**: Not a single category but a mix of heterogeneous functions. Reflects “ra”'s versatility.

### 2.5 Core insight: Same “decision”, different character

I(nira+decision) and III(ra+decision) both normative-dominant, but:

| Dimension | I nira+decision | III ra+decision |
|---|---|---|
| Sources | Four Books (Mengzi/Lunyu) | Five Classics (Zhouyi/Shijing/Shujing) |
| Decision character | Moral argumentation conclusion | Yixue/ritual adjudication |
| Top marker | Gu(829, 16.3%) “therefore” | Ke(447, 14.5%) “permissible” |
| Logic structure | Gu->conclusion (causal) | Ruo...Ze->adjudication (conditional) |
| Tail pattern | “RuCi hanira” “ErYiYi nira” | “ShiYe ra” “LiYe ra” “RuCi ra” |

-> **Both methods confirm**: “decision” content differs by ending marker. Gwoljeol/mijeol difference is multidimensional, not a single-axis intensity.

## 3. Method-specific unique contributions

### 3.1 DCI unique (quantitative precision)

- Exact frequencies: Gu 829(16.3%), Ke 742(14.6%)
- Tail pattern full census: “RuCi hanira” 66, “ShiYe ra” 35
- Translation “geureumeuro” freq: I=556, II=23, III=255, IV=9
- Ruo...Ze compound: 71 in III, confirming conditional->adjudication chain
- IV 12-cluster heterogeneity measured (silhouette 0.063)

### 3.2 LightRAG unique (semantic connections)

- Entity-relation inference: junzi<->xiaoren, tianming<->renshi
- Semantic clustering: moral-cultivation vs ritual-institutional
- Scholarly context: Zhuxi interpretive tradition and ending marker selection
- CQ7 hypothesis verdict from KG perspective

## 4. Hypothesis verification implications

### 4.1 DCI + LightRAG convergent conclusions

1. **nira O-rate > ra O-rate** (74.8% vs 68.1%, +6.7pp): statistically significant, small effect
2. **“Decision” content differs by ending marker**: nira=moral argumentation, ra=yixue adjudication
3. **“ra+non-decision” is heterogeneous**: “ra”'s versatility makes control group impure
4. **Gwoljeol/mijeol difference is multidimensional**: topic/style/source entangled, single binary prompt has inherent limitations

### 4.2 Methodological lessons

- DCI+LightRAG convergence cancels individual method bias
- Quantitative(DCI) + qualitative(LightRAG) complementary use effective
- LightRAG-alone quantification unreliable; DCI verification required

---
*DCI vs LightRAG Comparison Report -- 2026-05-20 19:36*