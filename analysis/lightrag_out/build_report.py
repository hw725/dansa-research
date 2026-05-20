"""Aggregate per-category LightRAG answers into a single comparison report
for the 任圭直 夬絶 hypothesis paper.

Output: REPORT.md (Korean) + all_answers.json (consolidated).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parent
RES = OUT / "results"

CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]
QIDS = ["Q1_themes", "Q2_cluster_diff", "Q3_decision_features",
        "Q4_logical_markers", "Q5_tense_mood", "Q6_sources"]
QTITLES = {
    "Q1_themes":            "Q1. 주요 주제 패턴",
    "Q2_cluster_diff":      "Q2. 클러스터 간 차이",
    "Q3_decision_features": "Q3. 행동·태도 결정 표지",
    "Q4_logical_markers":   "Q4. 인과·결론·조건 접속표지",
    "Q5_tense_mood":        "Q5. 시제·서법 분포",
    "Q6_sources":           "Q6. 출처 문헌·인물·개념",
}


def load_answer(cat: str, q: str) -> str:
    p = RES / f"{cat}__{q}.md"
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    body = txt.split("## Answer", 1)[-1].rsplit("_(elapsed", 1)[0]
    return body.strip()


def build_consolidated_json():
    all_answers = {}
    for c in CATS:
        all_answers[c] = {q: load_answer(c, q) for q in QIDS}
    with open(RES / "all_answers_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)


def build_markdown_report():
    lines = []
    lines.append("# LightRAG 클러스터 분석 보고서 v4: 任圭直 夬絶 가설 검증")
    lines.append("")
    lines.append(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("> **v4 변경사항**: (1) LightRAG 임베딩을 text-embedding-3-large(3072d)로 통일 "
                 "(v3에서 실수로 text-embedding-3-small 사용 → 검색 품질 저하). "
                 "(2) per-category Q7(가설 평가) 제거 — 단일 범주 KG로는 횡단 비교 불가. "
                 "(3) unified KG 횡단 비교(CQ1~CQ7) 폐기 — LightRAG는 정량 비교 도구가 아님, "
                 "top_k 제한으로 범주 누락 반복. 정량 비교는 pandas/scipy로 직접 수행.")
    lines.append("")
    lines.append("## 개요")
    lines.append("")
    lines.append("- **분석 단위**: 4범주 × 18클러스터, 총 16,381건 (3개 LLM 일치 판정 기준)")
    lines.append("- **판정 기준**: 행동·태도를 분명하게 결정하며 종결하는 형태 여부 (O/X)")
    lines.append("- **방법**: LightRAG 1.4.15로 카테고리별 지식그래프 4개 구축 후 6개 분석 질의 실행")
    lines.append("- **LLM**: gpt-5-mini (인서트 reasoning_effort=minimal, 질의 reasoning_effort=medium)")
    lines.append("- **임베딩**: text-embedding-3-large (3072d, 클러스터링 및 LightRAG 내부 통일)")
    lines.append("- **격리**: 카테고리별 KG는 별도 Python subprocess (LightRAG 싱글턴 KV 교차오염 차단)")
    lines.append("- **정량 비교**: pandas/scipy (chi-square, OR, sign test) — stats/ 폴더 참조")
    lines.append("- **chunk size**: 2400 토큰, 동시성 4")
    lines.append("")
    lines.append("### 4범주 데이터")
    lines.append("")
    lines.append("| 범주 | 종결어미 | 행동·태도 결정 | 문장 수 | 클러스터 수 |")
    lines.append("|---|---|---|---:|---:|")
    lines.append("| I_니라_O | 니라 | O | 3,150 | 2 |")
    lines.append("| II_니라_X | 니라 | X | 4,899 | 2 |")
    lines.append("| III_라_O | 라 | O | 1,984 | 2 |")
    lines.append("| IV_라_X | 라 | X | 6,348 | 12 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-category section
    for cat in CATS:
        lines.append(f"## {cat}")
        lines.append("")
        for q in QIDS:
            lines.append(f"### {QTITLES[q]}")
            lines.append("")
            ans = load_answer(cat, q)
            lines.append(ans)
            lines.append("")
        lines.append("---")
        lines.append("")

    text = "\n".join(lines)
    out_path = RES / "REPORT_v4.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, len(text)


if __name__ == "__main__":
    build_consolidated_json()
    out_path, n = build_markdown_report()
    print(f"REPORT_v4.md  ({n:,} chars)")
    print(f"  -> {out_path}")
    print(f"all_answers_v2.json saved")
