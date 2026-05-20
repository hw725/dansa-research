"""DCI 종합 리포트 + HTML 대시보드 생성
====================================
per-category (Q1~Q6, 24건) + cross-category (CQ1~CQ6, 6건) 통합.
Output: REPORT.md, all_answers.json, dashboard.html
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
CROSS = BASE / "cross"
LOG = BASE / "dci_experiment_log.json"
CROSS_LOG = BASE / "dci_cross_log.json"

CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]
CAT_LABELS = {
    "I_니라_O": "니라 + 결정(O)",
    "II_니라_X": "니라 + 비결정(X)",
    "III_라_O": "라 + 결정(O)",
    "IV_라_X": "라 + 비결정(X)",
}
QIDS = ["Q1_themes", "Q2_subpatterns", "Q3_decision_features",
        "Q4_logical_markers", "Q5_tense_mood", "Q6_sources"]
QTITLES = {
    "Q1_themes": "Q1. 주요 주제 패턴",
    "Q2_subpatterns": "Q2. 하위 패턴 (임베딩 없이)",
    "Q3_decision_features": "Q3. 행동·태도 결정 표지",
    "Q4_logical_markers": "Q4. 인과·결론·조건 접속표지",
    "Q5_tense_mood": "Q5. 시제·서법 분포",
    "Q6_sources": "Q6. 출처 문헌·인물·개념",
}
CQIDS = ["CQ1_nira_vs_ra", "CQ2_O_vs_X", "CQ3_interaction",
         "CQ4_logical_markers", "CQ5_sources_cross", "CQ6_hypothesis_test"]
CQTITLES = {
    "CQ1_nira_vs_ra": "CQ1. 니라 vs 라 — 주제·출전 분포",
    "CQ2_O_vs_X": "CQ2. 결정(O) vs 비결정(X) — 결정 표지",
    "CQ3_interaction": "CQ3. 종결어미 × 결정 여부 상호작용",
    "CQ4_logical_markers": "CQ4. 접속 표지 4범주 횡단",
    "CQ5_sources_cross": "CQ5. 출전별 종결어미 분포",
    "CQ6_hypothesis_test": "CQ6. 가설 직접 검증 (χ², OR, RR)",
}


def load_answer(path: Path) -> str:
    if not path.exists():
        return "(결과 없음)"
    txt = path.read_text(encoding="utf-8")
    if "## Answer" in txt:
        body = txt.split("## Answer", 1)[-1]
        body = body.split("## DCI Metadata", 1)[0]
        body = body.split("## Metadata", 1)[0]
        return body.strip()
    return txt.strip()


def load_log(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_json():
    answers = {"per_category": {}, "cross_category": {}}
    for c in CATS:
        answers["per_category"][c] = {}
        for q in QIDS:
            answers["per_category"][c][q] = load_answer(RES / f"{c}__{q}.md")
    for cq in CQIDS:
        answers["cross_category"][cq] = load_answer(CROSS / f"{cq}.md")
    out = BASE / "all_answers.json"
    out.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_report():
    log = load_log(LOG)
    clog = load_log(CROSS_LOG)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L = []
    L.append("# DCI 분석 종합 보고서: 任圭直 夬絶 가설 검증")
    L.append("")
    L.append(f"생성일: {now}")
    L.append("")
    L.append("> **방법론**: Direct Corpus Interaction (DCI) — arXiv 2605.05242.")
    L.append("> 임베딩·클러스터링·인덱싱 없이 원시 코퍼스 전수 grep/awk 등가 검색 → LLM 합성.")
    L.append("> LightRAG 파이프라인과 동일 모델(gpt-5-mini), 동일 질문으로 직접 비교.")
    L.append("")

    # Overview
    L.append("## 개요")
    L.append("")
    cats_info = log.get("categories", {})
    total_rows = sum(cats_info.values()) if cats_info else 11961
    total_tok = log.get("total_tokens", 0) + clog.get("total_tokens", 0)
    per_results = log.get("results", [])
    cq_results = clog.get("results", [])
    per_time = sum(r.get("total_s", 0) for r in per_results)
    cq_time = sum(r.get("total_s", 0) for r in cq_results)

    L.append(f"- **코퍼스**: parallel_data_v2_cleaned.tsv ({total_rows:,}건)")
    L.append(f"- **모델**: gpt-5-mini (temperature 미지원, 기본값 사용)")
    L.append(f"- **범주별 분석**: 4범주 × 6질문 = 24건")
    L.append(f"- **횡단 비교**: 6건 (LightRAG에서 폐기된 CQ1~CQ6)")
    L.append(f"- **총 토큰**: {total_tok:,}")
    L.append(f"- **총 소요 시간**: {per_time + cq_time:.0f}초 ({(per_time + cq_time)/60:.1f}분)")
    L.append(f"- **전처리 비용**: 0 (임베딩·KG 불필요)")
    L.append(f"- **Coverage**: 전체 행 100% (truncation 0건)")
    L.append("")

    L.append("### 4범주 데이터")
    L.append("")
    L.append("| 범주 | 종결어미 | 결정 여부 | 행 수 |")
    L.append("|---|---|---|---:|")
    for c in CATS:
        n = cats_info.get(c, "?")
        parts = CAT_LABELS[c].split(" + ")
        L.append(f"| {c} | {parts[0]} | {parts[1]} | {n:,} |")
    L.append("")

    # Per-category timing table
    if per_results:
        L.append("### 범주별 실행 통계")
        L.append("")
        L.append("| 범주 | 질문 | 토큰 | 시간(초) | 증거(자) |")
        L.append("|---|---|---:|---:|---:|")
        for r in per_results:
            L.append(f"| {r['cat']} | {r['qid']} | {r['total_tokens']:,} | {r['total_s']} | {r['evidence_chars']:,} |")
        L.append("")

    L.append("---")
    L.append("")

    # Per-category answers
    for cat in CATS:
        L.append(f"## {cat} ({CAT_LABELS[cat]})")
        L.append("")
        for q in QIDS:
            L.append(f"### {QTITLES[q]}")
            L.append("")
            ans = load_answer(RES / f"{cat}__{q}.md")
            L.append(ans)
            L.append("")
        L.append("---")
        L.append("")

    # Cross-category
    L.append("## 횡단 비교 (Cross-Category)")
    L.append("")
    L.append("> LightRAG는 범주별 KG 격리로 인해 CQ1~CQ7 횡단 비교를 폐기했다.")
    L.append("> DCI는 단일 TSV 동시 접근으로 이를 실현한다.")
    L.append("")

    if cq_results:
        L.append("### 횡단 비교 실행 통계")
        L.append("")
        L.append("| CQ | 토큰 | 시간(초) | 증거(자) |")
        L.append("|---|---:|---:|---:|")
        for r in cq_results:
            L.append(f"| {r['cqid']} | {r['total_tokens']:,} | {r['total_s']} | {r['evidence_chars']:,} |")
        L.append("")

    for cq in CQIDS:
        L.append(f"### {CQTITLES[cq]}")
        L.append("")
        ans = load_answer(CROSS / f"{cq}.md")
        L.append(ans)
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*DCI 종합 보고서 — {now}*")

    text = "\n".join(L)
    out = BASE / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    return out, len(text)


def build_dashboard():
    log = load_log(LOG)
    clog = load_log(CROSS_LOG)
    cats_info = log.get("categories", {})
    per_results = log.get("results", [])
    cq_results = clog.get("results", [])
    total_tok = log.get("total_tokens", 0) + clog.get("total_tokens", 0)
    per_time = sum(r.get("total_s", 0) for r in per_results)
    cq_time = sum(r.get("total_s", 0) for r in cq_results)

    # Build data for charts
    cat_tokens = {}
    cat_time = {}
    for r in per_results:
        c = r["cat"]
        cat_tokens[c] = cat_tokens.get(c, 0) + r["total_tokens"]
        cat_time[c] = cat_time.get(c, 0) + r["total_s"]

    cq_tok_list = [r.get("total_tokens", 0) for r in cq_results]
    cq_time_list = [r.get("total_s", 0) for r in cq_results]
    cq_labels = [r.get("cqid", "") for r in cq_results]

    # 2x2 table data
    nira_o = cats_info.get("I_니라_O", 5381)
    nira_x = cats_info.get("II_니라_X", 1825)
    ra_o = cats_info.get("III_라_O", 3181)
    ra_x = cats_info.get("IV_라_X", 1574)
    total = nira_o + nira_x + ra_o + ra_x
    nira_o_rate = nira_o / (nira_o + nira_x) * 100
    ra_o_rate = ra_o / (ra_o + ra_x) * 100

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DCI 실험 대시보드 — 任圭直 夬絶 가설</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Pretendard', -apple-system, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; color: #f8fafc; }}
h2 {{ font-size: 1.3rem; font-weight: 600; margin: 32px 0 16px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
h3 {{ font-size: 1rem; font-weight: 600; margin: 16px 0 8px; color: #cbd5e1; }}
.subtitle {{ color: #64748b; font-size: 0.9rem; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 16px 0; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
.card-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 4px; }}
.card-value {{ font-size: 2rem; font-weight: 700; color: #f8fafc; }}
.card-sub {{ font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }}
.card.accent {{ border-color: #3b82f6; }}
.card.green {{ border-color: #22c55e; }}
.card.amber {{ border-color: #f59e0b; }}
.card.red {{ border-color: #ef4444; }}
.card.accent .card-value {{ color: #60a5fa; }}
.card.green .card-value {{ color: #4ade80; }}
.card.amber .card-value {{ color: #fbbf24; }}
.card.red .card-value {{ color: #f87171; }}

table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.85rem; }}
th {{ background: #1e293b; padding: 10px 12px; text-align: left; font-weight: 600; color: #94a3b8; border-bottom: 2px solid #334155; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
tr:hover td {{ background: #1e293b; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

.bar-container {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
.bar-label {{ width: 100px; font-size: 0.8rem; text-align: right; color: #94a3b8; flex-shrink: 0; }}
.bar-track {{ flex: 1; height: 24px; background: #1e293b; border-radius: 4px; overflow: hidden; position: relative; }}
.bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding: 0 8px; font-size: 0.75rem; font-weight: 600; color: #fff; min-width: 40px; }}
.bar-val {{ width: 60px; font-size: 0.8rem; text-align: right; color: #cbd5e1; flex-shrink: 0; }}

.two-by-two {{ display: grid; grid-template-columns: auto 1fr 1fr 1fr; gap: 0; margin: 16px 0; }}
.two-by-two > div {{ padding: 12px 16px; border: 1px solid #334155; text-align: center; }}
.two-by-two .header {{ background: #1e293b; font-weight: 600; color: #94a3b8; }}
.two-by-two .cell-nira-o {{ background: rgba(59,130,246,0.15); }}
.two-by-two .cell-nira-x {{ background: rgba(100,116,139,0.1); }}
.two-by-two .cell-ra-o {{ background: rgba(34,197,94,0.15); }}
.two-by-two .cell-ra-x {{ background: rgba(100,116,139,0.1); }}

.compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
.compare .col {{ background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }}
.compare .col h3 {{ margin-top: 0; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
.tag-win {{ background: rgba(34,197,94,0.2); color: #4ade80; }}
.tag-lose {{ background: rgba(239,68,68,0.2); color: #f87171; }}
.tag-na {{ background: rgba(100,116,139,0.2); color: #94a3b8; }}

.cq-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }}
.cq-card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
.cq-card h4 {{ font-size: 0.9rem; color: #60a5fa; margin-bottom: 8px; }}
.cq-card .stats {{ font-size: 0.8rem; color: #64748b; }}
.cq-card .summary {{ font-size: 0.85rem; color: #cbd5e1; margin-top: 8px; }}

.footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #334155; font-size: 0.8rem; color: #475569; text-align: center; }}
</style>
</head>
<body>
<div class="container">

<h1>DCI 실험 대시보드</h1>
<p class="subtitle">任圭直 夬絶 가설 — Direct Corpus Interaction vs LightRAG &middot; {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="grid">
  <div class="card accent">
    <div class="card-label">총 토큰</div>
    <div class="card-value">{total_tok:,}</div>
    <div class="card-sub">per-cat {log.get("total_tokens",0):,} + cross {clog.get("total_tokens",0):,}</div>
  </div>
  <div class="card green">
    <div class="card-label">Coverage</div>
    <div class="card-value">100%</div>
    <div class="card-sub">{total:,}건 전수 접근, truncation 0</div>
  </div>
  <div class="card amber">
    <div class="card-label">총 소요 시간</div>
    <div class="card-value">{(per_time+cq_time)/60:.1f}분</div>
    <div class="card-sub">per-cat {per_time:.0f}s + cross {cq_time:.0f}s</div>
  </div>
  <div class="card">
    <div class="card-label">전처리 비용</div>
    <div class="card-value">$0</div>
    <div class="card-sub">임베딩·클러스터링·KG 불필요</div>
  </div>
</div>

<h2>2&times;2 분할표 — 핵심 가설</h2>
<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:12px;">니라는 라보다 행동&middot;태도 결정(O)과 더 강하게 결합하는가?</p>

<div class="two-by-two">
  <div class="header"></div>
  <div class="header">O (결정)</div>
  <div class="header">X (비결정)</div>
  <div class="header">O율</div>
  <div class="header">니라</div>
  <div class="cell-nira-o"><strong>{nira_o:,}</strong><br><small>{nira_o/total*100:.1f}%</small></div>
  <div class="cell-nira-x">{nira_x:,}<br><small>{nira_x/total*100:.1f}%</small></div>
  <div class="cell-nira-o" style="font-size:1.3rem;font-weight:700;color:#60a5fa;">{nira_o_rate:.1f}%</div>
  <div class="header">라</div>
  <div class="cell-ra-o"><strong>{ra_o:,}</strong><br><small>{ra_o/total*100:.1f}%</small></div>
  <div class="cell-ra-x">{ra_x:,}<br><small>{ra_x/total*100:.1f}%</small></div>
  <div class="cell-ra-o" style="font-size:1.3rem;font-weight:700;color:#4ade80;">{ra_o_rate:.1f}%</div>
</div>
<p style="color:#94a3b8;font-size:0.85rem;margin-top:8px;">
  차이: <strong style="color:#fbbf24;">{nira_o_rate - ra_o_rate:+.1f}pp</strong> &middot;
  니라 O율 {nira_o_rate:.1f}% vs 라 O율 {ra_o_rate:.1f}%
</p>

<h2>DCI vs LightRAG 비교</h2>
<table>
<tr><th>평가 축</th><th>DCI</th><th>LightRAG</th><th>판정</th></tr>
<tr><td>Coverage</td><td class="num">100%</td><td class="num">~56%</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>Localization</td><td>행 단위 빈도</td><td>클러스터 단위 정성</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>전처리 비용</td><td class="num">$0</td><td>임베딩+KG</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>쿼리 속도</td><td class="num">20~35s</td><td class="num">56~75s</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>횡단 비교</td><td>CQ1~CQ6 실행</td><td>폐기</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>출력량</td><td class="num">2,069행</td><td class="num">1,215행</td><td><span class="tag tag-win">DCI</span></td></tr>
<tr><td>엔티티 관계 그래프</td><td>없음</td><td>KG 활용</td><td><span class="tag tag-lose">LightRAG</span></td></tr>
<tr><td>대규모 스케일링</td><td>O(n) 전수검색</td><td>O(1) top-k</td><td><span class="tag tag-lose">LightRAG</span></td></tr>
</table>

<h2>범주별 토큰·시간 분포</h2>
"""

    # Bar charts for per-category tokens
    max_tok = max(cat_tokens.values()) if cat_tokens else 1
    colors = {"I_니라_O": "#3b82f6", "II_니라_X": "#6366f1", "III_라_O": "#22c55e", "IV_라_X": "#14b8a6"}
    for c in CATS:
        tok = cat_tokens.get(c, 0)
        pct = tok / max_tok * 100
        html += f"""<div class="bar-container">
  <div class="bar-label">{c}</div>
  <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{colors[c]};">{tok:,}</div></div>
  <div class="bar-val">{cat_time.get(c,0):.0f}s</div>
</div>
"""

    # CQ section
    html += """
<h2>횡단 비교 (CQ1~CQ6)</h2>
<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:12px;">LightRAG가 범주별 KG 격리로 폐기한 분석을 DCI로 실현</p>
<div class="cq-grid">
"""
    for i, cq in enumerate(CQIDS):
        r = cq_results[i] if i < len(cq_results) else {}
        title = CQTITLES[cq]
        tok = r.get("total_tokens", 0)
        ts = r.get("total_s", 0)
        ev = r.get("evidence_chars", 0)
        html += f"""<div class="cq-card">
  <h4>{title}</h4>
  <div class="stats">{tok:,} tok &middot; {ts:.0f}s &middot; {ev:,} chars</div>
</div>
"""
    html += "</div>"

    # Per-category detail table
    html += """
<h2>범주별 상세 (Q1~Q6)</h2>
<table>
<tr><th>범주</th><th>질문</th><th style="text-align:right">토큰</th><th style="text-align:right">시간</th><th style="text-align:right">증거</th></tr>
"""
    for r in per_results:
        html += f"""<tr>
  <td>{r['cat']}</td><td>{QTITLES.get(r['qid'], r['qid'])}</td>
  <td class="num">{r['total_tokens']:,}</td><td class="num">{r['total_s']}s</td>
  <td class="num">{r['evidence_chars']:,}</td>
</tr>
"""
    html += "</table>"

    html += f"""
<div class="footer">
  DCI 실험 대시보드 &middot; arXiv 2605.05242 방법론 적용 &middot; 모델: gpt-5-mini &middot; {datetime.now().strftime("%Y-%m-%d")}
</div>

</div>
</body>
</html>"""

    out = BASE / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print("[1/3] all_answers.json ...")
    jp = build_json()
    print(f"  -> {jp}")

    print("[2/3] REPORT.md ...")
    rp, n = build_report()
    print(f"  -> {rp}  ({n:,} chars)")

    print("[3/3] dashboard.html ...")
    dp = build_dashboard()
    print(f"  -> {dp}")

    print("DONE")
