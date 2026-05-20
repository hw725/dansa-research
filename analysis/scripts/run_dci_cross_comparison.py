"""
DCI 횡단 비교 (Cross-Category Comparison)
==========================================
LightRAG가 범주별 KG 격리로 폐기한 CQ1~CQ6를
단일 TSV 동시 접근으로 실행.

4범주 동시 grep → 비교 테이블 → gpt-5-mini 합성.
"""
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
TSV_PATH = ROOT / "parallel_data_v2_cleaned.tsv"
OUT_DIR = ROOT / "dci_out" / "cross"
CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]

SYSTEM_PROMPT = (
    "한문 고전 번역문의 언어학적 분석 전문가. "
    "아래 증거는 정제 코퍼스에서 4범주 동시 grep/awk 등가 연산으로 추출한 횡단 비교 데이터이다. "
    "증거에 기반하여 범주 간 차이를 정밀하게 분석하라. 증거에 없는 내용은 추론하지 마라. "
    "통계 수치는 증거의 숫자를 그대로 인용하라. "
    "핵심 가설: 종결어미 '니라'는 '라'보다 행동·태도 결정(O)과 더 강하게 결합하는가?"
)


def load_tsv():
    data = defaultdict(list)
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cat = row.get("cell", "").strip()
            if cat in CATS:
                data[cat].append(row)
    return data


def _marker_count(rows, markers):
    counts = {}
    for m in markers:
        counts[m] = sum(1 for r in rows if m in r.get("원문", ""))
    return counts


def _kr_marker_count(rows, markers):
    counts = {}
    for m in markers:
        counts[m] = sum(1 for r in rows if m in r.get("번역문", ""))
    return counts


def _top_books(rows, n=10):
    return Counter(r["book"] for r in rows).most_common(n)


def _pick_cross_examples(data, field, pattern, per_cat=5):
    """4범주에서 pattern 포함 행을 각 per_cat건씩 추출."""
    parts = []
    for cat in CATS:
        hits = [r for r in data[cat] if pattern in r.get(field, "")]
        step = max(1, len(hits) // per_cat) if hits else 1
        exs = []
        for i in range(0, len(hits), step):
            if len(exs) >= per_cat:
                break
            r = hits[i]
            exs.append(f"    [{r.get('book','')}] {r.get('원문','')[:120]} → {r.get('번역문','')[:120]}")
        parts.append(f"  {cat} ({len(hits)}건 중 {len(exs)}건):\n" + "\n".join(exs))
    return "\n".join(parts)


# ──────────────────────────────────────────────
# CQ Extractors
# ──────────────────────────────────────────────

def extract_cq1(data):
    """니라 vs 라 — 주제·출전 분포 차이."""
    nira = data["I_니라_O"] + data["II_니라_X"]
    ra = data["III_라_O"] + data["IV_라_X"]

    nira_books = _top_books(nira, 15)
    ra_books = _top_books(ra, 15)

    stop = {"것이다","하는","있는","이것","그것","것은","것이","있다","없다",
            "하여","하고","하니","이다","있으니","없으니","것을","하면"}
    def freq(rows):
        wf = Counter()
        for r in rows:
            wf.update(w for w in re.findall(r"[가-힣]{2,}", r.get("번역문","")) if w not in stop)
        return wf

    nira_wf = freq(nira).most_common(30)
    ra_wf = freq(ra).most_common(30)

    terms = ["君子","小人","天","道","德","禮","仁","義","聖","王",
             "臣","民","孝","忠","信","學","心","性","命","理","氣",
             "可","不可","故","當","必","是","非","宜"]
    nira_tc = {t: sum(1 for r in nira if t in r.get("원문","")) for t in terms}
    ra_tc = {t: sum(1 for r in ra if t in r.get("원문","")) for t in terms}

    table = "| 개념 | 니라 (건) | 니라 (%) | 라 (건) | 라 (%) | 차이(pp) |\n|---|---|---|---|---|---|\n"
    for t in terms:
        nc, rc = nira_tc.get(t, 0), ra_tc.get(t, 0)
        np, rp = nc / len(nira) * 100, rc / len(ra) * 100
        table += f"| {t} | {nc} | {np:.1f} | {rc} | {rp:.1f} | {np-rp:+.1f} |\n"

    book_table = "| 순위 | 니라 출전 (건) | 라 출전 (건) |\n|---|---|---|\n"
    for i in range(15):
        nb = nira_books[i] if i < len(nira_books) else ("—", 0)
        rb = ra_books[i] if i < len(ra_books) else ("—", 0)
        book_table += f"| {i+1} | {nb[0]} ({nb[1]}) | {rb[0]} ({rb[1]}) |\n"

    return (
        f"### 니라 그룹: {len(nira)}건 (I+II)  |  라 그룹: {len(ra)}건 (III+IV)\n\n"
        f"### 한문 핵심 개념어 횡단 비교\n{table}\n"
        f"### 출전 분포 횡단 비교\n{book_table}\n"
        f"### 니라 고빈도 핵심어 (번역문, 상위 30)\n"
        + "\n".join(f"  {c:>4}회  {w}" for w, c in nira_wf) + "\n\n"
        f"### 라 고빈도 핵심어 (번역문, 상위 30)\n"
        + "\n".join(f"  {c:>4}회  {w}" for w, c in ra_wf)
    )


def extract_cq2(data):
    """O vs X — 결정 표지 빈도 차이."""
    o_rows = data["I_니라_O"] + data["III_라_O"]
    x_rows = data["II_니라_X"] + data["IV_라_X"]

    hanja = ["是","非","可","不可","當","必","宜","故","不得","莫"]
    kr = ["마땅","해야","옳","그르","불가","반드시","가하"]

    o_h = _marker_count(o_rows, hanja)
    x_h = _marker_count(x_rows, hanja)
    o_k = _kr_marker_count(o_rows, kr)
    x_k = _kr_marker_count(x_rows, kr)

    table = "| 표지 | O(건) | O(%) | X(건) | X(%) | 차이(pp) |\n|---|---|---|---|---|---|\n"
    for m in hanja:
        oc, xc = o_h[m], x_h[m]
        op, xp = oc / len(o_rows) * 100, xc / len(x_rows) * 100
        table += f"| {m} | {oc} | {op:.1f} | {xc} | {xp:.1f} | {op-xp:+.1f} |\n"

    kr_table = "| 표지 | O(건) | O(%) | X(건) | X(%) | 차이(pp) |\n|---|---|---|---|---|---|\n"
    for m in kr:
        oc, xc = o_k[m], x_k[m]
        op, xp = oc / len(o_rows) * 100, xc / len(x_rows) * 100
        kr_table += f"| {m} | {oc} | {op:.1f} | {xc} | {xp:.1f} | {op-xp:+.1f} |\n"

    return (
        f"### O 그룹: {len(o_rows)}건 (I+III)  |  X 그룹: {len(x_rows)}건 (II+IV)\n\n"
        f"### 한문 결정 표지 횡단 비교\n{table}\n"
        f"### 한국어 결정 표지 횡단 비교\n{kr_table}"
    )


def extract_cq3(data):
    """종결어미 × 결정 여부 2×2 상호작용."""
    cells = {c: len(data[c]) for c in CATS}
    total = sum(cells.values())

    ct = "| | O (결정) | X (비결정) | 합계 |\n|---|---|---|---|\n"
    nira_o, nira_x = cells["I_니라_O"], cells["II_니라_X"]
    ra_o, ra_x = cells["III_라_O"], cells["IV_라_X"]
    ct += f"| 니라 | {nira_o} ({nira_o/total*100:.1f}%) | {nira_x} ({nira_x/total*100:.1f}%) | {nira_o+nira_x} |\n"
    ct += f"| 라 | {ra_o} ({ra_o/total*100:.1f}%) | {ra_x} ({ra_x/total*100:.1f}%) | {ra_o+ra_x} |\n"
    ct += f"| 합계 | {nira_o+ra_o} | {nira_x+ra_x} | {total} |\n"

    nira_o_rate = nira_o / (nira_o + nira_x) * 100
    ra_o_rate = ra_o / (ra_o + ra_x) * 100

    rates = (
        f"### 종결어미별 결정(O) 비율\n"
        f"- 니라: {nira_o}/{nira_o+nira_x} = {nira_o_rate:.1f}%\n"
        f"- 라: {ra_o}/{ra_o+ra_x} = {ra_o_rate:.1f}%\n"
        f"- 차이: {nira_o_rate - ra_o_rate:+.1f}pp\n"
    )

    hanja = ["故","當","必","可","不可","是","非","宜"]
    four_table = "| 표지 | I_니라_O | II_니라_X | III_라_O | IV_라_X |\n|---|---|---|---|---|\n"
    for m in hanja:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat] if m in r.get("원문", ""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        four_table += f"| {m} | {' | '.join(vals)} |\n"

    kr = ["마땅","반드시","해야","옳","불가"]
    kr_table = "| 표지 | I_니라_O | II_니라_X | III_라_O | IV_라_X |\n|---|---|---|---|---|\n"
    for m in kr:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat] if m in r.get("번역문", ""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        kr_table += f"| {m} | {' | '.join(vals)} |\n"

    examples = ""
    for m in ["故", "當", "必"]:
        examples += f"\n### '{m}' 4범주 예문\n"
        examples += _pick_cross_examples(data, "원문", m, per_cat=3) + "\n"

    return (
        f"### 2×2 분할표\n{ct}\n{rates}\n"
        f"### 4범주 한문 결정 표지 상세\n{four_table}\n"
        f"### 4범주 한국어 결정 표지 상세\n{kr_table}\n"
        f"{examples}"
    )


def extract_cq4(data):
    """접속 표지 횡단 비교."""
    markers = ["故","是以","則","若","然","蓋","夫","所以","雖","以"]
    kr_markers = ["그러므로","이므로","때문에","까닭에","만약","비록"]

    h_table = "| 표지 | I_니라_O | II_니라_X | III_라_O | IV_라_X |\n|---|---|---|---|---|\n"
    for m in markers:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat] if m in r.get("원문", ""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        h_table += f"| {m} | {' | '.join(vals)} |\n"

    kr_table = "| 표지 | I_니라_O | II_니라_X | III_라_O | IV_라_X |\n|---|---|---|---|---|\n"
    for m in kr_markers:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat] if m in r.get("번역문", ""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        kr_table += f"| {m} | {' | '.join(vals)} |\n"

    combos = [("故","則"), ("若","則"), ("雖","然"), ("蓋","故")]
    combo_table = "| 패턴 | I_니라_O | II_니라_X | III_라_O | IV_라_X |\n|---|---|---|---|---|\n"
    for a, b in combos:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat]
                    if a in r.get("원문","") and b in r.get("원문",""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        combo_table += f"| {a}…{b} | {' | '.join(vals)} |\n"

    return (
        f"### 한문 접속 표지 4범주 비교\n{h_table}\n"
        f"### 한국어 접속 표지 4범주 비교\n{kr_table}\n"
        f"### 복합 패턴 (공기) 4범주 비교\n{combo_table}"
    )


def extract_cq5(data):
    """출전별 종결어미 분포."""
    all_rows = []
    for cat in CATS:
        for r in data[cat]:
            all_rows.append({**r, "_cat": cat})

    book_counts = Counter(r["book"] for r in all_rows)
    top_books = [b for b, _ in book_counts.most_common(20)]

    table = "| 출전 | I_니라_O | II_니라_X | III_라_O | IV_라_X | 합계 | 니라율 | O율 |\n|---|---|---|---|---|---|---|---|\n"
    for book in top_books:
        vals = []
        cat_counts = {}
        for cat in CATS:
            c = sum(1 for r in data[cat] if r["book"] == book)
            cat_counts[cat] = c
            vals.append(str(c))
        total = sum(cat_counts.values())
        nira = cat_counts["I_니라_O"] + cat_counts["II_니라_X"]
        o = cat_counts["I_니라_O"] + cat_counts["III_라_O"]
        nira_rate = nira / total * 100 if total else 0
        o_rate = o / total * 100 if total else 0
        table += f"| {book} | {' | '.join(vals)} | {total} | {nira_rate:.0f}% | {o_rate:.0f}% |\n"

    nira_only = set(b for b, _ in _top_books(data["I_니라_O"] + data["II_니라_X"], 20))
    ra_only = set(b for b, _ in _top_books(data["III_라_O"] + data["IV_라_X"], 20))
    nira_exclusive = nira_only - ra_only
    ra_exclusive = ra_only - nira_only

    excl = f"### 니라 상위 20 전용 출전: {', '.join(nira_exclusive) if nira_exclusive else '없음'}\n"
    excl += f"### 라 상위 20 전용 출전: {', '.join(ra_exclusive) if ra_exclusive else '없음'}\n"

    return f"### 상위 20 출전 × 4범주 분포\n{table}\n{excl}"


def extract_cq6(data):
    """가설 직접 검증: 니라+O가 행동 결정과 더 강하게 결합하는가?"""
    cells = {c: len(data[c]) for c in CATS}
    total = sum(cells.values())

    ct = "| | O | X | 합 | O율 |\n|---|---|---|---|---|\n"
    nira_o, nira_x = cells["I_니라_O"], cells["II_니라_X"]
    ra_o, ra_x = cells["III_라_O"], cells["IV_라_X"]
    ct += f"| 니라 | {nira_o} | {nira_x} | {nira_o+nira_x} | {nira_o/(nira_o+nira_x)*100:.1f}% |\n"
    ct += f"| 라 | {ra_o} | {ra_x} | {ra_o+ra_x} | {ra_o/(ra_o+ra_x)*100:.1f}% |\n"

    from math import log as ln
    a, b, c, d = nira_o, nira_x, ra_o, ra_x
    n = a + b + c + d
    e_a = (a+b)*(a+c)/n
    e_b = (a+b)*(b+d)/n
    e_c = (c+d)*(a+c)/n
    e_d = (c+d)*(b+d)/n
    chi2 = sum((obs-exp)**2/exp for obs, exp in [(a,e_a),(b,e_b),(c,e_c),(d,e_d)])

    or_val = (a * d) / (b * c) if b * c else float('inf')
    rr_nira = a / (a + b)
    rr_ra = c / (c + d)
    rr = rr_nira / rr_ra if rr_ra else float('inf')

    stats = (
        f"### 통계 검정 (전수 — 표본이 아닌 모집단)\n"
        f"- χ²(1) = {chi2:.2f}\n"
        f"- 오즈비 (OR) = {or_val:.3f}\n"
        f"- 상대위험도 (RR) = {rr:.3f}\n"
        f"- 니라 O율: {rr_nira*100:.1f}%\n"
        f"- 라 O율: {rr_ra*100:.1f}%\n"
        f"- 차이: {(rr_nira-rr_ra)*100:+.1f}pp\n"
    )

    decision_markers = ["當","必","可","故","宜"]
    dm_table = "| 표지 | 니라+O | 니라+X | 라+O | 라+X |\n|---|---|---|---|---|\n"
    for m in decision_markers:
        vals = []
        for cat in CATS:
            c = sum(1 for r in data[cat] if m in r.get("원문",""))
            pct = c / len(data[cat]) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        dm_table += f"| {m} | {' | '.join(vals)} |\n"

    nira_o_exs = []
    step = max(1, len(data["I_니라_O"]) // 10)
    for i in range(0, len(data["I_니라_O"]), step):
        if len(nira_o_exs) >= 10:
            break
        r = data["I_니라_O"][i]
        nira_o_exs.append(f"  [{r.get('book','')}] {r.get('원문','')[:120]}\n    → {r.get('번역문','')[:120]}")

    ra_x_exs = []
    step = max(1, len(data["IV_라_X"]) // 10)
    for i in range(0, len(data["IV_라_X"]), step):
        if len(ra_x_exs) >= 10:
            break
        r = data["IV_라_X"][i]
        ra_x_exs.append(f"  [{r.get('book','')}] {r.get('원문','')[:120]}\n    → {r.get('번역문','')[:120]}")

    return (
        f"### 2×2 분할표\n{ct}\n{stats}\n"
        f"### 결정 표지 4범주 상세\n{dm_table}\n"
        f"### I_니라_O 대표 예문 (결정+니라)\n" + "\n".join(nira_o_exs) + "\n\n"
        f"### IV_라_X 대표 예문 (비결정+라)\n" + "\n".join(ra_x_exs)
    )


# ──────────────────────────────────────────────
# Questions
# ──────────────────────────────────────────────

CQ_SPEC = {
    "CQ1_nira_vs_ra": {
        "question": (
            "종결어미 '니라' 그룹(I+II)과 '라' 그룹(III+IV)의 주제·출전·핵심어 분포를 비교하라. "
            "두 그룹 간 가장 큰 차이점 3가지와 그 의미를 정리하라."
        ),
        "extractor": extract_cq1,
    },
    "CQ2_O_vs_X": {
        "question": (
            "행동·태도 결정(O) 그룹(I+III)과 비결정(X) 그룹(II+IV)에서 "
            "한문/한국어 결정 표지의 빈도를 비교하라. "
            "O 그룹에서 유의미하게 높은 표지와 X 그룹에서 높은 표지를 구분하고 해석하라."
        ),
        "extractor": extract_cq2,
    },
    "CQ3_interaction": {
        "question": (
            "종결어미(니라/라) × 결정 여부(O/X)의 2×2 상호작용을 분석하라. "
            "4범주 각각에서 결정 표지의 빈도 패턴이 어떻게 다른지, "
            "종결어미와 결정 여부가 독립적인지 상호작용하는지 판단하라."
        ),
        "extractor": extract_cq3,
    },
    "CQ4_logical_markers": {
        "question": (
            "인과·결론·조건 접속 표지(故, 則, 若, 蓋, 雖 등)의 4범주 분포를 비교하라. "
            "어떤 접속 패턴이 특정 종결어미나 결정 유형과 결합하는지 분석하라."
        ),
        "extractor": extract_cq4,
    },
    "CQ5_sources_cross": {
        "question": (
            "상위 20개 출전의 4범주 분포를 분석하라. "
            "특정 출전이 특정 종결어미나 결정 유형에 편중되는 패턴이 있는가? "
            "출전 편향이 가설 검증에 미치는 영향을 평가하라."
        ),
        "extractor": extract_cq5,
    },
    "CQ6_hypothesis_test": {
        "question": (
            "핵심 가설을 직접 검증하라: '종결어미 니라는 라보다 행동·태도 결정(O)과 더 강하게 결합하는가?' "
            "2×2 분할표, χ², 오즈비, 상대위험도를 해석하고, "
            "결정 표지(當·必·可·故·宜)의 4범주 분포가 이 가설을 지지하는지 평가하라. "
            "가설의 한계와 대안적 설명도 논의하라."
        ),
        "extractor": extract_cq6,
    },
}


def run():
    client = OpenAI()
    print(f"[DCI-CQ] Loading {TSV_PATH} ...")
    data = load_tsv()
    corpus_total = sum(len(data[c]) for c in CATS)
    for c in CATS:
        print(f"  {c}: {len(data[c])}건")
    print(f"  합계: {corpus_total}건")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    total_tokens = 0

    for cqid, spec in CQ_SPEC.items():
        q = spec["question"]
        extract = spec["extractor"]
        print(f"\n{'='*60}\n[DCI-CQ] {cqid}\n{'='*60}")

        t0 = time.time()
        evidence = extract(data)
        t_ext = time.time() - t0
        print(f"  evidence: {len(evidence)} chars  ({t_ext:.1f}s)")

        MAX_EV = 300_000
        truncated = False
        if len(evidence) > MAX_EV:
            evidence = evidence[:MAX_EV] + f"\n\n[증거 절단: {len(evidence)}자 중 {MAX_EV}자]"
            truncated = True

        t1 = time.time()
        try:
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"## 횡단 비교 질문\n{q}\n\n"
                        f"## DCI 추출 증거 (4범주 동시 전수 검색, {corpus_total:,}건)\n{evidence}"
                    )},
                ],
            )
            answer = resp.choices[0].message.content
            u = resp.usage
            pt, ct_tok, tt = u.prompt_tokens, u.completion_tokens, u.total_tokens
            total_tokens += tt
        except Exception as exc:
            answer = f"[API ERROR] {exc}"
            pt = ct_tok = tt = 0
        t_api = time.time() - t1
        print(f"  api: {t_api:.1f}s  tokens: {pt}+{ct_tok}={tt}")

        md = (
            f"# {cqid} (DCI 횡단 비교)\n\n"
            f"## Method\n"
            f"Direct Corpus Interaction — 4범주 동시 전수 검색.\n"
            f"LightRAG에서 불가능했던 횡단 비교를 단일 TSV 접근으로 실현.\n\n"
            f"## Question\n{q}\n\n"
            f"## Answer\n{answer}\n\n"
            f"## Metadata\n"
            f"- extraction_time: {t_ext:.1f}s\n"
            f"- api_time: {t_api:.1f}s\n"
            f"- total_time: {t_ext + t_api:.1f}s\n"
            f"- prompt_tokens: {pt}\n"
            f"- completion_tokens: {ct_tok}\n"
            f"- total_tokens: {tt}\n"
            f"- evidence_chars: {len(evidence)}\n"
            f"- truncated: {truncated}\n"
            f"- corpus_total: {corpus_total}\n"
            f"- model: gpt-5-mini\n"
            f"- timestamp: {datetime.now().isoformat()}\n"
        )
        out = OUT_DIR / f"{cqid}.md"
        out.write_text(md, encoding="utf-8")
        print(f"  → {out.name}")

        log.append({
            "cqid": cqid,
            "extract_s": round(t_ext, 1),
            "api_s": round(t_api, 1),
            "total_s": round(t_ext + t_api, 1),
            "prompt_tokens": pt, "completion_tokens": ct_tok,
            "total_tokens": tt, "evidence_chars": len(evidence),
            "truncated": truncated,
        })

    summary = OUT_DIR.parent / "dci_cross_log.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "DCI Cross-Category Comparison",
            "timestamp": datetime.now().isoformat(),
            "total_tokens": total_tokens,
            "categories": {c: len(data[c]) for c in CATS},
            "results": log,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE  total_tokens={total_tokens}")
    print(f"results: {OUT_DIR}")
    print(f"log:     {summary}")


if __name__ == "__main__":
    run()
