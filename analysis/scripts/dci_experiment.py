"""
DCI 실험: Direct Corpus Interaction vs LightRAG
================================================
순수 DCI: 임베딩 없음, 클러스터링 없음, 샘플링 없음.
전수 grep/awk 등가 패턴 매칭 → gpt-5-mini 합성.
비교 대상: lightrag_out/results/*.md (24건)
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# --- Config ---
TSV_PATH = Path(r"C:/Users/junto/Downloads/analysis_v8/parallel_data_v2.tsv")
OUT_DIR = Path(r"C:/Users/junto/Downloads/analysis_v8/dci_out/results")
CATS = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]

CAT_LABELS = {
    "I_니라_O":  "‘니라’, 행동\xb7태도 결정(O)",
    "II_니라_X": "‘니라’, 행동\xb7태도 결정 아님(X)",
    "III_라_O":  "‘라’, 행동\xb7태도 결정(O)",
    "IV_라_X":   "‘라’, 행동\xb7태도 결정 아님(X)",
}

QUESTIONS = {
    "Q1_themes": (
        "이 카테고리({cat}) 전체에 걸쳐 가장 두드러지는 "
        "주제\xb7내용 패턴 5가지를 근거 인용과 함께 요약하라."
    ),
    "Q2_subpatterns": (
        "이 카테고리({cat})에서 임베딩이나 클러스터링 없이, "
        "출전(book)\xb7문장구조\xb7어휘 분포만으로 식별 가능한 "
        "하위 패턴(subgroup)은 무엇인가? 근거와 함께 서술하라."
    ),
    "Q3_decision_features": (
        "{cat} 카테고리 종결문의 행동\xb7태도 결정 표지"
        "(예: 是, 非, 可, 不可, 故, 當, 必 등 한문 / "
        "옳다, 그르다, 마땅하다, ~해야 한다 등 번역)가 "
        "얼마나 자주, 어떤 패턴으로 나타나는지 분석하라. "
        "근거 문장을 인용하라."
    ),
    "Q4_logical_markers": (
        "{cat} 카테고리에 빈번한 인과\xb7결론\xb7조건 접속표지"
        "(故, 是以, 然이나, 若…則…, ~이니, ~이므로 등)"
        "와 그 사용 양상의 특징을 정리하라."
    ),
    "Q5_tense_mood": (
        "{cat} 카테고리의 시제\xb7서법 분포"
        "(당위/규범, 정의/현재, 과거사건, 인용중계 등) 중 "
        "어느 것이 가장 우세하며, 그것이 종결어미 선택과 "
        "어떤 관계가 있는가?"
    ),
    "Q6_sources": (
        "{cat} 카테고리에서 가장 많이 등장하는 출처 문헌(book) "
        "또는 인물\xb7개념을 5~10개 나열하고, 그 분포가 "
        "종결어미 선택과 의미하는 바를 해석하라."
    ),
}

SYSTEM_PROMPT = (
    "한문 고전 번역문의 언어학적 분석 전문가. "
    "아래 증거 데이터는 원시 코퍼스에서 grep/awk 등가 연산으로 전수 추출한 결과이다. "
    "증거에 기반하여 정밀하게 분석하라. 증거에 없는 내용은 추론하지 마라. "
    "통계 수치는 증거의 숫자를 그대로 인용하라."
)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def load_tsv():
    data = defaultdict(list)
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cat = row.get("cell", "").strip()
            if cat in CATS:
                data[cat].append(row)
    return data


# ──────────────────────────────────────────────
# Evidence Extractors (grep/awk equivalents)
# ──────────────────────────────────────────────

def _book_dist(rows):
    books = Counter(r["book"] for r in rows)
    return books, "\n".join(
        f"  {c:>4}건  {b}" for b, c in books.most_common()
    )


def _dansa_dist(rows):
    d = Counter(r.get("dansa_category", "") for r in rows)
    return "\n".join(f"  {c:>4}건  {cat}" for cat, c in d.most_common())


def _pick_examples(rows, n=150):
    """출전별 균등 간격 추출 (전수 접근 보장, 대표성 확보)."""
    books = Counter(r["book"] for r in rows)
    out = []
    for book, _ in books.most_common(15):
        br = [r for r in rows if r["book"] == book]
        step = max(1, len(br) // (n // 15 + 1))
        for i in range(0, len(br), step):
            if len(out) >= n:
                return out
            r = br[i]
            out.append(
                f"[{book}] 원문: {r.get('원문','')[:200]}\n"
                f"  번역: {r.get('번역문','')[:200]}"
            )
    return out


def extract_q1(rows, cat):
    books, book_txt = _book_dist(rows)
    dansa_txt = _dansa_dist(rows)

    # 번역문 고빈도 핵심어 (전수 스캔)
    wf = Counter()
    stop = {"것이다","하는","있는","이것","그것","것은","것이","있다","없다",
            "하여","하고","하니","이다","있으니","없으니","것을","하면"}
    for r in rows:
        wf.update(w for w in re.findall(r"[가-힣]{2,}", r.get("번역문","")) if w not in stop)
    top50 = "\n".join(f"  {c:>4}회  {w}" for w, c in wf.most_common(50))

    exs = _pick_examples(rows, 150)
    return (
        f"### 전체 통계: {len(rows)}건\n\n"
        f"### 출전 분포 (전수)\n{book_txt}\n\n"
        f"### 단사범주 분포\n{dansa_txt}\n\n"
        f"### 번역문 고빈도 핵심어 (전수, 상위 50)\n{top50}\n\n"
        f"### 대표 문장 ({len(exs)}건)\n" + "\n".join(exs)
    )


def extract_q2(rows, cat):
    _, book_txt = _book_dist(rows)
    dansa_txt = _dansa_dist(rows)

    # 원문 길이 분포
    buckets = Counter()
    for r in rows:
        L = len(r.get("원문", ""))
        if   L < 30:  buckets["~29자"] += 1
        elif L < 60:  buckets["30~59자"] += 1
        elif L < 100: buckets["60~99자"] += 1
        elif L < 150: buckets["100~149자"] += 1
        else:         buckets["150자+"] += 1
    len_txt = "\n".join(f"  {c:>4}건  {b}" for b, c in sorted(buckets.items()))

    # 문미 패턴 (원문 마지막 5자)
    ends = Counter()
    for r in rows:
        s = r.get("원문", "").strip()
        if len(s) >= 5:
            ends[s[-5:]] += 1
    end_txt = "\n".join(f"  {c:>4}건  …{e}" for e, c in ends.most_common(30))

    # 핵심 개념어 (전수)
    terms = ["君子","小人","天","道","德","禮","仁","義","聖","王",
             "臣","民","孝","忠","信","學","心","性","命","理","氣"]
    cooc = {t: sum(1 for r in rows if t in r.get("원문","")) for t in terms}
    cooc = {t: c for t, c in cooc.items() if c > 0}
    cooc_txt = "\n".join(f"  {c:>4}건  {t}" for t, c in sorted(cooc.items(), key=lambda x: -x[1]))

    return (
        f"### 전체: {len(rows)}건\n\n"
        f"### 출전 분포\n{book_txt}\n\n"
        f"### 단사범주 분포\n{dansa_txt}\n\n"
        f"### 원문 길이 분포\n{len_txt}\n\n"
        f"### 원문 문미 패턴 (상위 30)\n{end_txt}\n\n"
        f"### 한문 핵심 개념어 출현 (전수)\n{cooc_txt}"
    )


def _grep_marker(rows, markers_dict, ex_per_marker=20):
    """한문 표지별 전수 grep + 대표 예문."""
    parts = []
    for marker, meaning in markers_dict.items():
        hits = [(r.get("원문",""), r.get("번역문",""), r.get("book",""))
                for r in rows if marker in r.get("원문","")]
        if not hits:
            continue
        step = max(1, len(hits) // ex_per_marker)
        exs = []
        for i in range(0, len(hits), step):
            if len(exs) >= ex_per_marker:
                break
            o, t, b = hits[i]
            exs.append(f"  [{b}] {o[:150]} → {t[:150]}")
        parts.append(
            f"#### {marker} ({meaning}): {len(hits)}건 "
            f"({len(hits)/len(rows)*100:.1f}%)\n" + "\n".join(exs)
        )
    return "\n\n".join(parts)


def extract_q3(rows, cat):
    hanja = {
        "是":"옳다/이다", "非":"아니다/그르다",
        "可":"가하다", "不可":"불가하다",
        "當":"마땅하다", "必":"반드시",
        "宜":"마땅히", "故":"그러므로",
        "不得":"~할 수 없다", "莫":"~하지 말라",
    }
    hanja_txt = _grep_marker(rows, hanja)

    kr = {"마땅":0,"해야":0,"옳":0,"그르":0,"불가":0,"반드시":0,"가하":0}
    for r in rows:
        t = r.get("번역문","")
        for k in kr:
            if k in t:
                kr[k] += 1
    kr_txt = "\n".join(f"  {c:>4}건  '{k}'" for k, c in sorted(kr.items(), key=lambda x:-x[1]) if c)

    return (
        f"### 전체 {len(rows)}건 전수 검색\n\n"
        f"### 한문 결정 표지\n{hanja_txt}\n\n"
        f"### 번역문 한국어 결정 표지\n{kr_txt}"
    )


def extract_q4(rows, cat):
    markers = {
        "故":"그러므로", "是以":"이 때문에", "則":"~하면/곧",
        "若":"만약", "然":"그러하나", "蓋":"대개", "夫":"대저",
        "所以":"~하는 바", "雖":"비록", "以":"~로써",
    }
    m_txt = _grep_marker(rows, markers, 15)

    kr = {"그러므로":0,"이므로":0,"때문에":0,"까닭에":0,
          "그런즉":0,"그러하니":0,"만약":0,"비록":0}
    for r in rows:
        t = r.get("번역문","")
        for k in kr:
            if k in t:
                kr[k] += 1
    kr_txt = "\n".join(f"  {c:>4}건  '{k}'" for k, c in sorted(kr.items(), key=lambda x:-x[1]) if c)

    combos = Counter()
    for r in rows:
        s = r.get("원문","")
        if "故" in s and "則" in s: combos["故…則"] += 1
        if "若" in s and "則" in s: combos["若…則"] += 1
        if "雖" in s and "然" in s: combos["雖…然"] += 1
        if "蓋" in s and "故" in s: combos["蓋…故"] += 1
    combo_txt = "\n".join(f"  {c:>4}건  {p}" for p, c in combos.most_common()) or "  (없음)"

    return (
        f"### 전체 {len(rows)}건 전수 검색\n\n"
        f"### 한문 논리 접속표지\n{m_txt}\n\n"
        f"### 번역문 한국어 논리 접속표지\n{kr_txt}\n\n"
        f"### 복합 패턴 (공기)\n{combo_txt}"
    )


def extract_q5(rows, cat):
    cats_mood = {
        "당위/규범":  ["해야","마땅","반드시","옳","불가","~할 것이"],
        "정의/현재":  ["것이다","이다","이니","인 것이다"],
        "과거사건":   ["했다","하였","었다","였다","했으니"],
        "인용중계":   ["라 했","라 하였","고 하였","고 했"],
        "가능/허용":  ["할 수 있","가하","가능"],
        "부정/금지":  ["못하","않는","아니","없다","말라"],
    }
    mood_counts = {}
    mood_exs = {}
    for mood, pats in cats_mood.items():
        hits = []
        for r in rows:
            t = r.get("번역문","")
            if any(p in t for p in pats):
                hits.append(r)
        mood_counts[mood] = len(hits)
        mood_exs[mood] = [
            f"  [{h.get('book','')}] {h.get('번역문','')[:150]}"
            for h in hits[:10]
        ]

    dist = "\n".join(
        f"  {c:>4}건 ({c/len(rows)*100:.1f}%)  {m}"
        for m, c in sorted(mood_counts.items(), key=lambda x:-x[1])
    )

    ex_txt = ""
    for m, exs in mood_exs.items():
        if exs:
            ex_txt += f"\n#### {m} ({mood_counts[m]}건 중 {len(exs)}건)\n" + "\n".join(exs) + "\n"

    ends = Counter()
    for r in rows:
        t = r.get("번역문","").strip()
        if len(t) >= 3:
            ends[t[-3:]] += 1
    end_txt = "\n".join(f"  {c:>4}건  …{e}" for e, c in ends.most_common(20))

    return (
        f"### 전체 {len(rows)}건 전수 스캔\n\n"
        f"### 시제·서법 분포\n{dist}\n\n"
        f"### 번역문 문미 패턴 (상위 20)\n{end_txt}\n"
        f"{ex_txt}"
    )


def extract_q6(rows, cat):
    books, book_txt = _book_dist(rows)

    # 상위 10 출전 대표 문장
    ex_txt = ""
    for book, cnt in books.most_common(10):
        br = [r for r in rows if r["book"] == book]
        ex_txt += f"\n#### {book} ({cnt}건)\n"
        step = max(1, len(br) // 5)
        for i in range(0, min(len(br), step * 5), step):
            r = br[i]
            ex_txt += f"  원문: {r.get('원문','')[:120]}\n  번역: {r.get('번역문','')[:120]}\n"

    # 인물 (전수)
    figs = ["孔子","孟子","朱子","程子","周公","堯","舜","禹","湯",
            "文王","武王","伯夷","叔齊","顏子","曾子","子思","荀子",
            "老子","莊子","韓愈","司馬","董仲舒","王安石"]
    fc = {}
    for f in figs:
        c = sum(1 for r in rows if f in r.get("원문","") or f in r.get("번역문",""))
        if c:
            fc[f] = c
    fig_txt = "\n".join(f"  {c:>4}건  {f}" for f, c in sorted(fc.items(), key=lambda x:-x[1]))

    return (
        f"### 전체 {len(rows)}건 전수 집계\n\n"
        f"### 출전(book) 분포\n{book_txt}\n\n"
        f"### 주요 인물 출현 (전수)\n{fig_txt}\n"
        f"{ex_txt}"
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
EXTRACTORS = {
    "Q1_themes":            extract_q1,
    "Q2_subpatterns":       extract_q2,
    "Q3_decision_features": extract_q3,
    "Q4_logical_markers":   extract_q4,
    "Q5_tense_mood":        extract_q5,
    "Q6_sources":           extract_q6,
}


def run():
    client = OpenAI()
    print(f"[DCI] Loading {TSV_PATH} ...")
    data = load_tsv()
    for c in CATS:
        print(f"  {c}: {len(data[c])}건")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    total_tokens = 0

    for cat in CATS:
        rows = data[cat]
        label = CAT_LABELS[cat]

        for qid, qtpl in QUESTIONS.items():
            q = qtpl.format(cat=f"{cat} ({label})")
            tag = f"{cat} / {qid}"
            print(f"\n{'='*50}\n[DCI] {tag}\n{'='*50}")

            t0 = time.time()
            evidence = EXTRACTORS[qid](rows, cat)
            t_ext = time.time() - t0
            print(f"  evidence: {len(evidence)} chars  ({t_ext:.1f}s)")

            MAX_EV = 300_000
            truncated = False
            if len(evidence) > MAX_EV:
                evidence = evidence[:MAX_EV] + f"\n\n[증거 절단: {len(evidence)}자 중 {MAX_EV}자]"
                truncated = True
                print(f"  ⚠ truncated to {MAX_EV}")

            t1 = time.time()
            try:
                resp = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"[분석 대상: {cat} ({label})]\n\n"
                            f"## 질문\n{q}\n\n"
                            f"## DCI 추출 증거 (전수 검색)\n{evidence}"
                        )},
                    ],
                    # gpt-5-mini: temperature not supported
                )
                answer = resp.choices[0].message.content
                u = resp.usage
                pt, ct, tt = u.prompt_tokens, u.completion_tokens, u.total_tokens
                total_tokens += tt
            except Exception as exc:
                answer = f"[API ERROR] {exc}"
                pt = ct = tt = 0
            t_api = time.time() - t1
            print(f"  api: {t_api:.1f}s  tokens: {pt}+{ct}={tt}")

            md = (
                f"# {cat} — {qid} (DCI)\n\n"
                f"## Method\n"
                f"Direct Corpus Interaction (DCI) — "
                f"임베딩\xb7클러스터링\xb7인덱싱 없이 원시 코퍼스 전수 검색.\n"
                f"Evidence: grep/awk equivalents on {len(rows)} rows.\n\n"
                f"## Question\n[분석 대상: {cat} ({label})]\n\n{q}\n\n"
                f"## Answer\n{answer}\n\n"
                f"## DCI Metadata\n"
                f"- extraction_time: {t_ext:.1f}s\n"
                f"- api_time: {t_api:.1f}s\n"
                f"- total_time: {t_ext + t_api:.1f}s\n"
                f"- prompt_tokens: {pt}\n"
                f"- completion_tokens: {ct}\n"
                f"- total_tokens: {tt}\n"
                f"- evidence_chars: {len(evidence)}\n"
                f"- truncated: {truncated}\n"
                f"- corpus_rows: {len(rows)}\n"
                f"- model: gpt-5-mini\n"
                f"- timestamp: {datetime.now().isoformat()}\n"
            )
            out = OUT_DIR / f"{cat}__{qid}.md"
            out.write_text(md, encoding="utf-8")
            print(f"  → {out.name}")

            log.append({
                "cat": cat, "qid": qid,
                "extract_s": round(t_ext, 1),
                "api_s": round(t_api, 1),
                "total_s": round(t_ext + t_api, 1),
                "prompt_tokens": pt, "completion_tokens": ct,
                "total_tokens": tt, "evidence_chars": len(evidence),
                "truncated": truncated,
            })

    summary = OUT_DIR.parent / "dci_experiment_log.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "DCI vs LightRAG",
            "timestamp": datetime.now().isoformat(),
            "total_tokens": total_tokens,
            "categories": {c: len(data[c]) for c in CATS},
            "results": log,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"DONE  total_tokens={total_tokens}")
    print(f"results: {OUT_DIR}")
    print(f"log:     {summary}")


if __name__ == "__main__":
    run()
