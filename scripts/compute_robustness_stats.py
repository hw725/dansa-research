#!/usr/bin/env python3
"""3-model 판정의 강건성·일치도 통계 산출.

기존 final 통계(compute_final_stats.py)가 보고하는 점추정(consensus 분포, chi2, V)을
보완하는 4종 분석을 추가한다. 새 LLM 호출 없이 기존 판정 CSV만 사용한다.

1. 모델 간 일치도 — Fleiss' kappa, pairwise Cohen's kappa, 일치율
2. 효과크기 구간추정 — O율 차이 Newcombe 95% CI, OR Woolf 95% CI,
   Cramér's V 부트스트랩 95% CI
3. 합의 정의 민감도 — O 정의를 만장일치(3표)/과반(2표 이상)/1표 이상으로
   바꿔도 효과 방향이 유지되는지
4. 서종 층화 — book 층화 Mantel-Haenszel 공통 OR(+RBG 95% CI),
   Woolf 동질성 검정, book별 sign test, 부(경부 사서/오경, 집부) 층화,
   모델별 층화 MH OR

표준 라이브러리만 사용한다. 판정 CSV의 book·marker_type·llm_judgment만 쓰므로
익명 판정(`results/{model}/*_anon.csv`)으로도 동일하게 재현된다.

Outputs:
- results/robustness_stats.json
- results/ROBUSTNESS_REPORT.md
- logs/robustness_stats.jsonl (실행 기록 append)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from collections import Counter, OrderedDict
from pathlib import Path

import compute_final_stats as cfs

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
LOGS = REPO / "logs"

Z95 = 1.959963984540054
DEFINITIONS = OrderedDict(
    [
        ("unanimous", "만장일치(3표) O — 현행 기준"),
        ("majority", "과반(2표 이상) O"),
        ("any", "1표 이상 O"),
    ]
)


# ---------------------------------------------------------------------------
# 분포 함수 (표준 라이브러리 구현)
# ---------------------------------------------------------------------------

def _gser(a: float, x: float, itmax: int = 500, eps: float = 3e-12) -> float:
    gln = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delt = summ
    for _ in range(itmax):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * eps:
            break
    return summ * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float, itmax: int = 500, eps: float = 3e-12) -> float:
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b if b != 0 else 1e300
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    """자유도 df 카이제곱 분포의 생존함수 P(X >= x)."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    if xx < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gser(a, xx)))
    return max(0.0, min(1.0, _gcf(a, xx)))


# ---------------------------------------------------------------------------
# 구간추정·검정 도구
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = Z95 * Z95
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (center - margin) / denom, (center + margin) / denom


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """비율차 p1-p2의 Newcombe hybrid score 95% CI."""
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    l1, u1 = wilson_ci(k1, n1)
    l2, u2 = wilson_ci(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def odds_ratio_woolf(a: int, b: int, c: int, d: int) -> dict:
    """OR과 Woolf(logit) 95% CI. 0 셀이 있으면 Haldane-Anscombe +0.5 보정."""
    corrected = 0 in (a, b, c, d)
    aa, bb, cc, dd = (
        (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if corrected else (a, b, c, d)
    )
    or_ = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(or_) - Z95 * se)
    hi = math.exp(math.log(or_) + Z95 * se)
    return {
        "or": round(or_, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "haldane_corrected": corrected,
    }


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else None
    return (po - pe) / (1 - pe)


def fleiss_kappa(vote_sums: list[int], raters: int = 3) -> float | None:
    """이진 범주(True/False), raters명 평정의 Fleiss' kappa.

    vote_sums: 항목별 True 표 수(0..raters).
    """
    n = len(vote_sums)
    if n == 0:
        return None
    p_bar_sum = 0.0
    true_total = 0
    rr = raters * (raters - 1)
    for v in vote_sums:
        f = raters - v
        p_bar_sum += (v * (v - 1) + f * (f - 1)) / rr
        true_total += v
    p_bar = p_bar_sum / n
    p_true = true_total / (n * raters)
    pe = p_true * p_true + (1 - p_true) * (1 - p_true)
    if pe >= 1.0:
        return 1.0 if p_bar >= 1.0 else None
    return (p_bar - pe) / (1 - pe)


def two_by_two_stats(a: int, b: int, c: int, d: int) -> dict:
    """target(a,b) 대 control(c,d) 2x2의 비율차 CI, OR CI, chi2."""
    n1, n2 = a + b, c + d
    rate_t = a / n1 * 100 if n1 else 0.0
    rate_c = c / n2 * 100 if n2 else 0.0
    lo, hi = newcombe_diff_ci(a, n1, c, n2)
    chi2, p = cfs.chi_square_2x2(a, b, c, d)
    return {
        "target_positive": a,
        "target_n": n1,
        "control_positive": c,
        "control_n": n2,
        "target_rate": round(rate_t, 2),
        "control_rate": round(rate_c, 2),
        "diff_pp": round(rate_t - rate_c, 2),
        "diff_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        "odds_ratio": odds_ratio_woolf(a, b, c, d),
        "chi2": round(chi2, 2),
        "p": p,
    }


def sign_test(pos: int, neg: int) -> float:
    """양측 정확 이항검정 p (동률 제외)."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * cdf)


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> dict | None:
    """층별 (a,b,c,d)에서 MH 공통 OR과 Robins-Breslow-Greenland 95% CI."""
    r_sum = s_sum = 0.0
    sum_pr = sum_psqr = sum_qs = 0.0
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        used += 1
        rk = a * d / n
        sk = b * c / n
        pk = (a + d) / n
        qk = (b + c) / n
        r_sum += rk
        s_sum += sk
        sum_pr += pk * rk
        sum_psqr += pk * sk + qk * rk
        sum_qs += qk * sk
    if used == 0 or s_sum == 0 or r_sum == 0:
        return None
    or_mh = r_sum / s_sum
    var = (
        sum_pr / (2 * r_sum * r_sum)
        + sum_psqr / (2 * r_sum * s_sum)
        + sum_qs / (2 * s_sum * s_sum)
    )
    se = math.sqrt(var)
    lo = math.exp(math.log(or_mh) - Z95 * se)
    hi = math.exp(math.log(or_mh) + Z95 * se)
    return {
        "or": round(or_mh, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "strata_used": used,
    }


def woolf_homogeneity(strata: list[tuple[int, int, int, int]]) -> dict | None:
    """Woolf 동질성 검정 (Haldane +0.5 일괄 보정)."""
    ws, lnors = [], []
    for a, b, c, d in strata:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        w = 1.0 / (1 / aa + 1 / bb + 1 / cc + 1 / dd)
        ws.append(w)
        lnors.append(math.log((aa * dd) / (bb * cc)))
    k = len(ws)
    if k < 2:
        return None
    pooled = sum(w * l for w, l in zip(ws, lnors)) / sum(ws)
    x2 = sum(w * (l - pooled) ** 2 for w, l in zip(ws, lnors))
    df = k - 1
    return {"chi2": round(x2, 2), "df": df, "p": chi2_sf(x2, df), "strata": k}


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------

def classify_bu(book: str) -> str:
    """서명 → 부(部) 분류. analysis/scripts/analyze_genre_by_category.py와 동일 기준."""
    for stem in ("논어", "맹자", "대학", "중용"):
        if book.startswith(stem):
            return "경부(사서)"
    for stem in ("시경", "서경", "주역", "예기", "춘추"):
        if book.startswith(stem):
            return "경부(오경)"
    if "당송팔대가문초" in book:
        return "집부"
    if "자치통감강목" in book or "통감" in book:
        return "사부"
    return "기타"


def collect_items(cfg: dict) -> list[dict]:
    """세 모델 공통 키 항목을 (book, arm, votes)로 수집한다."""
    rows_by_model = {model: cfs.load_section_rows(model, cfg) for model in cfs.MODELS}
    keyed = {
        model: {cfs.row_key(row): row for row in rows}
        for model, rows in rows_by_model.items()
    }
    common = set.intersection(*(set(v.keys()) for v in keyed.values()))
    target, control = str(cfg["target"]), str(cfg["control"])
    items = []
    for key in sorted(common):
        marker_type = key[3]
        if marker_type == target:
            arm = "target"
        elif marker_type == control:
            arm = "control"
        else:
            continue
        votes = tuple(
            cfs.parse_bool(keyed[model][key].get("llm_judgment")) for model in cfs.MODELS
        )
        items.append(
            {
                "book": key[0],
                "bu": classify_bu(key[0]),
                "arm": arm,
                "votes": votes,
                "vote_sum": sum(votes),
            }
        )
    return items


# ---------------------------------------------------------------------------
# 섹션 분석
# ---------------------------------------------------------------------------

def agreement_block(items: list[dict]) -> dict:
    model_names = list(cfs.MODELS)
    vote_sums = [it["vote_sum"] for it in items]
    unanimous = sum(1 for v in vote_sums if v in (0, 3))
    pairwise = OrderedDict()
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs = [(it["votes"][i], it["votes"][j]) for it in items]
            agree = sum(1 for x, y in pairs if x == y)
            kappa = cohen_kappa(pairs)
            pairwise[f"{model_names[i]}__{model_names[j]}"] = {
                "percent_agreement": round(agree / len(pairs) * 100, 1) if pairs else None,
                "cohen_kappa": round(kappa, 3) if kappa is not None else None,
            }
    by_arm = {}
    for arm in ("target", "control"):
        arm_sums = [it["vote_sum"] for it in items if it["arm"] == arm]
        k = fleiss_kappa(arm_sums)
        by_arm[arm] = round(k, 3) if k is not None else None
    overall = fleiss_kappa(vote_sums)
    return {
        "n_items": len(items),
        "unanimous_pct": round(unanimous / len(items) * 100, 1) if items else None,
        "fleiss_kappa": round(overall, 3) if overall is not None else None,
        "fleiss_kappa_by_arm": by_arm,
        "pairwise": pairwise,
    }


def consensus_counts(items: list[dict]) -> dict[str, Counter]:
    """arm별 O/S/X 카운트 (만장일치 O, 만장일치 X, 분할 S)."""
    out = {"target": Counter(), "control": Counter()}
    for it in items:
        cat = "O" if it["vote_sum"] == 3 else ("X" if it["vote_sum"] == 0 else "S")
        out[it["arm"]][cat] += 1
    return out


def bootstrap_v(counts: dict[str, Counter], boot: int, rng: random.Random) -> dict | None:
    if boot <= 0:
        return None
    cats = ("O", "S", "X")
    arms = {}
    for arm in ("target", "control"):
        n = sum(counts[arm].values())
        weights = [counts[arm][c] for c in cats]
        cum = []
        acc = 0
        for w in weights:
            acc += w
            cum.append(acc)
        arms[arm] = (n, cum)
    vs = []
    for _ in range(boot):
        table = []
        for arm in ("target", "control"):
            n, cum = arms[arm]
            draw = Counter(rng.choices(cats, cum_weights=cum, k=n))
            table.append([draw.get(c, 0) for c in cats])
        chi2 = cfs.chi_square_table(table)
        total = sum(sum(r) for r in table)
        vs.append(math.sqrt(chi2 / total) if total else 0.0)
    vs.sort()
    lo = vs[max(0, int(0.025 * boot) - 1)]
    hi = vs[min(boot - 1, int(math.ceil(0.975 * boot)) - 1)]
    return {"boot": boot, "v_ci95": [round(lo, 3), round(hi, 3)]}


def sensitivity_block(items: list[dict]) -> OrderedDict:
    thresholds = {"unanimous": 3, "majority": 2, "any": 1}
    out = OrderedDict()
    for name, thr in thresholds.items():
        a = sum(1 for it in items if it["arm"] == "target" and it["vote_sum"] >= thr)
        b = sum(1 for it in items if it["arm"] == "target" and it["vote_sum"] < thr)
        c = sum(1 for it in items if it["arm"] == "control" and it["vote_sum"] >= thr)
        d = sum(1 for it in items if it["arm"] == "control" and it["vote_sum"] < thr)
        out[name] = {"label": DEFINITIONS[name], **two_by_two_stats(a, b, c, d)}
    return out


def build_strata(
    items: list[dict], group_key: str, positive=lambda it: it["vote_sum"] == 3
) -> "OrderedDict[str, tuple[int, int, int, int]]":
    """group_key('book'|'bu') 층별 (a,b,c,d). a/c는 positive, target/control 순."""
    table: "OrderedDict[str, list[int]]" = OrderedDict()
    for it in items:
        g = it[group_key]
        if g not in table:
            table[g] = [0, 0, 0, 0]
        idx = (0 if positive(it) else 1) + (0 if it["arm"] == "target" else 2)
        table[g][idx] += 1
    return OrderedDict((g, tuple(v)) for g, v in table.items())


def stratified_block(items: list[dict], min_arm: int = 5) -> dict:
    """book·부 층화 분석. positive = 만장일치 O (O 대 나머지), 모델별은 단독 판정."""
    out: dict = {}

    book_strata = build_strata(items, "book")
    both = OrderedDict(
        (g, s) for g, s in book_strata.items() if (s[0] + s[1]) > 0 and (s[2] + s[3]) > 0
    )
    eligible = OrderedDict(
        (g, s)
        for g, s in both.items()
        if (s[0] + s[1]) >= min_arm and (s[2] + s[3]) >= min_arm
    )
    out["books_total"] = len(book_strata)
    out["books_with_both_arms"] = len(both)
    out["books_min_arm"] = {"threshold": min_arm, "count": len(eligible)}

    # 전체(비층화) OR — 층화 OR과의 교란 비교 기준
    a = sum(s[0] for s in book_strata.values())
    b = sum(s[1] for s in book_strata.values())
    c = sum(s[2] for s in book_strata.values())
    d = sum(s[3] for s in book_strata.values())
    out["crude"] = two_by_two_stats(a, b, c, d)

    out["mh_book"] = mantel_haenszel(list(both.values()))
    out["woolf_homogeneity_book"] = woolf_homogeneity(list(eligible.values()))

    pos = neg = ties = 0
    per_book = []
    for g, (sa, sb, sc, sd) in eligible.items():
        rt = sa / (sa + sb) * 100
        rc = sc / (sc + sd) * 100
        if rt > rc:
            pos += 1
        elif rt < rc:
            neg += 1
        else:
            ties += 1
        per_book.append(
            {
                "book": g,
                "target_n": sa + sb,
                "control_n": sc + sd,
                "target_o_pct": round(rt, 1),
                "control_o_pct": round(rc, 1),
                "diff_pp": round(rt - rc, 1),
            }
        )
    per_book.sort(key=lambda r: -(r["target_n"] + r["control_n"]))
    out["sign_test_book"] = {
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "p_two_sided": sign_test(pos, neg),
    }
    out["per_book"] = per_book

    bu_strata = build_strata(items, "bu")
    out["per_bu"] = OrderedDict(
        (
            g,
            {
                "target_n": s[0] + s[1],
                "control_n": s[2] + s[3],
                "target_o_pct": round(s[0] / (s[0] + s[1]) * 100, 1) if s[0] + s[1] else None,
                "control_o_pct": round(s[2] / (s[2] + s[3]) * 100, 1) if s[2] + s[3] else None,
                "odds_ratio": odds_ratio_woolf(*s) if all(x >= 0 for x in s) else None,
            },
        )
        for g, s in bu_strata.items()
    )
    out["mh_bu"] = mantel_haenszel(
        [s for s in bu_strata.values() if (s[0] + s[1]) > 0 and (s[2] + s[3]) > 0]
    )

    per_model_mh = OrderedDict()
    for idx, model in enumerate(cfs.MODELS):
        strata = build_strata(items, "book", positive=lambda it, i=idx: it["votes"][i])
        usable = [s for s in strata.values() if (s[0] + s[1]) > 0 and (s[2] + s[3]) > 0]
        per_model_mh[model] = mantel_haenszel(usable)
    out["per_model_mh_book"] = per_model_mh

    return out


def analyze_section(cfg: dict, boot: int, rng: random.Random) -> dict:
    items = collect_items(cfg)
    counts = consensus_counts(items)
    t, c = counts["target"], counts["control"]
    a = t["O"]
    b = t["S"] + t["X"]
    cc = c["O"]
    dd = c["S"] + c["X"]

    ci_block = {
        "o_vs_rest": two_by_two_stats(a, b, cc, dd),
        "o_vs_x_strict": two_by_two_stats(t["O"], t["X"], c["O"], c["X"]),
        "cramers_v_bootstrap": bootstrap_v(counts, boot, rng),
    }
    return {
        "label": cfg["label"],
        "target_marker": cfg["target"],
        "control_marker": cfg["control"],
        "n_target": sum(t.values()),
        "n_control": sum(c.values()),
        "agreement": agreement_block(items),
        "interval_estimates": ci_block,
        "sensitivity": sensitivity_block(items),
        "stratified": stratified_block(items),
    }


# ---------------------------------------------------------------------------
# 보고서 생성
# ---------------------------------------------------------------------------

def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def fmt_or(block: dict | None) -> str:
    if not block:
        return "—"
    return f"{block['or']} [{block['ci95'][0]}, {block['ci95'][1]}]"


def render_report(payload: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 강건성·일치도 보고서")
    add("")
    add(f"생성: {payload['generated_at']} · source={payload['source']} · "
        f"bootstrap B={payload['boot']} (seed {payload['seed']})")
    add("")
    add("`scripts/compute_robustness_stats.py`가 기존 판정 CSV에서 산출한다. "
        "새 LLM 호출은 없으며, 익명 판정 CSV만으로 동일 수치가 재현된다. "
        "기준 점추정은 `results/final_stats_v3.1_cleaned_balanced.json`을 그대로 두고, "
        "본 보고서는 그 위에 구간추정·일치도·민감도·층화 결과를 보탠다.")
    add("")
    add("## 요약")
    add("")
    add("| 섹션 | Fleiss κ | O율차 (95% CI, pp) | OR (95% CI) | MH OR_book (95% CI) | 합의 정의 민감도 | book sign test |")
    add("|---|---|---|---|---|---|---|")
    for sec in payload["sections"].values():
        ci = sec["interval_estimates"]["o_vs_rest"]
        mh = sec["stratified"]["mh_book"]
        st = sec["stratified"]["sign_test_book"]
        sens = sec["sensitivity"]
        directions = {
            name: blk["diff_pp"] > 0 and blk["odds_ratio"]["ci95"][0] > 1
            for name, blk in sens.items()
        }
        stable = "3/3 유지" if all(directions.values()) else (
            f"{sum(directions.values())}/3 유지"
        )
        add(
            f"| {sec['label']} ({sec['target_marker']} 대 {sec['control_marker']}) "
            f"| {sec['agreement']['fleiss_kappa']} "
            f"| {ci['diff_pp']} [{ci['diff_ci95_pp'][0]}, {ci['diff_ci95_pp'][1]}] "
            f"| {fmt_or(ci['odds_ratio'])} "
            f"| {fmt_or(mh)} "
            f"| {stable} "
            f"| +{st['positive']}/-{st['negative']} (p={fmt_p(st['p_two_sided'])}) |"
        )
    add("")
    add("O율차·OR은 만장일치 O 대 나머지(S+X) 기준 2x2다. "
        "MH OR_book은 book 층화 Mantel-Haenszel 공통 OR로, 서종 구성 차이를 통제한 값이다.")
    add("")

    for key, sec in payload["sections"].items():
        add(f"## {key}: {sec['label']} — {sec['target_marker']} 대 {sec['control_marker']}")
        add("")
        add(f"공통 판정 항목: target {sec['n_target']:,} · control {sec['n_control']:,}")
        add("")

        ag = sec["agreement"]
        add("### 모델 간 일치도")
        add("")
        add(f"- Fleiss κ (전체): **{ag['fleiss_kappa']}** · 만장일치 비율 {ag['unanimous_pct']}%")
        add(f"- Fleiss κ (target): {ag['fleiss_kappa_by_arm']['target']} · "
            f"(control): {ag['fleiss_kappa_by_arm']['control']}")
        add("")
        add("| 모델 쌍 | 일치율 | Cohen κ |")
        add("|---|---:|---:|")
        for pair, blk in ag["pairwise"].items():
            m1, m2 = pair.split("__")
            add(f"| {cfs.MODELS[m1]} × {cfs.MODELS[m2]} | {blk['percent_agreement']}% | {blk['cohen_kappa']} |")
        add("")

        ci = sec["interval_estimates"]
        add("### 효과크기 구간추정")
        add("")
        add("| 비교 | target O% | control O% | 차이 pp (95% CI) | OR (95% CI) | χ² | p |")
        add("|---|---:|---:|---|---|---:|---|")
        for name, label in (("o_vs_rest", "O 대 나머지(S+X)"), ("o_vs_x_strict", "O 대 X (S 제외)")):
            blk = ci[name]
            add(
                f"| {label} | {blk['target_rate']} | {blk['control_rate']} "
                f"| {blk['diff_pp']} [{blk['diff_ci95_pp'][0]}, {blk['diff_ci95_pp'][1]}] "
                f"| {fmt_or(blk['odds_ratio'])} | {blk['chi2']} | {fmt_p(blk['p'])} |"
            )
        bv = ci["cramers_v_bootstrap"]
        if bv:
            add("")
            add(f"- Cramér’s V 부트스트랩 95% CI (3범주 O/S/X 표, B={bv['boot']}): "
                f"[{bv['v_ci95'][0]}, {bv['v_ci95'][1]}]")
        add("")

        add("### 합의 정의 민감도")
        add("")
        add("| O 정의 | target % | control % | 차이 pp (95% CI) | OR (95% CI) | p |")
        add("|---|---:|---:|---|---|---|")
        for blk in sec["sensitivity"].values():
            add(
                f"| {blk['label']} | {blk['target_rate']} | {blk['control_rate']} "
                f"| {blk['diff_pp']} [{blk['diff_ci95_pp'][0]}, {blk['diff_ci95_pp'][1]}] "
                f"| {fmt_or(blk['odds_ratio'])} | {fmt_p(blk['p'])} |"
            )
        add("")

        st = sec["stratified"]
        add("### 서종 층화 (book·부)")
        add("")
        add(f"- 층화 범위: 전체 {st['books_total']}책, 양군 보유 {st['books_with_both_arms']}책, "
            f"군별 {st['books_min_arm']['threshold']}건 이상 {st['books_min_arm']['count']}책")
        add(f"- 비층화(crude) OR: {fmt_or(st['crude']['odds_ratio'])}")
        add(f"- **book 층화 MH OR: {fmt_or(st['mh_book'])}** (strata {st['mh_book']['strata_used'] if st['mh_book'] else '—'})")
        wh = st["woolf_homogeneity_book"]
        if wh:
            add(f"- Woolf 동질성: χ²({wh['df']}) = {wh['chi2']}, p = {fmt_p(wh['p'])} — "
                + ("책 간 OR 이질성 유의" if wh["p"] < 0.05 else "책 간 OR 동질성 기각 못 함"))
        sg = st["sign_test_book"]
        add(f"- book sign test: 양(+) {sg['positive']}책 / 음(-) {sg['negative']}책 / 동률 {sg['ties']}책, "
            f"p = {fmt_p(sg['p_two_sided'])}")
        add(f"- 부(部) 층화 MH OR: {fmt_or(st['mh_bu'])}")
        add("")
        add("| 부 | target n | target O% | control n | control O% | OR (95% CI) |")
        add("|---|---:|---:|---:|---:|---|")
        for bu, blk in st["per_bu"].items():
            add(
                f"| {bu} | {blk['target_n']:,} | {blk['target_o_pct']} "
                f"| {blk['control_n']:,} | {blk['control_o_pct']} | {fmt_or(blk['odds_ratio'])} |"
            )
        add("")
        add("| 모델 | book 층화 MH OR (95% CI) |")
        add("|---|---|")
        for model, blk in st["per_model_mh_book"].items():
            add(f"| {cfs.MODELS[model]} | {fmt_or(blk)} |")
        add("")
        add("상위 10책 (표본 큰 순):")
        add("")
        add("| book | target n | target O% | control n | control O% | 차이 pp |")
        add("|---|---:|---:|---:|---:|---:|")
        for row in st["per_book"][:10]:
            add(
                f"| {row['book']} | {row['target_n']:,} | {row['target_o_pct']} "
                f"| {row['control_n']:,} | {row['control_o_pct']} | {row['diff_pp']:+} |"
            )
        add("")

    add("## 해석 지침과 한계")
    add("")
    add("- κ 해석은 관례상 0.21~0.40 보통, 0.41~0.60 중간, 0.61~0.80 상당 수준이다. "
        "κ가 낮은 섹션은 합의 판정(만장일치)이 보수적 필터로 작동했음을 뜻하며, "
        "단일 모델 결과보다 합의 통계를 인용해야 하는 근거가 된다.")
    add("- MH OR이 crude OR보다 작으면 서종 구성 차이가 효과를 부풀린 것이고, "
        "비슷하면 효과가 서종 통제 후에도 유지되는 것이다. "
        "Woolf 동질성이 유의하면 공통 OR은 요약치로만 읽고 book별 방향(sign test)을 함께 인용한다.")
    add("- 부트스트랩 CI는 현재 표본 내 재추출 변동만 반영한다. 대조군 표본 자체를 새로 뽑는 "
        "seed robustness(REPRODUCE.md §8.3)는 추가 LLM 판정이 필요한 별도 작업이다.")
    add("- 민감도 표의 1표 이상(any) 정의는 가장 관대한 모델의 단독 긍정까지 포함하므로 "
        "상한 점검용이다. 본문 인용 기준은 현행 만장일치 정의를 유지한다.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3-model 강건성·일치도 통계")
    parser.add_argument("--source", choices=["auto", "raw", "anon"], default="auto",
                        help="판정 CSV 소스 (compute_final_stats.py와 동일)")
    parser.add_argument("--boot", type=int, default=2000,
                        help="Cramér's V 부트스트랩 반복 수 (0이면 생략)")
    parser.add_argument("--seed", type=int, default=20260611,
                        help="부트스트랩 시드")
    args = parser.parse_args(argv)
    cfs.SOURCE = args.source

    rng = random.Random(args.seed)
    sections = OrderedDict()
    for section, cfg in cfs.SECTIONS.items():
        print(f"[{section}] {cfg['label']} 분석 중...")
        sections[section] = analyze_section(cfg, args.boot, rng)

    payload = {
        "generated_at": dt.date.today().isoformat(),
        "source": args.source,
        "boot": args.boot,
        "seed": args.seed,
        "basis": "results/{model}/section*_judgments(+supplement) CSV — final_stats v3.1과 동일 입력",
        "sections": sections,
    }

    json_path = RESULTS / "robustness_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {json_path.relative_to(REPO)}")

    report_path = RESULTS / "ROBUSTNESS_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(payload))
    print(f"wrote {report_path.relative_to(REPO)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    log_record = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "compute_robustness_stats.py",
        "source": args.source,
        "boot": args.boot,
        "seed": args.seed,
        "outputs": [str(json_path.relative_to(REPO)), str(report_path.relative_to(REPO))],
        "summary": {
            key: {
                "fleiss_kappa": sec["agreement"]["fleiss_kappa"],
                "diff_pp": sec["interval_estimates"]["o_vs_rest"]["diff_pp"],
                "or": sec["interval_estimates"]["o_vs_rest"]["odds_ratio"]["or"],
                "mh_or_book": sec["stratified"]["mh_book"]["or"] if sec["stratified"]["mh_book"] else None,
            }
            for key, sec in sections.items()
        },
    }
    with open(LOGS / "robustness_stats.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    print("logged logs/robustness_stats.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
