"""사부분류별·서종별·작가별 니라/라 O-ratio 통계"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "parallel_data_v2_cleaned.tsv", sep="\t", encoding="utf-8")
df["mt"] = df["cell"].apply(lambda c: "nira" if "니라" in c else "ra")
df["is_O"] = df["cell"].apply(lambda c: 1 if c.endswith("_O") else 0)

def classify(b):
    if b.startswith("논어"): return "경부", "사서", "논어집주"
    if b.startswith("맹자"): return "경부", "사서", "맹자집주"
    if b.startswith("대학"): return "경부", "사서", "대학장구"
    if b.startswith("중용"): return "경부", "사서", "중용장구"
    if b.startswith("시경"): return "경부", "오경", "시경집전"
    if b.startswith("서경"): return "경부", "오경", "서경집전"
    if b.startswith("주역"): return "경부", "오경", "주역전의"
    if b.startswith("예기"): return "경부", "오경", "예기집설대전"
    if b.startswith("춘추"): return "경부", "오경", "춘추좌씨전"
    if b.startswith("자치"): return "사부", "편년체", "자치통감강목"
    if "한유" in b: return "집부", "당송팔대가문초", "한유"
    if "유종원" in b: return "집부", "당송팔대가문초", "유종원"
    if "구양수" in b: return "집부", "당송팔대가문초", "구양수"
    if "소순" in b: return "집부", "당송팔대가문초", "소순"
    if "소식" in b: return "집부", "당송팔대가문초", "소식"
    if "소철" in b: return "집부", "당송팔대가문초", "소철"
    if "증공" in b: return "집부", "당송팔대가문초", "증공"
    if "왕안석" in b: return "집부", "당송팔대가문초", "왕안석"
    return "미분류", "미분류", b

df[["사부", "서종", "작가_서명"]] = df["book"].apply(lambda b: pd.Series(classify(b)))

def compute_or_chi2(sub):
    ct = pd.crosstab(sub["mt"], sub["is_O"])
    if ct.shape != (2, 2):
        return None
    nira_o = ct.loc["nira", 1]; nira_x = ct.loc["nira", 0]; nira_t = nira_o + nira_x
    ra_o = ct.loc["ra", 1]; ra_x = ct.loc["ra", 0]; ra_t = ra_o + ra_x
    if nira_t < 5 or ra_t < 5:
        return None
    nira_pct = nira_o / nira_t * 100
    ra_pct = ra_o / ra_t * 100
    chi2, p = stats.chi2_contingency(ct)[:2]
    if ra_o > 0 and ra_x > 0 and nira_x > 0:
        OR = (nira_o / nira_x) / (ra_o / ra_x)
    else:
        OR = float("inf")
    return {
        "nira_pct": round(nira_pct, 1), "ra_pct": round(ra_pct, 1),
        "diff": round(nira_pct - ra_pct, 1),
        "nira_N": nira_t, "ra_N": ra_t,
        "nira_O": nira_o, "ra_O": ra_o,
        "OR": round(OR, 3), "chi2": round(chi2, 1), "p": p,
    }

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

# === 1. 사부분류별 ===
print("=" * 90)
print("1. 사부분류별 (경부 / 사부 / 집부)")
print("=" * 90)
fmt = "{:<6} 니라: {:>5.1f}% ({:>4})  라: {:>5.1f}% ({:>4})  diff: {:>+6.1f}pp  OR={:.3f}  chi2={:.1f}  p={:.2e} {}"
for cat in ["경부", "사부", "집부"]:
    sub = df[df["사부"] == cat]
    r = compute_or_chi2(sub)
    if r:
        print(fmt.format(cat, r["nira_pct"], r["nira_N"], r["ra_pct"], r["ra_N"],
                         r["diff"], r["OR"], r["chi2"], r["p"], sig_label(r["p"])))

# === 2. 서종별 ===
print()
print("=" * 90)
print("2. 서종별 (사서 / 오경 / 편년체 / 당송팔대가문초)")
print("=" * 90)
for (sabu, sj), sub in df.groupby(["사부", "서종"]):
    r = compute_or_chi2(sub)
    if r:
        label = f"{sabu}-{sj}"
        print(fmt.format(label, r["nira_pct"], r["nira_N"], r["ra_pct"], r["ra_N"],
                         r["diff"], r["OR"], r["chi2"], r["p"], sig_label(r["p"])))

# === 3. 경부 개별 서종 ===
print()
print("=" * 90)
print("3. 경부 개별 서명")
print("=" * 90)
for title, sub in df[df["사부"] == "경부"].groupby("작가_서명"):
    r = compute_or_chi2(sub)
    if r:
        print(fmt.format(title, r["nira_pct"], r["nira_N"], r["ra_pct"], r["ra_N"],
                         r["diff"], r["OR"], r["chi2"], r["p"], sig_label(r["p"])))

# === 4. 당송팔대가문초 작가별 ===
print()
print("=" * 90)
print("4. 당송팔대가문초 작가별")
print("=" * 90)
results = []
for author, sub in df[df["서종"] == "당송팔대가문초"].groupby("작가_서명"):
    r = compute_or_chi2(sub)
    if r:
        r["작가"] = author
        results.append(r)
rdf = pd.DataFrame(results).sort_values("diff", ascending=False)
for _, r in rdf.iterrows():
    print(fmt.format(r["작가"], r["nira_pct"], r["nira_N"], r["ra_pct"], r["ra_N"],
                     r["diff"], r["OR"], r["chi2"], r["p"], sig_label(r["p"])))

# === 5. 사부별 방향 일치 book 수 ===
print()
print("=" * 90)
print("5. 사부분류별 방향 일치 (book 단위 sign test)")
print("=" * 90)
bm = df.groupby(["book", "mt", "사부"]).agg(tot=("is_O", "count"), oc=("is_O", "sum")).reset_index()
bm["pct"] = bm["oc"] / bm["tot"] * 100
bb = bm.groupby("book")["mt"].nunique()
bb = bb[bb == 2].index.tolist()
bm = bm[bm["book"].isin(bb)]
w = bm.pivot(index="book", columns="mt", values=["pct", "tot"])
w.columns = [f"{c[1]}_{c[0]}" for c in w.columns]
w = w[(w["nira_tot"] >= 5) & (w["ra_tot"] >= 5)]
w["diff"] = w["nira_pct"] - w["ra_pct"]
sabu_map = df.drop_duplicates("book").set_index("book")["사부"]
w["사부"] = w.index.map(sabu_map)

for cat in ["경부", "사부", "집부"]:
    ws = w[w["사부"] == cat]
    pos = (ws["diff"] > 0).sum()
    neg = (ws["diff"] < 0).sum()
    eq = (ws["diff"] == 0).sum()
    tot = len(ws)
    if pos + neg > 0:
        bp = stats.binomtest(pos, pos + neg, 0.5).pvalue
    else:
        bp = 1.0
    print(f"  {cat}: {tot}개 book — 니라>라 {pos}개, 라>니라 {neg}개, 동률 {eq}개 | binomial p={bp:.4f}")

# === 파일 저장 ===
import json

STAT_DIR = f"{ROOT}/stats"
import os; os.makedirs(STAT_DIR, exist_ok=True)

# (A) book별 전체 테이블
bm_all = df.groupby(["book", "mt", "사부", "서종", "작가_서명"]).agg(
    tot=("is_O", "count"), oc=("is_O", "sum")
).reset_index()
bm_all["O_ratio"] = round(bm_all["oc"] / bm_all["tot"] * 100, 1)
bm_both = bm_all.groupby("book")["mt"].nunique()
bm_both = bm_both[bm_both == 2].index.tolist()
bm_f = bm_all[bm_all["book"].isin(bm_both)].copy()
wp = bm_f.pivot(index="book", columns="mt", values=["oc", "tot", "O_ratio"])
wp.columns = [f"{c[1]}_{c[0]}" for c in wp.columns]
wp = wp[(wp["nira_tot"] >= 5) & (wp["ra_tot"] >= 5)].copy()
wp["diff"] = round(wp["nira_O_ratio"] - wp["ra_O_ratio"], 1)
meta = df.drop_duplicates("book").set_index("book")[["사부", "서종", "작가_서명"]]
wp = wp.join(meta).sort_values("diff", ascending=False)
wp.to_csv(f"{STAT_DIR}/per_book_stats.tsv", sep="\t", encoding="utf-8")
print(f"\n저장: {STAT_DIR}/per_book_stats.tsv ({len(wp)}행)")

# (B) 사부분류별
rows_sabu = []
for cat in ["경부", "사부", "집부"]:
    r = compute_or_chi2(df[df["사부"] == cat])
    if r:
        r["분류"] = cat
        r["sig"] = sig_label(r["p"])
        rows_sabu.append(r)
pd.DataFrame(rows_sabu).to_csv(f"{STAT_DIR}/by_sabu.tsv", sep="\t", index=False, encoding="utf-8")
print(f"저장: {STAT_DIR}/by_sabu.tsv")

# (C) 서종별
rows_sj = []
for (sabu, sj), sub in df.groupby(["사부", "서종"]):
    r = compute_or_chi2(sub)
    if r:
        r["사부"] = sabu; r["서종"] = sj; r["sig"] = sig_label(r["p"])
        rows_sj.append(r)
pd.DataFrame(rows_sj).to_csv(f"{STAT_DIR}/by_seojong.tsv", sep="\t", index=False, encoding="utf-8")
print(f"저장: {STAT_DIR}/by_seojong.tsv")

# (D) 경부 개별 서명
rows_gyeong = []
for title, sub in df[df["사부"] == "경부"].groupby("작가_서명"):
    r = compute_or_chi2(sub)
    if r:
        r["서명"] = title; r["sig"] = sig_label(r["p"])
        rows_gyeong.append(r)
pd.DataFrame(rows_gyeong).to_csv(f"{STAT_DIR}/by_gyeongbu_title.tsv", sep="\t", index=False, encoding="utf-8")
print(f"저장: {STAT_DIR}/by_gyeongbu_title.tsv")

# (E) 당송팔대가문초 작가별
rows_author = []
for author, sub in df[df["서종"] == "당송팔대가문초"].groupby("작가_서명"):
    r = compute_or_chi2(sub)
    if r:
        r["작가"] = author; r["sig"] = sig_label(r["p"])
        rows_author.append(r)
pd.DataFrame(rows_author).to_csv(f"{STAT_DIR}/by_author_dangpalgamuncho.tsv", sep="\t", index=False, encoding="utf-8")
print(f"저장: {STAT_DIR}/by_author_dangpalgamuncho.tsv")

# (F) sign test 결과
rows_sign = []
for cat in ["경부", "사부", "집부"]:
    ws = w[w["사부"] == cat]
    pos = (ws["diff"] > 0).sum()
    neg = (ws["diff"] < 0).sum()
    eq = (ws["diff"] == 0).sum()
    tot = len(ws)
    bp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg > 0 else 1.0
    rows_sign.append({"사부": cat, "books": tot, "니라>라": pos, "라>니라": neg,
                       "동률": eq, "binomial_p": round(bp, 6), "sig": sig_label(bp)})
pd.DataFrame(rows_sign).to_csv(f"{STAT_DIR}/sign_test_by_sabu.tsv", sep="\t", index=False, encoding="utf-8")
print(f"저장: {STAT_DIR}/sign_test_by_sabu.tsv")

# (G) 종합 JSON
summary = {
    "총건수": len(df),
    "사부분류별": rows_sabu,
    "서종별": rows_sj,
    "경부_개별서명": rows_gyeong,
    "당송팔대가문초_작가별": rows_author,
    "sign_test": rows_sign,
    "특이_서종": {
        "한유": "집부 유일 강한 역전. diff=-24.9pp, OR=0.192, p=6.56e-23. 라 종결이 행동·태도 결정을 더 많이 표지.",
        "춘추좌씨전": "경부 유일 역전(비유의). diff=-3.0pp, OR=0.877, ns. 서사체 특성으로 니라/라 기능분화 약함.",
        "소순": "집부 소폭 역전. diff=-3.4pp, ns.",
        "증공": "집부 소폭 역전. diff=-7.4pp, ns.",
        "집부_전체": "사부분류 중 유일 비유의. diff=+2.2pp, p=0.068, ns. 작가별 편차 극심(소식+24.5 vs 한유-24.9)."
    }
}
with open(f"{STAT_DIR}/genre_controlled_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"저장: {STAT_DIR}/genre_controlled_summary.json")
print("\n모든 통계 파일 저장 완료.")
