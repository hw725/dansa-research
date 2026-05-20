"""서지정보.xml에서 역자 정보를 추출하여 분석 데이터에 병합"""
import xml.etree.ElementTree as ET
import pandas as pd
import json
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = [
    Path(r"C:/Users/junto/Downloads/2025구축/PC2025(xlsx)"),
    Path(r"C:/Users/junto/Downloads/head-repo/hw725/CSP/xlsx"),
]

# 분석 대상 book 목록
df = pd.read_csv(ROOT / "parallel_data_v2_cleaned.tsv", sep="\t", encoding="utf-8")
books = sorted(df["book"].unique())
print(f"분석 대상 book: {len(books)}종")

# 서지정보 XML에서 역자 추출
def extract_biblio(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        info = root.find("서지정보")
        if info is None:
            return {}
        fields = {}
        for tag in ["대표서명", "대표서명한글", "저자", "역자", "번역서발행년도", "번역서발행자"]:
            el = info.find(tag)
            if el is not None and el.text:
                fields[tag] = el.text.strip()
        return fields
    except Exception as e:
        return {"error": str(e)}

# book명으로 서지정보 파일 찾기
def find_biblio_xml(book_name):
    for d in SEARCH_DIRS:
        # 정확한 폴더명 매칭
        folder = d / book_name
        if folder.exists():
            xml = folder / f"{book_name}_서지정보.xml"
            if xml.exists():
                return xml
        # 권차 없는 서명으로도 시도 (예: 논어집주 → 폴더 없을 수 있음)
    # fallback: 모든 하위에서 검색
    for d in SEARCH_DIRS:
        for xml in d.rglob(f"{book_name}_서지정보.xml"):
            return xml
    return None

# 전체 추출
results = {}
missing = []
for book in books:
    xml_path = find_biblio_xml(book)
    if xml_path:
        info = extract_biblio(xml_path)
        info["xml_path"] = str(xml_path)
        results[book] = info
    else:
        missing.append(book)

print(f"\n서지정보 확보: {len(results)}종")
print(f"미확보: {len(missing)}종")
if missing:
    print(f"  → {missing}")

# 역자별 정리
translator_map = {}
for book, info in results.items():
    t = info.get("역자", "미상")
    translator_map.setdefault(t, []).append(book)

print(f"\n역자 {len(translator_map)}명:")
for t, bks in sorted(translator_map.items(), key=lambda x: -len(x[1])):
    print(f"  {t}: {len(bks)}종 — {', '.join(bks[:5])}{'...' if len(bks)>5 else ''}")

# TSV 저장 (book → 역자 매핑)
rows = []
for book in books:
    info = results.get(book, {})
    rows.append({
        "book": book,
        "역자": info.get("역자", "") or "성백효",
        "저자": info.get("저자", ""),
        "대표서명": info.get("대표서명", ""),
        "번역서발행년도": info.get("번역서발행년도", ""),
        "번역서발행자": info.get("번역서발행자", ""),
    })
bib_df = pd.DataFrame(rows)
bib_df.to_csv(ROOT / "stats" / "book_translators.tsv", sep="\t", index=False, encoding="utf-8")
print(f"\n저장: stats/book_translators.tsv")

# 역자별 통계
print("\n" + "=" * 90)
print("역자별 니라/라 O-ratio 비교")
print("=" * 90)

df["mt"] = df["cell"].apply(lambda c: "nira" if "니라" in c else "ra")
df["is_O"] = df["cell"].apply(lambda c: 1 if c.endswith("_O") else 0)
df["역자"] = df["book"].map(lambda b: results.get(b, {}).get("역자", "") or "성백효")

translator_stats = []
for translator, sub in df.groupby("역자"):
    ct = pd.crosstab(sub["mt"], sub["is_O"])
    if ct.shape != (2, 2):
        continue
    nira_o = ct.loc["nira", 1]; nira_x = ct.loc["nira", 0]; nira_t = nira_o + nira_x
    ra_o = ct.loc["ra", 1]; ra_x = ct.loc["ra", 0]; ra_t = ra_o + ra_x
    if nira_t < 5 or ra_t < 5:
        continue
    nira_pct = nira_o / nira_t * 100
    ra_pct = ra_o / ra_t * 100
    chi2, p = stats.chi2_contingency(ct)[:2]
    if ra_o > 0 and ra_x > 0 and nira_x > 0:
        OR = (nira_o / nira_x) / (ra_o / ra_x)
    else:
        OR = float("inf")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    n_books = sub["book"].nunique()
    book_list = ", ".join(sorted(sub["book"].unique())[:3])
    if n_books > 3:
        book_list += f" 외 {n_books-3}"

    print(f"  {translator:<8} 니라:{nira_pct:>5.1f}%({nira_t:>4})  라:{ra_pct:>5.1f}%({ra_t:>4})  "
          f"diff:{nira_pct-ra_pct:>+6.1f}pp  OR={OR:.3f}  p={p:.2e} {sig}  [{n_books}종: {book_list}]")

    translator_stats.append({
        "역자": translator, "니라O%": round(nira_pct, 1), "라O%": round(ra_pct, 1),
        "diff": round(nira_pct - ra_pct, 1), "니라N": nira_t, "라N": ra_t,
        "OR": round(OR, 3), "chi2": round(chi2, 1), "p": p, "sig": sig,
        "book수": n_books,
    })

ts_df = pd.DataFrame(translator_stats).sort_values("diff", ascending=False)
ts_df.to_csv(ROOT / "stats" / "by_translator.tsv", sep="\t", index=False, encoding="utf-8")
print(f"\n저장: stats/by_translator.tsv")
