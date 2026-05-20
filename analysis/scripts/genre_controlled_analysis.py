"""Genre-controlled analysis: compare 니라 vs 라 O-ratio within same book.

Controls for genre/source confound by comparing markers within identical books.
Only uses 3-model unanimous consensus sentences (16,381 total).
"""
import pandas as pd
import numpy as np
from scipy import stats

ROOT = r"C:/Users/junto/Downloads/analysis_v8"

df = pd.read_csv(f"{ROOT}/parallel_data_v2.tsv", sep="\t", encoding="utf-8")
print(f"Loaded {len(df)} rows")
print(f"Cells: {df['cell'].value_counts().to_dict()}")

# Extract marker_type and O/X from cell
df["marker_type"] = df["cell"].apply(lambda c: "니라" if "니라" in c else "라")
df["is_O"] = df["cell"].apply(lambda c: 1 if c.endswith("_O") else 0)

# === 1. Overall comparison (baseline) ===
print("\n" + "="*70)
print("1. OVERALL (no genre control)")
print("="*70)
ct_all = pd.crosstab(df["marker_type"], df["is_O"])
ct_all.columns = ["X", "O"]
ct_all["total"] = ct_all.sum(axis=1)
ct_all["O_ratio"] = (ct_all["O"] / ct_all["total"] * 100).round(1)
print(ct_all)
chi2, p, dof, expected = stats.chi2_contingency(ct_all[["X", "O"]])
print(f"\nChi-square = {chi2:.2f}, p = {p:.2e}, dof = {dof}")

# === 2. Per-book comparison ===
print("\n" + "="*70)
print("2. PER-BOOK ANALYSIS (genre control)")
print("="*70)

book_marker = df.groupby(["book", "marker_type"]).agg(
    total=("is_O", "count"),
    O_count=("is_O", "sum")
).reset_index()
book_marker["X_count"] = book_marker["total"] - book_marker["O_count"]
book_marker["O_ratio"] = (book_marker["O_count"] / book_marker["total"] * 100).round(1)

# Books with BOTH 니라 and 라
books_both = book_marker.groupby("book")["marker_type"].nunique()
books_both = books_both[books_both == 2].index.tolist()
print(f"\nBooks with both 니라 and 라: {len(books_both)} / {df['book'].nunique()} total")

# Filter to books with both markers and minimum count
MIN_PER_MARKER = 5
bm = book_marker[book_marker["book"].isin(books_both)].copy()
bm_wide = bm.pivot(index="book", columns="marker_type", values=["O_count", "X_count", "total", "O_ratio"])
bm_wide.columns = [f"{col[1]}_{col[0]}" for col in bm_wide.columns]
bm_wide = bm_wide[
    (bm_wide["니라_total"] >= MIN_PER_MARKER) & (bm_wide["라_total"] >= MIN_PER_MARKER)
].copy()
print(f"Books with both markers and >= {MIN_PER_MARKER} each: {len(bm_wide)}")

bm_wide["O_ratio_diff"] = bm_wide["니라_O_ratio"] - bm_wide["라_O_ratio"]
bm_wide = bm_wide.sort_values("O_ratio_diff", ascending=False)

# Print top results
print(f"\n{'Book':<40} {'니라 O%':>8} {'라 O%':>8} {'Diff':>8} {'니라 N':>7} {'라 N':>7}")
print("-" * 80)
for book, row in bm_wide.iterrows():
    name = book[:38] if len(book) > 38 else book
    print(f"{name:<40} {row['니라_O_ratio']:>7.1f}% {row['라_O_ratio']:>7.1f}% {row['O_ratio_diff']:>+7.1f} {int(row['니라_total']):>7} {int(row['라_total']):>7}")

# === 3. Aggregated controlled comparison (Cochran-Mantel-Haenszel style) ===
print("\n" + "="*70)
print("3. AGGREGATED CONTROLLED COMPARISON")
print("="*70)

# Pool across books: sum O and X for each marker across all shared books
nira_O_sum = int(bm_wide["니라_O_count"].sum())
nira_X_sum = int(bm_wide["니라_X_count"].sum())
ra_O_sum = int(bm_wide["라_O_count"].sum())
ra_X_sum = int(bm_wide["라_X_count"].sum())

nira_total = nira_O_sum + nira_X_sum
ra_total = ra_O_sum + ra_X_sum

print(f"니라: O={nira_O_sum}, X={nira_X_sum}, total={nira_total}, O%={nira_O_sum/nira_total*100:.1f}%")
print(f"라:   O={ra_O_sum}, X={ra_X_sum}, total={ra_total}, O%={ra_O_sum/ra_total*100:.1f}%")

ct_controlled = np.array([[nira_O_sum, nira_X_sum], [ra_O_sum, ra_X_sum]])
chi2_c, p_c = stats.chi2_contingency(ct_controlled)[:2]
print(f"\nChi-square (controlled) = {chi2_c:.2f}, p = {p_c:.2e}")

# Odds ratio
odds_nira = nira_O_sum / nira_X_sum if nira_X_sum > 0 else float('inf')
odds_ra = ra_O_sum / ra_X_sum if ra_X_sum > 0 else float('inf')
OR = odds_nira / odds_ra if odds_ra > 0 else float('inf')
print(f"Odds ratio (니라/라) = {OR:.3f}")
if OR > 1:
    print("  → 니라가 라보다 O(행동·태도 결정) odds가 높음")
else:
    print("  → 라가 니라보다 O(행동·태도 결정) odds가 높음")

# === 4. Per-book sign test ===
print("\n" + "="*70)
print("4. SIGN TEST: how many books show 니라 O% > 라 O%?")
print("="*70)

n_nira_higher = (bm_wide["O_ratio_diff"] > 0).sum()
n_ra_higher = (bm_wide["O_ratio_diff"] < 0).sum()
n_equal = (bm_wide["O_ratio_diff"] == 0).sum()
total_books = len(bm_wide)

print(f"니라 O% > 라 O%: {n_nira_higher} books")
print(f"니라 O% < 라 O%: {n_ra_higher} books")
print(f"니라 O% = 라 O%: {n_equal} books")
print(f"Total: {total_books} books")

# Binomial test (H0: P(니라 higher) = 0.5)
if n_nira_higher + n_ra_higher > 0:
    binom_p = stats.binomtest(n_nira_higher, n_nira_higher + n_ra_higher, 0.5).pvalue
    print(f"\nBinomial test p = {binom_p:.4f}")
    if binom_p < 0.05:
        print("  → 유의: 니라가 라보다 O 비율이 높은 book이 유의하게 많음")
    else:
        print("  → 비유의: book별 방향성이 일관되지 않음")

# === 5. Weighted per-book comparison ===
print("\n" + "="*70)
print("5. WEIGHTED MEAN O-RATIO DIFFERENCE")
print("="*70)
weights = bm_wide["니라_total"] + bm_wide["라_total"]
weighted_diff = np.average(bm_wide["O_ratio_diff"], weights=weights)
unweighted_diff = bm_wide["O_ratio_diff"].mean()
print(f"Unweighted mean diff (니라 O% - 라 O%): {unweighted_diff:+.1f}pp")
print(f"Weighted mean diff (by total N):         {weighted_diff:+.1f}pp")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
전체(비통제): 니라 O%={nira_O_sum/(nira_O_sum+nira_X_sum)*100:.1f}% vs 라 O%={ra_O_sum/(ra_O_sum+ra_X_sum)*100:.1f}%
동일 book 통제 후:
  - {total_books}개 book에서 양쪽 마커 모두 {MIN_PER_MARKER}건 이상 존재
  - 니라 O% > 라 O%: {n_nira_higher}개 book ({n_nira_higher/total_books*100:.0f}%)
  - 가중 평균 차이: {weighted_diff:+.1f}pp
  - Chi-square p = {p_c:.2e}
  - Odds ratio = {OR:.3f}
""")
