import csv
from collections import Counter

c_cell = Counter()
c_dansa = Counter()
total = 0

with open(r"C:\Users\junto\Downloads\analysis_v8\parallel_data_v2.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        total += 1
        c_cell[row.get("cell", "")] += 1
        c_dansa[row.get("dansa_category", "")] += 1

print(f"Total rows: {total}")
print(f"\n=== cell column ({len(c_cell)} unique) ===")
for val, cnt in c_cell.most_common():
    print(f"  {cnt:>6}  {val}")

print(f"\n=== dansa_category column ({len(c_dansa)} unique) ===")
for val, cnt in c_dansa.most_common():
    print(f"  {cnt:>6}  {val}")
