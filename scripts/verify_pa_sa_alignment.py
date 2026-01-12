#!/usr/bin/env python3
"""PA와 SA가 같은 문장을 쪼갠 것인지 식별자로 검증"""
import pandas as pd

print("PA와 SA 데이터 로딩...")
pa = pd.read_csv('hyeonto/datasets/pa_train_merged.csv')
sa = pd.read_csv('hyeonto/datasets/sa_train_merged.csv')

print(f"\nPA 행수: {len(pa):,}")
print(f"SA 행수: {len(sa):,}")

# 컬럼 확인
print(f"\nPA 컬럼: {pa.columns.tolist()}")
print(f"SA 컬럼: {sa.columns.tolist()}")

# 식별자 분석
print("\n=== 식별자 비교 ===")

# PA는 문단식별자, 문장식별자 사용
# SA는 문장식별자, 구식별자 사용

# 샘플 데이터로 확인
print("\nPA 샘플 (첫 10행):")
print(pa[['book_name', '문단식별자', '문장식별자']].head(10))

print("\nSA 샘플 (첫 20행):")
print(sa[['book_name', '문장식별자', '구식별자']].head(20))

# 같은 문장식별자를 가진 데이터 비교
print("\n=== 같은 문장을 쪼갠 것인지 검증 ===")

# 첫 번째 책의 첫 번째 문장 확인
first_book = pa['book_name'].iloc[0]
print(f"\n첫 번째 책: {first_book}")

# PA에서 문장식별자 2인 것 찾기 (1은 보통 제목)
pa_sample = pa[(pa['book_name'] == first_book) & (pa['문장식별자'] == 2)]
print(f"\nPA 문장식별자=2:")
if len(pa_sample) > 0:
    print(f"  원문: {pa_sample.iloc[0]['원문'][:100]}...")
    print(f"  번역문: {pa_sample.iloc[0]['번역문'][:100]}...")

# SA에서 같은 문장식별자인 것들 찾기
sa_sample = sa[(sa['book_name'] == first_book) & (sa['문장식별자'] == 2)]
print(f"\nSA 문장식별자=2 (총 {len(sa_sample)}개 구):")
for idx, row in sa_sample.head(10).iterrows():
    print(f"  구{row['구식별자']}: {row['원문']}")

# 통계
print("\n=== 통계 ===")
pa_sentences = pa[['book_name', '문장식별자']].drop_duplicates()
sa_sentences = sa[['book_name', '문장식별자']].drop_duplicates()

print(f"PA 고유 (책, 문장): {len(pa_sentences):,}")
print(f"SA 고유 (책, 문장): {len(sa_sentences):,}")

# SA가 PA보다 많은 문장이 있는지 확인
pa_set = set(zip(pa_sentences['book_name'], pa_sentences['문장식별자']))
sa_set = set(zip(sa_sentences['book_name'], sa_sentences['문장식별자']))

only_in_pa = pa_set - sa_set
only_in_sa = sa_set - pa_set

print(f"\nPA에만 있는 문장: {len(only_in_pa):,}")
print(f"SA에만 있는 문장: {len(only_in_sa):,}")
print(f"공통 문장: {len(pa_set & sa_set):,}")

if len(only_in_pa) > 0:
    print(f"\nPA에만 있는 문장 샘플 (처음 5개):")
    for item in list(only_in_pa)[:5]:
        print(f"  {item}")

if len(only_in_sa) > 0:
    print(f"\nSA에만 있는 문장 샘플 (처음 5개):")
    for item in list(only_in_sa)[:5]:
        print(f"  {item}")

# 결론
print("\n=== 결론 ===")
if len(only_in_pa) == 0 and len(only_in_sa) == 0:
    print("✅ PA와 SA는 완전히 동일한 문장 세트를 포함합니다.")
    print("✅ PA는 문장 단위, SA는 구 단위로 쪼갠 것이 맞습니다.")
elif len(only_in_pa) == 0:
    print("⚠️ SA가 PA보다 더 많은 문장을 포함합니다.")
elif len(only_in_sa) == 0:
    print("⚠️ PA가 SA보다 더 많은 문장을 포함합니다.")
else:
    print("❌ PA와 SA가 서로 다른 문장을 포함합니다.")
