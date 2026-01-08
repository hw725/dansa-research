#!/usr/bin/env python3
"""
올바른 7:2:1 분할: (책이름, 식별자) 조합 기준

핵심:
- 44책을 합칠 때 책 이름 컬럼 추가
- (책이름, 문단식별자) 조합으로 고유성 보장
- 이 조합을 기준으로 7:2:1 분할

예:
  Book1_Para1, Book1_Para2, ... (Book1)
  Book2_Para1, Book2_Para2, ... (Book2 - 중복 없음!)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_all_book_folders():
    """xlsx 폴더 내 모든 책 폴더 반환"""
    xlsx_dir = Path("xlsx")
    books = sorted([d for d in xlsx_dir.iterdir() if d.is_dir()])
    return books

def read_excel_safely(filepath):
    """안전하게 Excel 파일 읽기"""
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def combine_all_books_with_book_name(file_suffix):
    """
    모든 44책의 특정 파일을 합침 (책 이름 컬럼 추가)
    """
    books = get_all_book_folders()
    all_dfs = []
    
    for book_dir in books:
        book_name = book_dir.name
        xlsx_files = list(book_dir.glob(f"*{file_suffix}.xlsx"))
        xlsx_files = [f for f in xlsx_files if not str(f).endswith('.bak')]
        
        if not xlsx_files:
            continue
        
        filepath = xlsx_files[0]
        df = read_excel_safely(filepath)
        if df is not None and len(df) > 0:
            # 책 이름 컬럼 추가
            df['book_name'] = book_name
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError(f"{file_suffix} 파일을 찾을 수 없습니다!")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def main():
    print("=" * 80)
    print("올바른 7:2:1 분할: (책이름, 문단식별자) 조합 기준")
    print("=" * 80)
    
    # 1. 각 병렬 유형 로드 (책 이름 컬럼 추가)
    print("\n[Step 1] 44책 데이터 로드 (책 이름 컬럼 추가)...")
    
    print("  문단병렬(_문단병렬) 로드 중...")
    df_pd = combine_all_books_with_book_name('_문단병렬')
    
    print("  문장병렬(_문장병렬) 로드 중...")
    df_pa = combine_all_books_with_book_name('_문장병렬')
    
    print("  구병렬(_구병렬) 로드 중...")
    df_sa = combine_all_books_with_book_name('_구병렬')
    
    print(f"\n  ✓ PD 로드: {len(df_pd)}행 (컬럼: {df_pd.columns.tolist()})")
    print(f"  ✓ PA 로드: {len(df_pa)}행")
    print(f"  ✓ SA 로드: {len(df_sa)}행")
    
    # 2. PD의 고유 (책이름, 문단식별자) 조합을 7:2:1로 분할
    print("\n[Step 2] PD의 고유 (책이름, 문단식별자) 조합을 7:2:1로 분할...")
    
    # (책이름, 문단식별자) 조합 생성
    df_pd['para_key'] = df_pd['book_name'] + '_Para' + df_pd['문단식별자'].astype(str)
    
    unique_para_keys = df_pd['para_key'].unique()
    n_paras = len(unique_para_keys)
    print(f"  고유 (책, 문단) 조합 개수: {n_paras}개")
    
    # 7:2:1로 분할
    train_size = int(n_paras * 0.7)
    val_size = int(n_paras * 0.2)
    
    np.random.seed(42)
    shuffled_idx = np.random.permutation(n_paras)
    
    train_para_keys = set(unique_para_keys[shuffled_idx[:train_size]])
    val_para_keys = set(unique_para_keys[shuffled_idx[train_size:train_size + val_size]])
    test_para_keys = set(unique_para_keys[shuffled_idx[train_size + val_size:]])
    
    print(f"  분할 완료:")
    print(f"    Train: {len(train_para_keys)}개 (책,문단) ({len(train_para_keys)/n_paras*100:.1f}%)")
    print(f"    Val:   {len(val_para_keys)}개 (책,문단) ({len(val_para_keys)/n_paras*100:.1f}%)")
    print(f"    Test:  {len(test_para_keys)}개 (책,문단) ({len(test_para_keys)/n_paras*100:.1f}%)")
    
    # 3. PD 분할
    print("\n[Step 3] PD 분할 ((책, 문단ID) 조합 기준)...")
    
    pd_train = df_pd[df_pd['para_key'].isin(train_para_keys)].drop(columns=['para_key'])
    pd_val = df_pd[df_pd['para_key'].isin(val_para_keys)].drop(columns=['para_key'])
    pd_test = df_pd[df_pd['para_key'].isin(test_para_keys)].drop(columns=['para_key'])
    
    print(f"  PD (문단병렬):")
    print(f"    train: {len(pd_train)}행")
    print(f"    val:   {len(pd_val)}행")
    print(f"    test:  {len(pd_test)}행")
    
    # 4. PA 분할 (PD의 분할된 (책, 문단ID)를 기준으로)
    print("\n[Step 4] PA 분할 ((책, 문단ID) 조합 기준)...")
    
    # PA도 para_key 생성
    df_pa['para_key'] = df_pa['book_name'] + '_Para' + df_pa['문단식별자'].astype(str)
    
    pa_train = df_pa[df_pa['para_key'].isin(train_para_keys)].drop(columns=['para_key'])
    pa_val = df_pa[df_pa['para_key'].isin(val_para_keys)].drop(columns=['para_key'])
    pa_test = df_pa[df_pa['para_key'].isin(test_para_keys)].drop(columns=['para_key'])
    
    print(f"  PA (문장병렬):")
    print(f"    train: {len(pa_train)}행")
    print(f"    val:   {len(pa_val)}행")
    print(f"    test:  {len(pa_test)}행")
    
    # 5. SA 분할 (PA의 분할된 문장ID를 기준으로)
    print("\n[Step 5] SA 분할 (문장식별자 기준, 책이름 포함)...")
    
    # SA도 책_문장ID 조합 필요
    df_sa['sent_key'] = df_sa['book_name'] + '_Sent' + df_sa['문장식별자'].astype(str)
    
    # PA의 train/val/test에 포함된 문장ID들 (책이름 포함)
    train_sent_keys = set(pa_train[pa_train['book_name'].notna()].apply(
        lambda x: x['book_name'] + '_Sent' + str(x['문장식별자']), axis=1))
    val_sent_keys = set(pa_val[pa_val['book_name'].notna()].apply(
        lambda x: x['book_name'] + '_Sent' + str(x['문장식별자']), axis=1))
    test_sent_keys = set(pa_test[pa_test['book_name'].notna()].apply(
        lambda x: x['book_name'] + '_Sent' + str(x['문장식별자']), axis=1))
    
    sa_train = df_sa[df_sa['sent_key'].isin(train_sent_keys)].drop(columns=['sent_key'])
    sa_val = df_sa[df_sa['sent_key'].isin(val_sent_keys)].drop(columns=['sent_key'])
    sa_test = df_sa[df_sa['sent_key'].isin(test_sent_keys)].drop(columns=['sent_key'])
    
    print(f"  SA (구병렬):")
    print(f"    train: {len(sa_train)}행")
    print(f"    val:   {len(sa_val)}행")
    print(f"    test:  {len(sa_test)}행")
    
    # 6. 저장
    print("\n[Step 6] 데이터셋 저장...")
    
    os.makedirs("datasets/pd", exist_ok=True)
    os.makedirs("datasets/pa", exist_ok=True)
    os.makedirs("datasets/sa", exist_ok=True)
    
    pd_train.to_csv("datasets/pd/train.csv", index=False, encoding='utf-8')
    pd_val.to_csv("datasets/pd/val.csv", index=False, encoding='utf-8')
    pd_test.to_csv("datasets/pd/test.csv", index=False, encoding='utf-8')
    print("  ✓ PD 저장 완료")
    
    pa_train.to_csv("datasets/pa/train.csv", index=False, encoding='utf-8')
    pa_val.to_csv("datasets/pa/val.csv", index=False, encoding='utf-8')
    pa_test.to_csv("datasets/pa/test.csv", index=False, encoding='utf-8')
    print("  ✓ PA 저장 완료")
    
    sa_train.to_csv("datasets/sa/train.csv", index=False, encoding='utf-8')
    sa_val.to_csv("datasets/sa/val.csv", index=False, encoding='utf-8')
    sa_test.to_csv("datasets/sa/test.csv", index=False, encoding='utf-8')
    print("  ✓ SA 저장 완료")
    
    # 7. 검증
    print("\n[Step 7] 검증: 같은 (책, 식별자)로 연결되는지 확인...")
    
    if len(pd_test) > 0:
        sample_row = pd_test.iloc[0]
        sample_book = sample_row['book_name']
        sample_para_id = sample_row['문단식별자']
        
        pd_count = len(pd_test[(pd_test['book_name'] == sample_book) & 
                               (pd_test['문단식별자'] == sample_para_id)])
        pa_count = len(pa_test[(pa_test['book_name'] == sample_book) & 
                               (pa_test['문단식별자'] == sample_para_id)])
        
        print(f"\n  검증 - {sample_book}, 문단ID: {sample_para_id}")
        print(f"    ✓ PD에서: {pd_count}행")
        print(f"    ✓ PA에서: {pa_count}행")
        
        if pd_count > 0 and pa_count > 0:
            # PA의 문장IDs로 SA에서 찾기
            sample_sent_ids = set(pa_test[(pa_test['book_name'] == sample_book) & 
                                          (pa_test['문단식별자'] == sample_para_id)]['문장식별자'].unique())
            sa_count = len(sa_test[(sa_test['book_name'] == sample_book) & 
                                   (sa_test['문장식별자'].isin(sample_sent_ids))])
            print(f"    ✓ SA에서: {sa_count}행")
            
            if pd_count > 0 and pa_count > 0 and sa_count > 0:
                print("\n  ✓✓✓ 완벽합니다! 같은 (책, 식별자)로 올바르게 연결되었습니다!")
    
    print("\n" + "=" * 80)
    print("✓ 올바른 7:2:1 분할 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  datasets/pd/: train.csv, val.csv, test.csv")
    print("  datasets/pa/: train.csv, val.csv, test.csv")
    print("  datasets/sa/: train.csv, val.csv, test.csv")
    print("\n설명:")
    print("  1. 44책을 합칠 때 book_name 컬럼 추가")
    print("  2. (book_name, 문단식별자) 조합을 기준으로 7:2:1 분할")
    print("  3. PA와 SA는 해당 (책, ID)를 가진 모든 행 포함")
    print(f"\n결과:")
    print(f"  - 학습(Train):  {len(train_para_keys)}개 (책, 문단)")
    print(f"  - 검증(Val):    {len(val_para_keys)}개 (책, 문단)")
    print(f"  - 테스트(Test):  {len(test_para_keys)}개 (책, 문단)")

if __name__ == "__main__":
    main()
