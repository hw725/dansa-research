#!/usr/bin/env python3
"""hyeonto/xlsx → hyeonto/datasets(pD/pA/sA) 생성

- 입력: hyeonto/xlsx/{book_name}/{book_name}_{문단|문장|구}병렬.xlsx
- 출력:
  - hyeonto/datasets/pd/{train,val,test}.csv
  - hyeonto/datasets/pa/{train,val,test}.csv
  - hyeonto/datasets/sa/{train,val,test}.csv

분할 규칙:
- PD의 고유 (book_name, 문단식별자) 조합을 기준으로 7:2:1 분할
- PA는 동일 (book_name, 문단식별자)에 속하는 행을 그대로 포함
- SA는 PA에 포함된 (book_name, 문장식별자) 기준으로 포함

실행 예:
  python scripts/hyeonto_build_datasets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

HYEONTO_DIR = WORKSPACE_ROOT / "hyeonto"
XLSX_DIR = HYEONTO_DIR / "xlsx"
OUT_DATASETS_DIR = HYEONTO_DIR / "datasets"


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _collect_book_dirs(xlsx_dir: Path) -> list[Path]:
    return sorted([d for d in xlsx_dir.iterdir() if d.is_dir()])


def _load_parallel(file_suffix: str) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    for book_dir in _collect_book_dirs(XLSX_DIR):
        book_name = book_dir.name
        files = [p for p in book_dir.glob(f"*{file_suffix}.xlsx") if not str(p).endswith(".bak")]
        if not files:
            continue
        df = _read_excel(files[0])
        if df is None or len(df) == 0:
            continue
        df["book_name"] = book_name
        all_dfs.append(df)

    if not all_dfs:
        raise SystemExit(f"XLSX를 찾지 못했습니다: suffix={file_suffix}, dir={XLSX_DIR}")

    return pd.concat(all_dfs, ignore_index=True)


def _write_split(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir, index=False, encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
        return
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build hyeonto datasets from hyeonto/xlsx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "9:1", "7:2:1"],
        help="데이터 분할 방식. all=전체를 train로 사용, 9:1=train/val, 7:2:1=train/val/test",
    )
    args = ap.parse_args()

    if not XLSX_DIR.exists():
        raise SystemExit(f"입력 폴더가 없습니다: {XLSX_DIR} (먼저 hyeonto_build_xlsx.py 실행)")

    df_pd = _load_parallel("_문단병렬")
    df_pa = _load_parallel("_문장병렬")
    df_sa = _load_parallel("_구병렬")

    # 키 생성
    df_pd["para_key"] = df_pd["book_name"].astype(str) + "_Para" + df_pd["문단식별자"].astype(str)
    unique_para_keys = df_pd["para_key"].unique()

    n_paras = len(unique_para_keys)
    if n_paras <= 0:
        raise SystemExit("문단이 0개입니다. 입력 XLSX를 확인하세요.")

    split_mode = str(args.split).strip().lower()
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_paras)

    if split_mode == "all":
        train_keys = set(unique_para_keys)
        val_keys: set[str] = set()
        test_keys: set[str] = set()
    elif split_mode == "9:1":
        train_n = int(round(n_paras * 0.9))
        train_keys = set(unique_para_keys[perm[:train_n]])
        val_keys = set(unique_para_keys[perm[train_n:]])
        test_keys = set()
    else:
        train_n = int(n_paras * 0.7)
        val_n = int(n_paras * 0.2)
        train_keys = set(unique_para_keys[perm[:train_n]])
        val_keys = set(unique_para_keys[perm[train_n : train_n + val_n]])
        test_keys = set(unique_para_keys[perm[train_n + val_n :]])

    # PD
    pd_train = df_pd[df_pd["para_key"].isin(train_keys)].drop(columns=["para_key"])
    pd_val = df_pd[df_pd["para_key"].isin(val_keys)].drop(columns=["para_key"])
    pd_test = df_pd[df_pd["para_key"].isin(test_keys)].drop(columns=["para_key"])

    # PA: PD split key 기준
    df_pa["para_key"] = df_pa["book_name"].astype(str) + "_Para" + df_pa["문단식별자"].astype(str)
    pa_train = df_pa[df_pa["para_key"].isin(train_keys)].drop(columns=["para_key"])
    pa_val = df_pa[df_pa["para_key"].isin(val_keys)].drop(columns=["para_key"])
    pa_test = df_pa[df_pa["para_key"].isin(test_keys)].drop(columns=["para_key"])

    # SA: PA split의 (book_name, 문장식별자) 기준
    df_sa["sent_key"] = df_sa["book_name"].astype(str) + "_Sent" + df_sa["문장식별자"].astype(str)

    train_sent_keys = set(pa_train.apply(lambda r: str(r["book_name"]) + "_Sent" + str(r["문장식별자"]), axis=1))
    val_sent_keys = set(pa_val.apply(lambda r: str(r["book_name"]) + "_Sent" + str(r["문장식별자"]), axis=1))
    test_sent_keys = set(pa_test.apply(lambda r: str(r["book_name"]) + "_Sent" + str(r["문장식별자"]), axis=1))

    sa_train = df_sa[df_sa["sent_key"].isin(train_sent_keys)].drop(columns=["sent_key"])
    sa_val = df_sa[df_sa["sent_key"].isin(val_sent_keys)].drop(columns=["sent_key"])
    sa_test = df_sa[df_sa["sent_key"].isin(test_sent_keys)].drop(columns=["sent_key"])

    # 저장
    (OUT_DATASETS_DIR / "pd").mkdir(parents=True, exist_ok=True)
    (OUT_DATASETS_DIR / "pa").mkdir(parents=True, exist_ok=True)
    (OUT_DATASETS_DIR / "sa").mkdir(parents=True, exist_ok=True)

    pd_cols = [c for c in df_pd.columns if c != "para_key"]
    pa_cols = [c for c in df_pa.columns if c != "para_key"]
    sa_cols = [c for c in df_sa.columns if c != "sent_key"]

    _write_csv(OUT_DATASETS_DIR / "pd" / "train.csv", pd_train, pd_cols)
    _write_csv(OUT_DATASETS_DIR / "pd" / "val.csv", pd_val, pd_cols)
    _write_csv(OUT_DATASETS_DIR / "pd" / "test.csv", pd_test, pd_cols)

    _write_csv(OUT_DATASETS_DIR / "pa" / "train.csv", pa_train, pa_cols)
    _write_csv(OUT_DATASETS_DIR / "pa" / "val.csv", pa_val, pa_cols)
    _write_csv(OUT_DATASETS_DIR / "pa" / "test.csv", pa_test, pa_cols)

    _write_csv(OUT_DATASETS_DIR / "sa" / "train.csv", sa_train, sa_cols)
    _write_csv(OUT_DATASETS_DIR / "sa" / "val.csv", sa_val, sa_cols)
    _write_csv(OUT_DATASETS_DIR / "sa" / "test.csv", sa_test, sa_cols)

    print("✅ wrote:")
    print(f"- {OUT_DATASETS_DIR / 'pd'}")
    print(f"- {OUT_DATASETS_DIR / 'pa'}")
    print(f"- {OUT_DATASETS_DIR / 'sa'}")
    print(f"- split: {args.split}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
