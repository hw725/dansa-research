#!/usr/bin/env python3
"""Debug PA translation sentence mismatches vs gold.

Usage:
  python scripts/debug_pa_tgt_mismatch.py \
    --out test_results/pa_strict_boundary_aware.csv \
    --gold datasets/pa/test_100_from_pd.csv \
    --limit 20

Optionally inspect a single paragraph:
  python scripts/debug_pa_tgt_mismatch.py --out ... --gold ... --book-name <name> --pid 14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _norm_text(s: str) -> str:
    # keep semantics; only normalize common invisible differences for display
    return str(s).replace("\r\n", "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["out_vs_gold", "splitter_vs_gold"],
        default="out_vs_gold",
        help="비교 모드: out_vs_gold(기본) 또는 splitter_vs_gold(PD 입력을 splitter로 분할해 gold와 비교)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--pd", default="datasets/pd/test_100.csv", help="splitter_vs_gold 모드에서 사용할 PD 입력")
    p.add_argument("--max-length", type=int, default=150, help="splitter_vs_gold 모드에서 split_target_sentences_advanced(max_length=...)에 전달")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--book-name", default=None)
    p.add_argument("--pid", type=int, default=None)
    args = p.parse_args()

    out_path = Path(args.out)
    gold_path = Path(args.gold)

    df_out = _load(out_path)
    df_gold = _load(gold_path)

    for df in (df_out, df_gold):
        df["book_name"] = df["book_name"].astype(str)
        df["문단식별자"] = df["문단식별자"].astype(int)
        df["번역문"] = df["번역문"].astype(str)

    out_grp = df_out.groupby(["book_name", "문단식별자"])["번역문"].apply(list)
    gold_grp = df_gold.groupby(["book_name", "문단식별자"])["번역문"].apply(list)

    if args.mode == "splitter_vs_gold":
        # processor와 동일하게 pa/ 디렉토리를 sys.path에 넣고, sentence_splitter를 직접 임포트한다.
        import sys
        from pathlib import Path as _Path

        workspace_root = _Path(__file__).resolve().parents[1]
        pa_dir = workspace_root / "pa"
        sys.path.insert(0, str(workspace_root))
        sys.path.insert(0, str(pa_dir))

        from sentence_splitter import split_target_sentences_advanced

        df_pd = _load(Path(args.pd))
        df_pd["book_name"] = df_pd["book_name"].astype(str)
        df_pd["문단식별자"] = df_pd["문단식별자"].astype(int)
        df_pd["번역문"] = df_pd["번역문"].astype(str)

        pd_map = {
            (str(r["book_name"]), int(r["문단식별자"])): str(r["번역문"])
            for _, r in df_pd.iterrows()
        }

        if args.book_name is not None and args.pid is not None:
            key = (str(args.book_name), int(args.pid))
            text = pd_map.get(key)
            gold_list = gold_grp.get(key)
            if text is None or gold_list is None:
                print("key", key)
                print("pd exists", text is not None)
                print("gold exists", gold_list is not None)
                return 0

            pred_list = [s.strip() for s in split_target_sentences_advanced(text, max_length=int(args.max_length))]
            gold_list_norm = [str(x).strip() for x in list(gold_list)]

            print("key", key)
            print("pred_n", len(pred_list))
            print("gold_n", len(gold_list_norm))
            for i, (pseg, gseg) in enumerate(zip(pred_list, gold_list_norm)):
                if _norm_text(pseg) != _norm_text(gseg):
                    print("first diff idx", i)
                    print("PRED:", repr(_norm_text(pseg)[:400]))
                    print("GOL :", repr(_norm_text(gseg)[:400]))
                    break
            else:
                if len(pred_list) != len(gold_list_norm):
                    print("same prefix, len differs")
            return 0

        mismatches: list[tuple[tuple[str, int], str, int | None, int | None]] = []
        exact = 0
        for key, gold_list in gold_grp.items():
            text = pd_map.get(key)
            if text is None:
                mismatches.append((key, "MISSING_PD", len(gold_list), None))
                continue
            pred_list = [s.strip() for s in split_target_sentences_advanced(text, max_length=int(args.max_length))]
            gold_list_norm = [str(x).strip() for x in gold_list]
            if list(map(_norm_text, pred_list)) == list(map(_norm_text, gold_list_norm)):
                exact += 1
            else:
                mismatches.append((key, "DIFF", len(gold_list_norm), len(pred_list)))

        total = len(gold_grp)
        print("translation exact via splitter:", f"{exact}/{total}")
        print("mismatch count", len(mismatches))
        for (book, pid), kind, gn, pn in mismatches[: int(args.limit)]:
            print(f"{kind}: {book}:{pid} gold_n={gn} pred_n={pn}")
        return 0

    if args.book_name is not None and args.pid is not None:
        key = (str(args.book_name), int(args.pid))
        out_list = out_grp.get(key)
        gold_list = gold_grp.get(key)
        print("key", key)
        print("out_n", None if out_list is None else len(out_list))
        print("gold_n", None if gold_list is None else len(gold_list))
        if out_list is None or gold_list is None:
            return 0

        for i, (o, g) in enumerate(zip(out_list, gold_list)):
            if _norm_text(o) != _norm_text(g):
                print("first diff idx", i)
                print("OUT:", repr(_norm_text(o)[:400]))
                print("GOL:", repr(_norm_text(g)[:400]))
                break
        else:
            if len(out_list) != len(gold_list):
                print("same prefix, len differs")

        out_concat = "".join(map(_norm_text, out_list))
        gold_concat = "".join(map(_norm_text, gold_list))
        print("out contains \\n", "\n" in out_concat, "gold contains \\n", "\n" in gold_concat)
        return 0

    mismatches: list[tuple[tuple[str, int], str, int | None, int | None]] = []
    for key, gold_list in gold_grp.items():
        out_list = out_grp.get(key)
        if out_list is None:
            mismatches.append((key, "MISSING_OUT", len(gold_list), None))
            continue
        if list(map(_norm_text, out_list)) != list(map(_norm_text, gold_list)):
            mismatches.append((key, "DIFF", len(gold_list), len(out_list)))

    print("gold groups", len(gold_grp), "out groups", len(out_grp))
    print("mismatch count", len(mismatches))
    for (book, pid), kind, gn, on in mismatches[: int(args.limit)]:
        print(f"{kind}: {book}:{pid} gold_n={gn} out_n={on}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
