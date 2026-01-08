#!/usr/bin/env python3
"""Train an interpretable classifier to predict boundary cluster IDs and induce human-readable labels.

Goal
- Not "ground-truth" semantic labels, but inductive, explainable descriptions per cluster.
- Uses a linear model + TF-IDF features to make the model weights interpretable.

Outputs
- Markdown report with per-cluster: top weighted features (vector->words), representative examples,
  and heuristic label guess.

Designed to run inside docker-compose `csp` container.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class LabelGuess:
    label: str
    reason: str


_RE_ATTR = re.compile(r"^(?:子曰|有子曰|孟子曰|程子曰|范氏曰|何氏曰|謝氏曰|史記世家曰|又曰)$")
_RE_SAYS = re.compile(r"曰$")
_RE_DEFINE = re.compile(r"^[^\s]{1,10}(?:은|는)\s+.+(?:也|矣|라)[^\s]*$")
_RE_YEAR = re.compile(r"[一二三四五六七八九十百千0-9]{1,4}年")


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s == "nan" else s


def _join_cols(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = _safe_str(row.get(c))
        if v:
            parts.append(v)
    return " \n ".join(parts)


def guess_label_from_examples(examples: pd.DataFrame) -> LabelGuess:
    left = examples["src_left"].astype(str).map(_safe_str)
    right = examples["src_right"].astype(str).map(_safe_str)
    n = max(1, len(examples))

    frac_attr = float(left.map(lambda s: bool(_RE_ATTR.match(s.strip()))).sum()) / n
    frac_says = float(left.map(lambda s: bool(_RE_SAYS.search(s.strip()))).sum()) / n
    frac_define = float(left.map(lambda s: bool(_RE_DEFINE.match(s.strip()))).sum()) / n
    frac_year = float((left + " " + right).map(lambda s: bool(_RE_YEAR.search(s))).sum()) / n

    if frac_attr >= 0.35 or frac_says >= 0.55:
        return LabelGuess(
            label="발화/인용 도입(…曰)",
            reason=f"src_left '曰' 패턴 다수(attr={frac_attr:.2f}, says={frac_says:.2f})",
        )
    if frac_define >= 0.25:
        return LabelGuess(
            label="용어 풀이/주석 정의(…은/…는 …也/…라)",
            reason=f"정의문 형태 다수(define={frac_define:.2f})",
        )
    if frac_year >= 0.15:
        return LabelGuess(
            label="연표/사건 서술(연도 포함)",
            reason=f"연도 표기 포함 비율 높음(year={frac_year:.2f})",
        )
    return LabelGuess(label="기타/혼합", reason="규칙 기반 패턴이 지배적이지 않음")


def _format_example_rows(df: pd.DataFrame, k: int) -> str:
    lines: list[str] = []
    for _, r in df.head(int(k)).iterrows():
        lines.append(f"- book={r.get('book_name','')}, para={r.get('paragraph_id','')}, sent={r.get('left_sentence_id','')}→{r.get('right_sentence_id','')}")
        lines.append(f"  - src_L: {_safe_str(r.get('src_left'))}")
        lines.append(f"  - src_R: {_safe_str(r.get('src_right'))}")
    return "\n".join(lines)


def _top_features_for_class(coef: np.ndarray, feature_names: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    idx = np.argsort(coef)[::-1][: int(top_k)]
    return [(str(feature_names[i]), float(coef[i])) for i in idx]


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train interpretable labeler for boundary clusters")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("hyeonto/reports/recluster_k16_child_minper50/reclustered.csv"),
        help="Input reclustered.csv (parent_cluster_id, child_cluster_id, src_left, src_right, ...)",
    )
    ap.add_argument(
        "--target",
        choices=["parent", "parent_child"],
        default="parent",
        help="Which label to predict",
    )
    ap.add_argument("--text-cols", type=str, default="src_left,src_right", help="Comma-separated columns to use as text")
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-group", type=int, default=50, help="Skip labels with fewer than this many rows")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--ngram-max", type=int, default=3)
    ap.add_argument("--top-features", type=int, default=25)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hyeonto/reports/k16_analysis_minper50/labeler"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"parent_cluster_id", "child_cluster_id", "src_left", "src_right"}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise SystemExit(f"입력 CSV에 필요한 컬럼이 없습니다: {missing}")

    text_cols = [c.strip() for c in str(args.text_cols).split(",") if c.strip()]

    if args.target == "parent":
        y_raw = df["parent_cluster_id"].astype(int)
        label_name = "parent"
    else:
        # stable composite id
        y_raw = df["parent_cluster_id"].astype(int).astype(str) + "_" + df["child_cluster_id"].astype(int).astype(str)
        label_name = "parent_child"

    # filter low-count labels (mostly to avoid tiny classes dominating confusion)
    vc = y_raw.value_counts()
    keep_labels = set(vc[vc >= int(args.min_group)].index.tolist())
    keep_mask = y_raw.isin(keep_labels)
    df2 = df.loc[keep_mask].copy()
    y2 = y_raw.loc[keep_mask].copy()

    if df2.empty:
        raise SystemExit("필터 후 데이터가 비었습니다. --min-group 값을 낮추세요.")

    texts = df2.apply(lambda r: _join_cols(r, text_cols), axis=1).tolist()

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        texts,
        y2,
        df2,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y2,
    )

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, int(args.ngram_max)),
        min_df=2,
        max_features=int(args.max_features),
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)
    acc = float(accuracy_score(y_test, pred))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics
    labels_sorted = sorted(list(keep_labels), key=lambda x: (str(x)))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    _write_json(out_dir / f"metrics_{label_name}.json", {"accuracy": acc, "report": report})
    np.save(out_dir / f"confusion_{label_name}.npy", cm)

    # interpretability
    feature_names = np.array(vec.get_feature_names_out())

    # for representative examples, use decision scores
    scores = clf.decision_function(Xte)
    # binary vs multiclass normalization
    if scores.ndim == 1:
        scores = scores[:, None]

    classes = clf.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}

    md_lines: list[str] = []
    md_lines.append(f"# boundary cluster labeler ({label_name})\n")
    md_lines.append(f"- source: {args.csv}")
    md_lines.append(f"- target: {label_name}")
    md_lines.append(f"- keep_labels(min_group={int(args.min_group)}): {len(keep_labels)}")
    md_lines.append(f"- test_size: {float(args.test_size):.2f}")
    md_lines.append(f"- accuracy: {acc:.4f}\n")

    # per cluster: top features + top examples + heuristic label guess
    # Use train-set slices for heuristic; use test-set scores for representativeness
    df_test_local = df_test.copy()
    df_test_local["_y_true"] = list(y_test)

    for c in classes:
        # train slice for heuristic & examples
        train_slice = df_train.loc[y_train == c]
        if train_slice.empty:
            continue

        # heuristic guess
        if label_name == "parent":
            ex_for_guess = train_slice[["src_left", "src_right", "book_name", "paragraph_id", "left_sentence_id", "right_sentence_id"]].head(200)
            guess = guess_label_from_examples(ex_for_guess)
            header = f"## {label_name} {int(c)} (n={len(train_slice)})"
        else:
            header = f"## {label_name} {str(c)} (n={len(train_slice)})"
            ex_for_guess = train_slice[["src_left", "src_right", "book_name", "paragraph_id", "left_sentence_id", "right_sentence_id"]].head(200)
            # reuse same heuristic but will be noisier for tiny subtypes
            guess = guess_label_from_examples(ex_for_guess)

        md_lines.append(header)
        md_lines.append(f"- guess: **{guess.label}**")
        md_lines.append(f"- reason: {guess.reason}")

        # top features from weight vector
        i = class_to_idx[c]
        coef = clf.coef_[i]
        top_feats = _top_features_for_class(coef, feature_names, top_k=int(args.top_features))
        md_lines.append("- top_features:")
        md_lines.append("  - " + ", ".join([f"{t}" for t, _ in top_feats[:15]]))

        # representative test examples: highest margin for this class
        if i < scores.shape[1]:
            class_scores = scores[:, i]
            # select examples predicted or true? we use predicted-as-this-class for clarity
            pred_mask = (pred == c)
            idxs = np.argsort(class_scores[pred_mask])[::-1]
            # map back indices in df_test_local
            test_rows = df_test_local.loc[pred_mask].copy()
            if not test_rows.empty:
                test_rows["_score"] = class_scores[pred_mask]
                test_rows = test_rows.sort_values("_score", ascending=False)
                md_lines.append("")
                md_lines.append("- representative_examples:")
                md_lines.append(_format_example_rows(test_rows, k=int(args.examples)))
        md_lines.append("")

    out_md = out_dir / f"labeler_{label_name}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {out_md}")

    # also write a compact CSV for downstream manual editing
    rows = []
    for c in classes:
        i = class_to_idx[c]
        coef = clf.coef_[i]
        top_feats = _top_features_for_class(coef, feature_names, top_k=int(args.top_features))
        label = str(c)
        train_n = int((y_train == c).sum())
        guess = guess_label_from_examples(df_train.loc[y_train == c].head(200))
        rows.append(
            {
                "target": label_name,
                "cluster": label,
                "n_train": train_n,
                "guess": guess.label,
                "reason": guess.reason,
                "top_features": " ".join([t for t, _ in top_feats[: int(args.top_features)]]),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / f"labeler_{label_name}.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
