"""entrypoint.py — 재현 샌드박스 실행 진입점.

흐름: 무결성 검증 → 벤더 preflight → 재실행 판정 → 드리프트 → 문장별 열람.

환경변수:
  OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY  사용할 벤더 키(필수 ≥1)
  VIEW_CAP=30   문장별 열람에서 모델·섹션당 표시 상한. 0이면 식별자만 표시.
  SHOW_IDS="book/문단/문장,..."   특정 문장만 골라 열람(상한은 그대로 적용)
  SANDBOX_MOCK=1   네트워크 없이 결정적 가짜 판정
"""
from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "sandbox"))
sys.path.insert(0, str(APP / "scripts"))

import verify_corpus  # noqa: E402
import vendor_clients  # noqa: E402
import run_a_run  # noqa: E402
import compare_run_drift  # noqa: E402


def banner() -> None:
    print("=" * 64)
    print(" dansa-research 재현 샌드박스")
    print(" 입력 무결성을 확인하고 동결 기준 결과와 비교합니다.")
    print("=" * 64)


def _load_translations() -> dict[tuple[str, str, str], str]:
    """마운트된 로컬 입력에서 (book,문단,문장)->번역문 룩업."""
    import run_multimodel_judgments as rmj
    import pandas as pd
    lut: dict[tuple[str, str, str], str] = {}
    df = pd.read_csv(rmj.DATA_FILE,
                     usecols=lambda c: c in ("book", "문단식별자", "문장식별자", "번역문"))
    for b, p, s, t in zip(df["book"], df["문단식별자"], df["문장식별자"], df["번역문"]):
        lut[(str(b), str(p), str(s))] = "" if t is None else str(t)
    return lut


def per_sentence_view() -> None:
    """문장별 판정 미리보기를 표시한다."""
    cap = int(os.environ.get("VIEW_CAP", "30"))
    a_run_dir = run_a_run.A_RUN_DIR
    import run_multimodel_judgments as rmj

    print("\n## 문장별 판정 열람: 재실행 결과")
    if cap <= 0:
        print("  VIEW_CAP=0 — 식별자·O/X 만 표시. 전수 요약은 drift_report.json.")
    else:
        print(f"  표시 상한 {cap}건/모델·섹션. 전수 요약은 drift_report.json.")

    lut: dict = {}
    if cap > 0:
        try:
            lut = _load_translations()
        except Exception as exc:  # noqa: BLE001
            print(f"  (번역문 룩업 불가: {type(exc).__name__} — 식별자만 표시)")

    sel = set(x for x in os.environ.get("SHOW_IDS", "").split(",") if x)

    if not a_run_dir.exists():
        print("  (재실행 산출물 없음)")
        return
    for model_dir in sorted(p for p in a_run_dir.iterdir() if p.is_dir()):
        for sk in ("section1", "section2", "section3"):
            f = model_dir / rmj.SECTIONS[sk]["csv"]
            if not f.exists():
                continue
            with open(f, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if sel:
                rows = [r for r in rows
                        if f"{r.get('book','')}/{r.get('문단식별자','')}/{r.get('문장식별자','')}" in sel]
            pick = rows[:cap]
            if not pick:
                continue
            print(f"\n[{model_dir.name} · {sk}] {len(pick)}건 표시")
            for r in pick:
                ox = "O" if str(r.get("llm_judgment", "")).lower() in ("true", "1", "o") else "X"
                ident = f"{r.get('book','')}/{r.get('문단식별자','')}/{r.get('문장식별자','')}"
                if cap > 0 and lut:
                    t = lut.get((r.get("book", ""), r.get("문단식별자", ""), r.get("문장식별자", "")), "")
                    t = (t[:80] + "…") if len(t) > 80 else t
                    print(f"   {ox}  [{r.get('marker_type','')}]  {t}")
                else:
                    print(f"   {ox}  [{r.get('marker_type','')}]  {ident}")


def main() -> int:
    banner()

    print("\n[1/4] 입력 무결성 검증")
    rc = verify_corpus.main()
    if rc != 0:
        print("중단: 입력이 동결 기준과 다릅니다.")
        return rc

    print("\n[2/4] 벤더 preflight")
    models = vendor_clients.available_models()
    if not models:
        print("중단: 사용할 키 없음 — OPENAI/ANTHROPIC/GEMINI 중 최소 1개 필요")
        return 1
    pf = asyncio.run(vendor_clients.preflight(models))
    for mk, ok, detail in pf:
        print(f"   {vendor_clients.SANDBOX_MODELS[mk]['display']:>16}: "
              f"{'OK' if ok else 'FAIL'}  {detail}")
    if any(not ok for _, ok, _ in pf):
        print("중단: 위 모델 호출 실패 — 키·모델ID·엔드포인트를 확인하세요.")
        return 1

    print("\n[3/4] 재실행 판정 수행")
    rc = run_a_run.main()
    if rc != 0:
        return rc

    print("\n[4/4] 드리프트 리포트")
    rc = compare_run_drift.main()
    if rc != 0:
        return rc

    try:
        per_sentence_view()
    except Exception as exc:  # noqa: BLE001  — 열람 실패가 전체를 깨지 않게
        print(f"\n(문장별 열람 생략: {type(exc).__name__})")

    print("\n완료. 재실행 산출물은 컨테이너 임시 공간에 저장되었습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
