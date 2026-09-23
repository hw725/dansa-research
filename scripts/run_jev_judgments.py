#!/usr/bin/env python3
"""판정 모델 Jev로 3모델과 같은 斷辭 문항을 판정한다 — 네 번째 판정자.

왜:
    3모델(gpt-5-mini·gemini·claude-sonnet)은 생성 모델이라 «1. O»처럼 글로 답하고
    파서가 그 글에서 O/X를 건진다(`parse_ox_response` — 번호가 안 맞으면 X로 떨어진다).
    Jev(TypeSafe System One)는 문항마다 **0~1 확률 하나**만 돌려준다. 그래서
    ① 파서 사고가 없고 ② 문턱을 바꿔 가며 다시 부르지 않고 잴 수 있다.
    classical-text-browser에서 «행에서 새 글이 시작하는가»를 같은 방식으로 물었다
    (docs/sessions/session_jev_structure.md). 여기서는 같은 모델에 3모델과 **같은 정의**를 묻는다.

무엇을 묻는가:
    섹션마다 `run_multimodel_judgments.py`의 프롬프트 정의를 그대로 noul 질문으로 옮긴다.
    3모델이 번역문만 봤으므로 Jev도 번역문만 본다(원문·현토 표지는 주지 않는다 —
    표지를 주면 «니라면 O»처럼 표지로 답할 길이 열린다).

    --mode flip 은 **반대 틀**로 묻는 대조군이다(예: «일반론인가» 대신 «특정 사건·인물에 대한
    진술인가»). 글을 읽고 답하면 p(정) + p(반) ≈ 1 이고, 「예」로 기우는 모델이면 둘 다 높다.
    classical-text-browser의 «남의 후보를 주면 없음을 고르는가»와 같은 자리다 — 무작위 기준선
    없이 일치율만 보면 «읽는가, 기우는가»를 가를 수 없다.

묶는 방식(--order):
    single(기본) 한 요청에 한 문장·한 질문. 논문 산출물(results/jev/panels)은 이것으로 냈다.
                 20개씩 묶으면 생성형(3모델·Solar 래퍼)은 문항끼리 서로 보고 Jev는 옆 문장을
                 state로 봐서 판정 모델마다 조건이 달라진다 — 한 문장씩이 같은 조건이다.
    paper        3모델처럼 20개씩, target과 control을 따로 묶는다.
    mixed        target·control을 섞어(seed) 20개씩 묶는다. 2026-09-23 첫 시험(대체됨).

    noul의 criteria에 «false» 설명이 필요하다. 원 프롬프트는 «아니면 X»뿐이라 정의의 부정을
    적었다(아래 SECTION_Q). 내용을 더하지 않도록 정의 문구만 뒤집었다.

게이트(전역 규칙 11):
    실행 전 `size`로 호출 수·글자 수를 센다. `run`은 계획 호출이 --max-calls를 넘으면
    **한 건도 쏘지 않고** 거부한다. 결과는 문항마다 JSONL에 바로 붙여 쓰고(재시작 시 이어감),
    번역문은 저장하지 않는다 — 키(book·문단·문장·marker_type)와 확률만 남긴다.

사용:
    py scripts/run_jev_judgments.py size --sections section3,section1,section2
    py scripts/run_jev_judgments.py run --sections section3,section1,section2 --workers 2 --max-calls 60000
    py scripts/run_jev_judgments.py run --model solar-mini4-jev --sections section3 ...
    py scripts/compute_judge_panels.py         # 논문 산출물을 판정자 구성별로(호출 0건)
    (첫 시험: --order mixed, --mode flip, 채점 compute_jev_agreement.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_final_stats as cfs  # noqa: E402
import jev_client  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "jev"
LOG = REPO / "logs" / "jev_judgments.jsonl"
SEED = 20260923
DEFAULT_BATCH = 20

# 섹션별 질문. pos는 run_multimodel_judgments.py의 정의를 옮긴 것, neg는 그 반대 틀(flip 대조군).
SECTION_Q = {
    "section1": {
        "pos": {
            "q": "다음 번역문이 가벼운 감탄이나 여운을 남기며 마무리하는 뉘앙스입니까?",
            "true": "가벼운 감탄·여운으로 마무리한다 — 무거운 논단이나 엄격한 결론이 아닌 가벼운 "
                    "어감의 마무리이고, 정서적 고양이 동반된다",
            "false": "가벼운 감탄·여운의 마무리가 아니다",
        },
        "neg": {
            "q": "다음 번역문이 감탄·여운 없이, 무거운 논단이나 엄격한 결론으로 담담하게 "
                 "마무리합니까?",
            "true": "감탄·여운이 없다 — 논단·결론·서술로 담담하게 끝맺는다",
            "false": "가벼운 감탄이나 여운을 남기며, 정서적 고양이 동반된 마무리다",
        },
    },
    "section2": {
        "pos": {
            "q": "다음 번역문에서 화자가 행동이나 태도를 분명하게 정하며 마무리합니까?",
            "true": "화자가 입장·판단·방침을 분명하게 확정하는 마침이다 — 분명한 결론이나 판단을 "
                    "내리며 끝맺는 진술이다",
            "false": "행동이나 태도를 분명하게 정하며 마무리하지 않는다",
        },
        "neg": {
            "q": "다음 번역문에서 화자가 입장·판단·방침을 확정하지 않은 채 마무리합니까?",
            "true": "입장·판단을 확정하지 않는다 — 사실을 전하거나 묘사하거나 유보하며 끝난다",
            "false": "화자가 입장·판단·방침을 분명하게 확정하며 끝맺는다",
        },
    },
    "section3": {
        "pos": {
            "q": "다음 번역문이 두루 진술하는 내용입니까?",
            "true": "특정 사건·인물이 아닌 일반론을 서술한다 — 개별 상황이 아닌 통론적 진술이다",
            "false": "두루 진술하는 내용이 아니다",
        },
        "neg": {
            "q": "다음 번역문이 특정 사건·인물이나 개별 상황에 대한 진술입니까?",
            "true": "특정 사건·인물·개별 상황을 서술한다",
            "false": "특정 사건·인물이 아닌 일반론·통론적 진술이다",
        },
    },
}
STATE_HEAD = (
    "한국고전번역원 국역본에서 뽑은 번역문 {n}개입니다. 문장마다 출처가 다르며 서로 이어지지 "
    "않습니다.\n"
)


def load_items(section: str) -> list[dict]:
    """3모델이 모두 판정한 문항. 출력: [{key, marker_type, text, votes, per_model}].

    번역문이 필요하므로 raw CSV를 읽는다(익명본에는 해시만 있다). 채점만 할 때는
    compute_jev_agreement가 need_text=False로 익명본도 쓸 수 있게 따로 부른다.
    """
    return _load(section, need_text=True)


def _load(section: str, need_text: bool) -> list[dict]:
    cfg = cfs.SECTIONS[section]
    prev = cfs.SOURCE
    cfs.SOURCE = "raw" if need_text else "auto"
    try:
        by_model = {
            m: {cfs.row_key(r): r for r in cfs.load_section_rows(m, cfg)} for m in cfs.MODELS
        }
    finally:
        cfs.SOURCE = prev
    common = set.intersection(*(set(v) for v in by_model.values()))
    target, control = str(cfg["target"]), str(cfg["control"])
    # 3모델이 처리한 순서(판정 CSV의 행 순서: 본 표본 → 보충 대조군). 논문 조건으로 묶을 때 쓴다.
    rank = {k: i for i, k in enumerate(by_model["gpt5mini"])}
    items = []
    for key in sorted(common, key=lambda k: rank[k]):
        if key[3] not in (target, control):
            continue
        first = by_model["gpt5mini"][key]
        per_model = {m: cfs.parse_bool(by_model[m][key].get("llm_judgment")) for m in cfs.MODELS}
        items.append(
            {
                "key": list(key),
                "marker_type": key[3],
                "is_target": key[3] == target,
                "text": (first.get("번역문") or "").strip() if need_text else "",
                "votes": sum(per_model.values()),
                "per_model": per_model,
            }
        )
    return items


def order(items: list[dict], seed: int = SEED) -> list[dict]:
    """target·control을 섞는다. 같은 seed면 pos와 flip이 같은 순서·같은 앞 N개를 쓴다."""
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def paper_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    """논문(3모델)과 같은 묶음: target과 control을 **따로**, 처리 순서대로 batch_size씩.
    run_multimodel_judgments.process_batches가 target_todo와 control_todo를 따로 돌린 것과 같다."""
    out = []
    for arm in (True, False):
        group = [it for it in items if it["is_target"] == arm]
        out += [group[i : i + batch_size] for i in range(0, len(group), batch_size)]
    return out


def build_call(section: str, mode: str, batch: list[dict]) -> tuple[str, dict]:
    """묶음 하나 → (state, questions). 질문 id는 모델에 가지 않으므로 문장을 instructions에 다시 적는다."""
    spec = SECTION_Q[section][mode]
    if len(batch) == 1:
        # 한 요청 = 한 문장·한 질문(single). 판정 모델의 설계대로 state에 관찰(그 문장)만 둔다.
        # Jev와 Solar 래퍼(속은 생성형 — 질문들을 한 채팅으로 묶는다)가 같은 조건이 되는 유일한 형태다.
        state = f"한국고전번역원 국역본의 번역문 한 문장입니다.\n「{batch[0]['text']}」"
        return state, {"q1": {"type": "noul", "instructions": spec["q"],
                              "criteria": {"true": spec["true"], "false": spec["false"]}}}
    lines = [f"[{i + 1}] {it['text']}" for i, it in enumerate(batch)]
    state = STATE_HEAD.format(n=len(batch)) + "\n".join(lines)
    questions = {
        f"q{i + 1}": {
            "type": "noul",
            "instructions": f"{spec['q']}\n문장 [{i + 1}]: 「{it['text']}」",
            "criteria": {"true": spec["true"], "false": spec["false"]},
        }
        for i, it in enumerate(batch)
    }
    return state, questions


def out_path(section: str, mode: str, model: str = jev_client.TYPESAFE_MODEL,
             order_name: str = "mixed") -> Path:
    """모델·묶는 방식마다 파일을 가른다 — 한 파일에 섞이면 resume이 «이미 물었다»로 건너뛴다.
    mixed(섞어 묶기, 2026-09-23 첫 시험)는 옛 이름을 지키고, paper(논문 조건)는 paper/ 아래에 둔다."""
    if order_name in ("paper", "single"):
        return OUT_DIR / order_name / f"{section}_{mode}.{model}.jsonl"
    if model == jev_client.TYPESAFE_MODEL:
        return OUT_DIR / f"{section}_{mode}.jsonl"
    return OUT_DIR / f"{section}_{mode}.{model}.jsonl"


# 같은 System One 형식을 받는 판정 모델들. solar-mini4-jev(https://github.com/hunkim/solar-mini4-jev)는
# Upstage Solar Mini4를 Jev 모양으로 감싼 BYOK 서버다. 엔드포인트·단가는 llm_pipeline
# `llm_runtime/jev_decisions.py`의 PROVIDERS(stopword_pipeline이 쓰는 값)와 같게 둔다.
# Jev는 입력만, Solar는 입력+출력이 청구된다.
JUDGES = {
    "jev-latest": {"url": jev_client.TYPESAFE_URL, "keys": jev_client.KEY_NAMES, "env": None,
                   "usd_in_per_m": jev_client.INPUT_USD_PER_M, "usd_out_per_m": 0.0},
    "solar-mini4-jev": {"url": "https://solar-mini4-jev.vercel.app/v1/systemone",
                        "keys": ("UPSTAGE_API_KEY",), "env": None,  # 공용 키 파일(2026-09-23 일원화)
                        "usd_in_per_m": 0.10, "usd_out_per_m": 0.40},
}


def make_client(model: str, max_calls: int) -> jev_client.JevClient:
    spec = JUDGES[model]
    key = jev_client.resolve_key(env_file=spec["env"], names=spec["keys"]) or ""
    return jev_client.JevClient(api_key=key, url=spec["url"], model=model, max_calls=max_calls)


def cost_usd(model: str, usage: dict) -> float:
    """공급자 단가로 다시 셈한다 — JevClient는 Jev 단가(입력만)로 센다."""
    spec = JUDGES[model]
    return round((usage.get("input_tokens", 0) * spec["usd_in_per_m"]
                  + usage.get("output_tokens", 0) * spec["usd_out_per_m"]) / 1_000_000, 6)


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("noul") is not None:
                keys.add(tuple(rec["key"]))
    return keys


def plan(section: str, mode: str, limit: int | None, batch_size: int,
         model: str = jev_client.TYPESAFE_MODEL, order_name: str = "mixed") -> dict:
    """보낼 것을 센다(호출 0건). 출력: todo 문항·호출 수·보낼 글자 수.

    order_name: mixed = target·control을 섞어(seed) 묶는다(첫 시험).
                paper = 3모델과 같게 — target과 control을 따로, 처리 순서대로 묶는다.
    """
    items = load_items(section)
    items = items if order_name in ("paper", "single") else order(items)
    if order_name == "single":
        batch_size = 1
    if limit:
        items = items[:limit]
    done = done_keys(out_path(section, mode, model, order_name))
    todo = [it for it in items if tuple(it["key"]) not in done]
    if order_name == "paper":
        batches = paper_batches(todo, batch_size)
    else:
        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    chars = 0
    for b in batches:
        state, qs = build_call(section, mode, b)
        chars += len(state) + sum(len(q["instructions"]) + len(q["criteria"]["true"])
                                  + len(q["criteria"]["false"]) for q in qs.values())
    return {"section": section, "mode": mode, "order": order_name, "items": len(items),
            "done": len(items) - len(todo), "todo": todo, "batches": batches,
            "calls": len(batches), "chars": chars}


RATE_RETRIES = 8
RATE_WAIT_START = 5.0
RATE_WAIT_MAX = 120.0


def is_rate_limited(e: Exception) -> bool:
    """속도 제한인가. solar-mini4-jev 호스팅 래퍼는 Upstage의 429를 **401로 바꿔** 돌려준다
    (본문 «Solar HTTP 429 … reached your API request limit», 2026-09-23 실측). 401을 전부
    다시 보내면 진짜 인증 실패도 되풀이하므로 본문에 429가 있을 때만 속도 제한으로 본다."""
    status = getattr(e, "status", None)
    detail = getattr(e, "detail", "") or ""
    return status in (429, 529) or (status == 401 and "429" in detail)


class _Budget:
    """여러 스레드가 나눠 쓰는 호출 상한. 넘으면 그 호출은 나가지 않는다."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0
        self.lock = threading.Lock()

    def take(self) -> None:
        with self.lock:
            if self.used >= self.max_calls:
                raise jev_client.JevGateExceeded(f"상한 {self.max_calls}회에 도달했습니다.")
            self.used += 1


def run(section: str, mode: str, limit: int | None, batch_size: int, client,
        order_name: str = "mixed", workers: int = 1, factory=None) -> dict:
    """묶음을 보내고 문항마다 JSONL에 붙여 쓴다. workers > 1이면 factory로 스레드마다
    클라이언트를 따로 만든다(JevClient의 셈은 스레드 안전하지 않다)."""
    model = getattr(client, "model", jev_client.TYPESAFE_MODEL)
    p = plan(section, mode, limit, batch_size, model, order_name)
    client.gate(p["calls"] + getattr(client, "calls_made", 0))  # 상한을 넘으면 여기서 한 건도 쏘지 않고 멈춘다
    path = out_path(section, mode, model, order_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    budget = _Budget(getattr(client, "_max_calls", 10**9) - getattr(client, "calls_made", 0))
    write_lock = threading.Lock()
    local = threading.local()
    clients = [client]
    errors: list[str] = []
    got = 0

    def my_client():
        if workers <= 1 or factory is None:
            return client
        if not hasattr(local, "c"):
            local.c = factory()
            with write_lock:
                clients.append(local.c)
        return local.c

    cool = {"until": 0.0, "wait": RATE_WAIT_START}

    def ask_with_backoff(state: str, questions: dict):
        """속도 제한이면 **모든 스레드가 함께** 쉬었다가 다시 보낸다. 다른 오류는 그대로 올린다."""
        for attempt in range(RATE_RETRIES + 1):
            with write_lock:
                pause = cool["until"] - time.time()
            if pause > 0:
                time.sleep(pause)
            budget.take()
            try:
                answers = my_client().ask(state, questions)
                with write_lock:
                    cool["wait"] = RATE_WAIT_START
                return answers
            except jev_client.JevCallFailed as e:
                if not is_rate_limited(e) or attempt == RATE_RETRIES:
                    raise
                with write_lock:
                    cool["until"] = max(cool["until"], time.time() + cool["wait"])
                    cool["wait"] = min(cool["wait"] * 2, RATE_WAIT_MAX)

    def one(bi: int, batch: list[dict]) -> None:
        nonlocal got
        state, questions = build_call(section, mode, batch)
        try:
            answers = ask_with_backoff(state, questions)
        except jev_client.JevGateExceeded:
            raise
        except Exception as e:  # noqa: BLE001 — 한 묶음이 죽어도 이미 쓴 것은 남는다
            status = getattr(e, "status", None)
            with write_lock:
                detail = " ".join((getattr(e, "detail", "") or "")[:80].split())
                errors.append(f"batch {bi}: {type(e).__name__}: {e}" + (f" [{status}]" if status else "")
                              + (f" {detail}" if detail else ""))
            return
        lines = []
        for i, it in enumerate(batch):
            p_yes = jev_client.noul(answers.get(f"q{i + 1}"))
            lines.append(json.dumps({"key": it["key"], "marker_type": it["marker_type"],
                                     "noul": None if p_yes is None else round(p_yes, 4),
                                     "model": model, "batch": bi}, ensure_ascii=False))
        with write_lock:
            got += sum(1 for ln in lines if '"noul": null' not in ln)
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")

    if workers <= 1:
        for bi, batch in enumerate(p["batches"]):
            one(bi, batch)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(one, bi, b) for bi, b in enumerate(p["batches"])]:
                fut.result()

    usage: dict = {}
    for c in clients[1:] if len(clients) > 1 else clients:
        for k, v in (c.usage() if hasattr(c, "usage") else {}).items():
            usage[k] = usage.get(k, 0) + v
    return {"section": section, "mode": mode, "order": order_name, "planned_calls": p["calls"],
            "answered": got, "asked": len(p["todo"]), "errors": errors, "usage": usage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["size", "run"])
    ap.add_argument("--sections", default="section3")
    ap.add_argument("--mode", choices=["pos", "neg", "flip"], default="pos",
                    help="pos=원 정의, flip(=neg)=반대 틀 대조군")
    ap.add_argument("--order", choices=["single", "paper", "mixed"], default="single",
                    help="single=한 요청에 한 문장(기본·두 판정 모델의 같은 조건), "
                         "paper=3모델처럼 20개씩 arm별, mixed=섞어 묶기(첫 시험)")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", choices=list(JUDGES), default=jev_client.TYPESAFE_MODEL)
    a = ap.parse_args(argv)
    mode = "neg" if a.mode == "flip" else a.mode
    sections = [s.strip() for s in a.sections.split(",") if s.strip()]

    if a.cmd == "size":
        for s in sections:
            p = plan(s, mode, a.limit, a.batch, a.model, a.order)
            print(json.dumps({k: p[k] for k in ("section", "mode", "order", "items", "done", "calls", "chars")},
                             ensure_ascii=False))
        return 0

    client = make_client(a.model, a.max_calls)
    if not client.has_key:
        print(f"{a.model} 키가 없습니다({' / '.join(JUDGES[a.model]['keys'])}).", file=sys.stderr)
        return 2
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for s in sections:
        res = run(s, mode, a.limit, a.batch, client, a.order, a.workers,
                  factory=lambda: make_client(a.model, a.max_calls))
        res["model"] = a.model
        res["usage"]["cost_usd"] = cost_usd(a.model, res["usage"])  # 공급자 단가로
        res["ts"] = dt.datetime.now().isoformat(timespec="seconds")
        res["limit"], res["workers"] = a.limit, a.workers
        res["batch"] = 1 if a.order == "single" else a.batch  # 실제로 쓴 묶음 크기
        with LOG.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "errors"} | {"n_errors": len(res["errors"]),
                          "errors_head": res["errors"][:5]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
