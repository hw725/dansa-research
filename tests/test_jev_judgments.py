"""Jev 실행기·패널 생성기를 가짜 클라이언트와 합성 문항으로 잰다(실호출 없음).

대부분의 테스트는 합성 문항을 쓰므로 공개 클론에서도 돈다. 실제 번역문이 필요한 것
(질문에 문장이 실리는지 등)만 raw 판정 CSV가 있을 때 돈다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compute_final_stats as cfs  # noqa: E402
import compute_jev_agreement as cja  # noqa: E402
import compute_judge_panels as cjp  # noqa: E402
import jev_client  # noqa: E402
import run_jev_judgments as rj  # noqa: E402

RAW = rj.REPO / "results" / "gpt5mini" / "section3_judgments.csv"
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="raw 판정 CSV 없음(로컬 전용)")


def fake_items(n: int) -> list[dict]:
    out = []
    for i in range(n):
        target = i % 2 == 0
        marker = "하나니라" if target else "라(대조군)"
        out.append({"key": ["책", str(i), str(i), marker], "marker_type": marker, "is_target": target,
                    "text": f"군자 문장 {i}" if target else f"문장 {i}", "votes": 3 if target else 0,
                    "per_model": {}})
    return out


class FakeClient:
    """«군자»가 든 문장에 0.9, 아니면 0.2. errors에 준 예외를 차례로 먼저 낸다. 호출 수·사용량을 센다."""

    model = jev_client.TYPESAFE_MODEL

    def __init__(self, errors=None, value=None):
        self.calls_made = 0
        self.errors = list(errors or [])
        self.value = value

    def ask(self, state, questions):
        self.calls_made += 1
        if self.errors:
            raise self.errors.pop(0)
        text = state + " ".join(q["instructions"] for q in questions.values())
        v = self.value if self.value is not None else (0.9 if "군자" in text else 0.2)
        return {qid: {"noul": v} for qid in questions}

    def usage(self):
        return {"calls": self.calls_made}


def http(status, detail=""):
    return jev_client.JevCallFailed("jev_http_error", status=status, detail=detail)


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    """합성 문항 10개(target·control 번갈아) · 출력은 tmp · 잠은 건너뛴다."""
    monkeypatch.setattr(rj, "OUT_DIR", tmp_path)
    monkeypatch.setattr(rj, "load_items", lambda section: fake_items(10))
    monkeypatch.setattr(rj.time, "sleep", lambda s: None)
    return tmp_path


def records(section="section3", order_name="single"):
    recs, bad = rj.read_jsonl(rj.out_path(section, "pos", jev_client.TYPESAFE_MODEL, order_name))
    return recs


# ── 상한(게이트) ─────────────────────────────────────────────────────────────

def test_gate_refuses_before_any_call(synth):
    c = FakeClient()
    with pytest.raises(jev_client.JevGateExceeded):
        rj.run("section3", "pos", None, 20, c, budget=rj.Budget(5))  # single: 10회 계획 > 5
    assert c.calls_made == 0
    assert not rj.out_path("section3", "pos").exists()


def test_budget_is_shared_across_sections(synth):
    budget = rj.Budget(14)
    rj.run("section3", "pos", None, 20, FakeClient(), budget=budget)  # 10회
    c = FakeClient()
    with pytest.raises(jev_client.JevGateExceeded):  # 다음 섹션 10회 > 남은 4
        rj.run("section1", "pos", None, 20, c, budget=budget)
    assert c.calls_made == 0 and budget.remaining == 4


def test_retries_count_against_budget(synth):
    budget = rj.Budget(100)
    c = FakeClient(errors=[http(429), http(529)])
    r = rj.run("section3", "pos", 1, 20, c, budget=budget)
    assert r["answered"] == 1 and c.calls_made == 3 and budget.used == 3


def test_gate_mid_run_stops_and_keeps_usage(synth):
    c = FakeClient(errors=[http(429), http(429)])  # 재시도 2번이 상한을 먹는다
    r = rj.run("section3", "pos", None, 20, c, budget=rj.Budget(10))
    assert r["stopped"] == "gate"
    assert r["usage"]["calls"] == 10 and c.calls_made == 10
    assert r["answered"] == 8 and len(records()) == 8


def test_which_errors_are_retried(synth):
    assert rj.is_rate_limited(http(401, 'Solar HTTP 429: {"error": "limit"}'))
    assert not rj.is_rate_limited(http(401, "invalid api key"))
    assert rj.is_transient(http(520)) and rj.is_transient(jev_client.JevCallFailed("jev_transport_error"))
    c = FakeClient(errors=[http(401, "invalid api key")])  # 진짜 인증 실패는 다시 보내지 않는다
    r = rj.run("section3", "pos", 1, 20, c)
    assert c.calls_made == 1 and r["answered"] == 0 and "[401]" in r["errors"][0]
    c = FakeClient(errors=[http(520)])
    assert rj.run("section3", "pos", 1, 20, c, budget=rj.Budget(5))["answered"] == 1  # 새 실행: 1번 문항


def test_cooldown_rechecks_after_sleep(synth, monkeypatch):
    """자는 동안 다른 스레드가 쉬는 시각을 늘렸으면 다시 잔다."""
    slept = []
    clock = {"t": 0.0}
    monkeypatch.setattr(rj.time, "time", lambda: clock["t"])

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(rj.time, "sleep", sleep)
    c = FakeClient(errors=[http(429)])
    rj.run("section3", "pos", 1, 20, c)
    assert slept and slept[0] == rj.RATE_WAIT_START  # 429 뒤 공유 쉬는 시각만큼 잤다


# ── 기록·이어 돌리기 ─────────────────────────────────────────────────────────

def test_resume_skips_done_and_survives_truncated_line(synth):
    rj.run("section3", "pos", 4, 20, FakeClient())
    path = rj.out_path("section3", "pos")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": ["책", "9", "9", "라(대')  # 강제 종료로 잘린 줄
    assert len(rj.done_keys(path)) == 4
    c = FakeClient()
    r = rj.run("section3", "pos", None, 20, c)
    assert r["asked"] == 6 and c.calls_made == 6
    recs, bad = rj.read_jsonl(path)
    assert len(recs) == 10 and bad == 1


def test_records_keep_raw_probability_and_no_text(synth):
    rj.run("section3", "pos", 1, 20, FakeClient(value=0.49996))
    rec = records()[0]
    assert rec["noul"] == 0.49996  # 반올림하면 0.5가 되어 O로 뒤집힌다
    assert set(rec) == {"key", "marker_type", "noul", "model", "batch"}  # 번역문은 남기지 않는다


def test_usage_is_per_run_not_cumulative(synth):
    c = FakeClient()
    r1 = rj.run("section3", "pos", 3, 20, c)
    r2 = rj.run("section1", "pos", 4, 20, c)
    assert r1["usage"]["calls"] == 3 and r2["usage"]["calls"] == 4


def test_single_is_default_and_orders_are_checked(synth):
    assert rj.out_path("section3", "pos").parent.name == "single"
    assert rj.plan("section3", "pos", None, 20)["batch_size"] == 1
    with pytest.raises(ValueError):
        rj.out_path("section3", "pos", order_name="singel")


def test_paper_batches_keep_arms_apart(synth):
    p = rj.plan("section3", "pos", None, 3, order_name="paper")
    assert all(len({it["is_target"] for it in b}) == 1 for b in p["batches"])


# ── 병렬 ─────────────────────────────────────────────────────────────────────

def test_workers_need_factory(synth):
    with pytest.raises(ValueError):
        rj.run("section3", "pos", None, 20, FakeClient(), workers=3)


def test_parallel_run_uses_per_thread_clients(synth):
    made = []

    def factory():
        made.append(FakeClient())
        return made[-1]

    r = rj.run("section3", "pos", None, 20, FakeClient(), workers=3, factory=factory, budget=rj.Budget(10))
    assert r["answered"] == 10 and r["usage"]["calls"] == 10
    assert sum(c.calls_made for c in made) == 10 and len(records()) == 10


# ── 패널 생성기 ───────────────────────────────────────────────────────────────

def test_panel_state_restores_on_error(monkeypatch):
    monkeypatch.setattr(cjp.rj, "_load", lambda sec, need_text: [])
    before = (cfs.MODELS, cfs.SECTIONS, cfs.load_section_rows)
    loader = cjp.Loader(0.5, "single")
    loader.install()
    try:
        with pytest.raises(RuntimeError):
            with cjp.panel_state({"jev": "Jev"}, {}, loader, {"section3": set()}):
                raise RuntimeError("boom")
        assert (cfs.MODELS, cfs.SECTIONS) == before[:2] and loader.only == {}
    finally:
        loader.restore()
    assert cfs.load_section_rows is before[2]


def test_panel_rejects_unknown_order():
    with pytest.raises(ValueError):
        cjp.Loader(0.5, "singel")


def test_kappa_auc_basics():
    assert cja.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cja.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert cja.auc([0.5], [0.5]) == 0.5


# ── 실제 번역문이 필요한 것 ───────────────────────────────────────────────────

@needs_raw
def test_question_quotes_sentence_and_state_has_no_marker():
    items = rj.order(rj.load_items("section3"))[:3]
    state, qs = rj.build_call("section3", "pos", items)
    assert len(qs) == 3
    for i, it in enumerate(items):
        assert it["text"] in qs[f"q{i + 1}"]["instructions"]
    single_state, single_q = rj.build_call("section3", "pos", items[:1])
    assert items[0]["text"] in single_state and list(single_q) == ["q1"]
    for word in ("하나니라", "汎論", "marker"):  # 현토 표지·범주는 모델에 가지 않는다
        assert word not in state and word not in single_state


@needs_raw
def test_paper_order_follows_three_model_processing_order():
    a = rj.load_items("section3")
    assert [x["key"] for x in a] == [x["key"] for x in rj.load_items("section3")]
    assert [x["key"] for x in rj.order(a)[:50]] == [x["key"] for x in rj.order(rj.load_items("section3"))[:50]]
