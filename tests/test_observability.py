"""Tests for the offline Langfuse observability stack.

All stdlib + pytest. No network, no keys. Each test points the shim at a
throwaway SQLite file so the demo database is never touched.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from analysis import (  # noqa: E402
    cost_over_time, score_distribution, summary, version_comparison,
)
from pricing import estimate_cost_usd  # noqa: E402
from shim import OfflineLangfuse  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def client(db):
    c = OfflineLangfuse(db)
    yield c
    c.close()


def _seed_two_versions(c: OfflineLangfuse) -> None:
    """v1: expensive+faithful, v2: cheap+slightly less faithful."""
    for version, model, faith in (("v1", "gpt-4o", 0.92),
                                  ("v2", "gpt-4o-mini", 0.85)):
        for _ in range(3):
            tr = c.trace("test-qa", prompt_version=version)
            tr.generation("llm", model=model, input="q", completion="a",
                          input_tokens=1000, output_tokens=500,
                          latency_ms=600)
            tr.score("faithfulness", faith)
            tr.score("toxicity", 0.03)
            tr.end()
    c.flush()


def test_trace_generation_score_roundtrip(client, db):
    tr = client.trace("roundtrip", prompt_version="v1")
    with tr.span("retriever", {"top_k": 5}):
        pass
    tr.generation("llm", model="gpt-4o-mini", input="hello",
                  completion="hi", input_tokens=1000, output_tokens=1000,
                  latency_ms=250)
    tr.score("faithfulness", 0.9, "judge")
    tr.score("toxicity", 0.01)
    tr.end()
    client.flush()

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM traces").fetchone()
        assert row["model"] == "gpt-4o-mini"
        assert row["total_tokens"] == 2000
        assert row["cost_usd"] == estimate_cost_usd("gpt-4o-mini", 1000, 1000)
        assert row["latency_ms"] == 250
        scores = {r["name"]: r["value"] for r in
                  conn.execute("SELECT name, value FROM scores")}
        assert scores == {"faithfulness": 0.9, "toxicity": 0.01}
        spans = conn.execute("SELECT name FROM spans").fetchall()
        assert {s["name"] for s in spans} == {"retriever", "llm"}


def test_pricing_math_and_unknown_model():
    assert estimate_cost_usd("gpt-4o-mini", 1000, 1000) == 0.00075
    # unknown model falls back to DEFAULT_RATE, never raises
    assert estimate_cost_usd("future-model-9000", 1000, 1000) > 0
    with pytest.raises(ValueError):
        estimate_cost_usd("gpt-4o-mini", -1, 0)


def test_version_comparison_tradeoff(client, db):
    _seed_two_versions(client)
    cmp = version_comparison(db)
    assert set(cmp) == {"v1", "v2"}
    assert cmp["v2"]["avg_cost_usd"] < cmp["v1"]["avg_cost_usd"]
    assert cmp["v1"]["avg_faithfulness"] > cmp["v2"]["avg_faithfulness"]
    assert cmp["v1"]["n"] == cmp["v2"]["n"] == 3


def test_score_distribution(client, db):
    _seed_two_versions(client)
    d = score_distribution(db, "faithfulness")
    assert d["n"] == 6
    assert sum(d["counts"]) == 6
    assert 0.8 < d["mean"] < 0.95


def test_cost_over_time_is_cumulative(client, db):
    _seed_two_versions(client)
    series = cost_over_time(db)
    assert len(series) == 6
    assert series[-1]["cumulative_usd"] == round(
        sum(p["cost_usd"] for p in series), 6)
    assert all(series[i]["cumulative_usd"] <= series[i + 1]["cumulative_usd"]
               for i in range(len(series) - 1))


def test_summary_shape(client, db):
    _seed_two_versions(client)
    s = summary(db)
    assert s["total_traces"] == 6
    assert s["total_cost_usd"] > 0
    assert set(s["versions"]) == {"v1", "v2"}


def test_dashboard_builds_offline(tmp_path, monkeypatch):
    """build_dashboard must render dashboard.html from a seeded store."""
    import importlib.util

    db_path = tmp_path / "traces.db"
    c = OfflineLangfuse(db_path)
    _seed_two_versions(c)
    c.close()

    spec = importlib.util.spec_from_file_location(
        "build_dashboard",
        Path(__file__).resolve().parent.parent / "src" / "build_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DB", db_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    out = tmp_path / "dashboard.html"
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.main()

    page = out.read_text()
    assert "LLM Observability Dashboard" in page
    assert "v1" in page and "v2" in page
    assert "<table>" in page
