"""Analytics over the SQLite observability store.

Used by build_dashboard.py and by the test suite. All functions take a path
to the traces.db file so tests can point at throwaway copies.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path


def _conn(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def traces(db_path: str | Path) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM traces ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def scores_by_trace(db_path: str | Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with _conn(db_path) as c:
        for r in c.execute("SELECT trace_id, name, value FROM scores"):
            out[r["trace_id"]][r["name"]] = r["value"]
    return dict(out)


def score_distribution(db_path: str | Path, score_name: str,
                       buckets: int = 10) -> dict:
    """Histogram of a score's values in [0,1] across all traces."""
    values: list[float] = []
    with _conn(db_path) as c:
        for r in c.execute("SELECT value FROM scores WHERE name=?",
                           (score_name,)):
            values.append(float(r["value"]))
    hist = [0] * buckets
    for v in values:
        i = min(int(v * buckets), buckets - 1)
        hist[i] += 1
    return {"score": score_name, "buckets": buckets,
            "counts": hist, "n": len(values),
            "mean": round(sum(values) / len(values), 4) if values else 0.0}


def cost_over_time(db_path: str | Path) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT created_at, cost_usd FROM traces ORDER BY created_at"
        ).fetchall()
    cum = 0.0
    series = []
    for r in rows:
        cum += float(r["cost_usd"])
        series.append({"t": r["created_at"],
                       "cost_usd": round(float(r["cost_usd"]), 6),
                       "cumulative_usd": round(cum, 6)})
    return series


def version_comparison(db_path: str | Path) -> dict:
    """Aggregate quality + cost metrics per prompt version (v1 vs v2)."""
    tlist = traces(db_path)
    sc = scores_by_trace(db_path)
    agg: dict[str, dict] = {}
    for t in tlist:
        v = t["prompt_version"] or "unversioned"
        g = agg.setdefault(v, {"n": 0, "cost": 0.0, "latency": 0.0,
                               "faithfulness": [], "toxicity": []})
        g["n"] += 1
        g["cost"] += t["cost_usd"] or 0.0
        g["latency"] += t["latency_ms"] or 0.0
        ts = sc.get(t["id"], {})
        if "faithfulness" in ts:
            g["faithfulness"].append(ts["faithfulness"])
        if "toxicity" in ts:
            g["toxicity"].append(ts["toxicity"])
    for v, g in agg.items():
        n = g["n"]
        g["avg_cost_usd"] = round(g["cost"] / n, 6)
        g["total_cost_usd"] = round(g["cost"], 6)
        g["avg_latency_ms"] = round(g["latency"] / n, 1)
        g["avg_faithfulness"] = (round(sum(g["faithfulness"]) /
                                      len(g["faithfulness"]), 4)
                                 if g["faithfulness"] else None)
        g["avg_toxicity"] = (round(sum(g["toxicity"]) /
                                   len(g["toxicity"]), 4)
                             if g["toxicity"] else None)
        del g["cost"], g["latency"]
    return agg


def summary(db_path: str | Path) -> dict:
    tlist = traces(db_path)
    total_cost = round(sum(t["cost_usd"] or 0.0 for t in tlist), 6)
    avg_lat = (round(sum(t["latency_ms"] or 0.0 for t in tlist) / len(tlist), 1)
               if tlist else 0.0)
    return {
        "total_traces": len(tlist),
        "total_cost_usd": total_cost,
        "avg_latency_ms": avg_lat,
        "versions": version_comparison(db_path),
        "faithfulness": score_distribution(db_path, "faithfulness"),
        "toxicity": score_distribution(db_path, "toxicity"),
    }
