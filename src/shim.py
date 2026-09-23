"""Offline Langfuse-style observability shim.

Implements the Langfuse SDK interface (trace, span, generation, score) backed
by stdlib SQLite — no API keys, no network. All demos and tests run against
this so CI stays green offline.

To swap in the real client, wrap with src/langfuse_client.py::

    from langfuse import Langfuse
    client = Langfuse(host=os.environ["LANGFUSE_HOST"],
                      public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                      secret_key=os.environ["LANGFUSE_SECRET_KEY"])

The shim mirrors the method names of the real SDK so call-site code does not
need to change.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "reports" / "traces.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt_version TEXT,
    model TEXT,
    input TEXT,
    output TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(id),
    name TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS scores (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    comment TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_scores_trace ON scores(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_version ON traces(prompt_version);
"""


class _Timer:
    def __init__(self):
        self.started = time.perf_counter()


class Span:
    """A timed operation inside a trace (mirrors langfuse span objects)."""

    def __init__(self, client: "OfflineLangfuse", trace_id: str, name: str,
                 metadata: dict | None = None):
        self._client = client
        self.id = uuid.uuid4().hex
        self.trace_id = trace_id
        self.name = name
        self.metadata = metadata or {}
        self._t = _Timer()
        self._ended = False
        client._insert_span(self, ended_at=None)

    def end(self, metadata: dict | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if metadata:
            self.metadata.update(metadata)
        self._client._update_span(self)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, *exc) -> None:
        self.end()


class Trace:
    """One observed LLM call (mirrors the langfuse trace object)."""

    def __init__(self, client: "OfflineLangfuse", name: str,
                 prompt_version: str | None = None,
                 metadata: dict | None = None):
        self._client = client
        self.id = uuid.uuid4().hex
        self.name = name
        self.prompt_version = prompt_version
        self.metadata = metadata or {}
        self.model: str | None = None
        self.input: str | None = None
        self.output: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self._latency_explicit = False  # True when generation() pinned it
        self.created_at = time.time()
        self._t = _Timer()
        client._insert_trace(self)

    def span(self, name: str, metadata: dict | None = None) -> Span:
        return Span(self._client, self.id, name, metadata)

    def generation(self, name: str, model: str, input: str,
                   completion: str | None = None,
                   input_tokens: int = 0, output_tokens: int = 0,
                   latency_ms: float | None = None,
                   metadata: dict | None = None) -> Span:
        """Log a model generation: a span plus token/cost bookkeeping.

        Mirrors ``trace.generation()`` in the Langfuse SDK. If latency is not
        supplied it is measured from trace creation.
        """
        try:
            from .pricing import estimate_cost_usd  # local import avoids cycles
        except ImportError:  # src/ used as a top-level script path
            from pricing import estimate_cost_usd  # type: ignore[no-redef]

        self.model = model
        self.input = input
        self.output = completion
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = (latency_ms if latency_ms is not None
                           else (time.perf_counter() - self._t.started) * 1000)
        self._latency_explicit = latency_ms is not None
        self._client._estimate = estimate_cost_usd
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        self._client._update_trace(self, cost_usd=cost)
        span = Span(self._client, self.id, name,
                    {"model": model, "kind": "generation", **(metadata or {})})
        span.end()
        return span

    def score(self, name: str, value: float, comment: str = "") -> None:
        self._client._insert_score(self.id, name, value, comment)

    def end(self) -> None:
        self._client._update_trace(self)

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc) -> None:
        self.end()


class OfflineLangfuse:
    """Drop-in Langfuse replacement storing everything in SQLite."""

    def __init__(self, db_path: str | os.PathLike | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- SDK-style public API ------------------------------------------------
    def trace(self, name: str, prompt_version: str | None = None,
              metadata: dict | None = None) -> Trace:
        return Trace(self, name, prompt_version, metadata)

    def flush(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- internal persistence ------------------------------------------------
    def _insert_trace(self, t: Trace) -> None:
        self.conn.execute(
            """INSERT INTO traces (id, name, prompt_version, model, input, output,
                                   input_tokens, output_tokens, total_tokens,
                                   latency_ms, cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (t.id, t.name, t.prompt_version, t.model, t.input, t.output,
             t.input_tokens, t.output_tokens,
             t.input_tokens + t.output_tokens, t.latency_ms, 0.0, t.created_at))
        self.conn.commit()

    def _update_trace(self, t: Trace, cost_usd: float | None = None) -> None:
        if cost_usd is not None:
            self.conn.execute(
                """UPDATE traces SET model=?, input=?, output=?,
                   input_tokens=?, output_tokens=?, total_tokens=?,
                   latency_ms=?, cost_usd=? WHERE id=?""",
                (t.model, t.input, t.output, t.input_tokens, t.output_tokens,
                 t.input_tokens + t.output_tokens, t.latency_ms, cost_usd,
                 t.id))
        else:
            final_latency = (t.latency_ms if t._latency_explicit
                             else (time.perf_counter() - t._t.started) * 1000)
            self.conn.execute("UPDATE traces SET latency_ms=? WHERE id=?",
                              (final_latency, t.id))
        self.conn.commit()

    def _insert_span(self, s: Span, ended_at: float | None) -> None:
        self.conn.execute(
            "INSERT INTO spans (id, trace_id, name, started_at, ended_at, metadata)"
            " VALUES (?,?,?,?,?,?)",
            (s.id, s.trace_id, s.name, s._t.started, ended_at,
             json.dumps(s.metadata)))
        self.conn.commit()

    def _update_span(self, s: Span) -> None:
        elapsed = (time.perf_counter() - s._t.started) * 1000
        self.conn.execute(
            "UPDATE spans SET ended_at=?, metadata=? WHERE id=?",
            (s._t.started + elapsed / 1000, json.dumps(s.metadata), s.id))
        self.conn.commit()

    def _insert_score(self, trace_id: str, name: str, value: float,
                      comment: str) -> None:
        self.conn.execute(
            "INSERT INTO scores (id, trace_id, name, value, comment)"
            " VALUES (?,?,?,?,?)",
            (uuid.uuid4().hex, trace_id, name, value, comment))
        self.conn.commit()
