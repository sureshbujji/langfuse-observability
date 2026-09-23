# langfuse-observability

LLM observability for QA teams — trace every prompt, score every answer, attribute every cent. **Runs 100% offline**: no API keys, no network, no cloud account needed.

## What I built

I built a Langfuse-compatible observability layer for LLM apps so a QA team can treat prompts like code under test: every generation gets a **trace** (prompt, completion, model, tokens, latency, cost) plus **scores** (faithfulness, toxicity from an LLM-judge/classifier pipeline). An offline SQLite shim implements the Langfuse SDK interface (`trace`, `span`, `generation`, `score`), so demos, tests, and CI run with zero keys — and the exact same call sites talk to real Langfuse Cloud once you drop in `LANGFUSE_HOST` and keys.

To make the story concrete I seeded a realistic A/B: prompt **v1** on GPT-4o (thorough, expensive) vs **v2** on GPT-4o-mini (terse, ~97% cheaper) — and showed what the tradeoff looks like in the numbers.

## Architecture

```
                  +-----------------+
                  |  Your LLM app   |
                  +--------+--------+
                           | trace(name, prompt_version)
                           v
  +------------------------+--------------------------+
  |              get_client()  (src/langfuse_client.py) |
  |   keys present?  --yes-->  Langfuse SDK -> Cloud   |
  |        \                                             |
  |         no                                           |
  |          v                                           |
  |   OfflineLangfuse (src/shim.py)                      |
  |   trace / span / generation / score                  |
  +------------+--------------+--------------------------+
               |              |
               v              v
   src/pricing.py      SQLite (reports/traces.db)
   $/1k tokens         traces | spans | scores
               |              |
               +------+-------+
                      v
            src/analysis.py  (distributions, version compare, cost series)
                      v
            src/build_dashboard.py  -->  reports/dashboard.html (static)
```

## Quickstart

```bash
pip install -r requirements.txt      # pytest only; stdlib does the rest
python src/seed_traces.py            # simulate 30 traces across v1/v2
python src/build_dashboard.py        # render the static dashboard
open reports/dashboard.html          # no server needed
python -m pytest -q                  # 7 tests, all offline
```

## Sample output

```
$ python src/seed_traces.py
Seeded 30 traces into reports/traces.db

$ python -m pytest -q
7 passed in 0.10s

$ python src/build_dashboard.py
Wrote reports/dashboard.html (13377 bytes)
```

Version comparison from a real seeded run (`reports/summary.json`):

| version | traces | avg cost/trace | total cost | avg latency | avg faithfulness |
|---------|--------|----------------|------------|-------------|------------------|
| v1 (gpt-4o) | 15 | $0.005555 | $0.083325 | 870.9 ms | 0.9051 |
| v2 (gpt-4o-mini) | 15 | $0.000132 | $0.001973 | 474.8 ms | 0.8165 |

v2 costs **~97% less** per trace and answers **~2x faster** — but faithfulness drops by **0.089**, concentrated on long-context questions. That is exactly the decision a QA sign-off should gate: cheaper is not automatically shippable.

The dashboard (`reports/dashboard.html`) renders all of this statically: summary cards, trace table, faithfulness/toxicity histograms, cost-over-time curve, and the v1-vs-v2 panel — open it straight from the file system.

## Swap to real Langfuse

One line — no call-site changes, because the shim mirrors the SDK's method names:

```python
from langfuse_client import get_client
client = get_client()   # real Langfuse when keys exist, SQLite shim otherwise
```

```bash
cp .env.example .env     # then set:
# LANGFUSE_HOST=https://cloud.langfuse.com
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
```

(`langfuse` is an optional dependency in `requirements.txt` — uncomment and `pip install langfuse` to enable the real path.)

## What observability buys a QA team

- **Regression detection.** Every release gets scored traces. When mean faithfulness slips from 0.90 to 0.82 on the same question set, you know the prompt (or the model) regressed before users do — and the trace table shows you exactly which questions.
- **Cost attribution.** Per-trace cost from a per-1k-token price table (`src/pricing.py`) rolls up per version, per model, per day. "Prompt v2 will save us $X/month at current volume" stops being a guess.
- **Prompt-version comparison.** Versions are first-class (`prompt_version` on every trace). Ship v2 only when the dashboard shows quality parity — the v1/v2 panel exists to make that sign-off a data call, not a vibe call.
- **Reproducible evidence.** SQLite + static HTML means any engineer, auditor, or hiring manager can rerun `seed → build → open` and see the same numbers. No dashboards that only exist on someone's cloud account.

## Roadmap

- [ ] Live LLM-judge scoring hook (faithfulness/toxicity scored by a model call, stored as scores)
- [ ] Alerting thresholds: CI fails when version-over-version faithfulness drops > 0.03
- [ ] Token-level cost breakdown per span, not just per generation
- [ ] Prompt template registry: versioned prompts stored alongside traces
- [ ] Export to real Langfuse Cloud for long-term retention (the swap point already exists)

## Project layout

```
src/
  shim.py            Offline Langfuse-compatible client (SQLite)
  pricing.py         Per-1k-token price table + cost estimation
  langfuse_client.py get_client(): real SDK if keys exist, shim otherwise
  seed_traces.py     Simulate ~30 traces across prompt v1/v2
  build_dashboard.py Static HTML dashboard generator
  analysis.py        Distributions, version comparison, cost series
tests/
  test_observability.py   7 tests, stdlib + pytest only
reports/
  dashboard.html     Generated dashboard (open in browser)
  summary.json       Machine-readable aggregates
.github/workflows/ci.yml   pytest + offline smoke test on 3.10/3.11/3.12
```

MIT License — Suresh Itha.
