"""Seed the observability store with ~30 simulated traces.

Simulates a RAG customer-support assistant running behind two prompt
versions:

* v1 — GPT-4o, thorough prompts: higher faithfulness, higher cost.
* v2 — GPT-4o-mini, tighter prompts: ~85% cheaper, but faithfulness dips
  on long-context questions (a realistic cost/quality tradeoff story).

No network, no keys — random but seeded for reproducibility.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shim import OfflineLangfuse  # noqa: E402

random.seed(42)

QUESTIONS = [
    ("How do I reset my password?", "short"),
    ("What is your refund policy for annual plans?", "short"),
    ("Explain how the SSO integration works with Okta", "long"),
    ("Summarize ticket #8842 and suggest next steps", "long"),
    ("Where can I download my invoices?", "short"),
    ("Compare the Pro and Enterprise tiers", "long"),
    ("Why did my API key stop working?", "short"),
    ("Draft a status update for the Q3 migration", "long"),
    ("How do I invite teammates to my workspace?", "short"),
    ("What data do you retain after account deletion?", "long"),
]

VERSIONS = {
    "v1": {"model": "gpt-4o", "input_mult": 3.0, "out_mult": 2.5,
           "faith_base": 0.90, "faith_drop_long": 0.02},
    "v2": {"model": "gpt-4o-mini", "input_mult": 1.0, "out_mult": 1.0,
           "faith_base": 0.86, "faith_drop_long": 0.09},
}


def fake_completion(question: str, version: str) -> str:
    tail = (" (detailed runbook with 5 steps)" if version == "v1"
            else " (concise answer)")
    return f"Answer to: {question}{tail}"


def main() -> None:
    client = OfflineLangfuse()
    now = time.time()

    n = 0
    for i in range(30):
        version = "v1" if i < 15 else "v2"   # 15 traces per version
        cfg = VERSIONS[version]
        question, qlen = QUESTIONS[i % len(QUESTIONS)]

        with client.trace("support-qa", prompt_version=version) as tr:
            # retrieval step
            with tr.span("retriever", {"top_k": 5, "index": "docs-v3"}):
                pass
            # model generation
            in_tok = int(random.randint(180, 260) * cfg["input_mult"])
            out_tok = int(random.randint(120, 200) * cfg["out_mult"])
            tr.generation(
                name="llm-call",
                model=cfg["model"],
                input=question,
                completion=fake_completion(question, version),
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=random.uniform(400, 1400)
                * (0.55 if version == "v2" else 1.0),
            )
            faith = min(0.99, max(
                0.5, random.gauss(cfg["faith_base"] -
                                  (cfg["faith_drop_long"] if qlen == "long"
                                   else 0.0), 0.05)))
            tox = max(0.0, min(0.2, random.gauss(0.04, 0.03)))
            tr.score("faithfulness", round(faith, 3),
                     "LLM-judge faithfulness vs retrieved docs")
            tr.score("toxicity", round(tox, 3), "toxicity classifier")
            tr.end()

        # spread created_at over the last 6 hours for a nicer chart
        ts = now - (29 - i) * 720
        client.conn.execute("UPDATE traces SET created_at=? WHERE id=?",
                            (ts, tr.id))
        client.conn.commit()
        n += 1

    client.flush()
    client.close()
    print(f"Seeded {n} traces into {client.db_path}")


if __name__ == "__main__":
    main()
