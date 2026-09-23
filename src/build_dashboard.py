"""Build a static HTML dashboard from the SQLite observability store.

Renders: summary cards, a trace table, score distributions, cost over time,
and a v1-vs-v2 comparison panel. Pure static HTML + inline SVG/CSS — no
server, no JS framework, opens straight from the file system.

Usage:  python src/build_dashboard.py
Output: reports/dashboard.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import (  # noqa: E402
    cost_over_time, score_distribution, summary, traces,
    version_comparison,
)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "reports" / "traces.db"
OUT = ROOT / "reports" / "dashboard.html"


def _esc(v) -> str:
    return html.escape(str(v))


def _histogram_svg(dist: dict, color: str) -> str:
    buckets, counts = dist["buckets"], dist["counts"]
    w, h, pad = 420, 160, 30
    bw = (w - 2 * pad) / buckets
    mx = max(counts) or 1
    bars = []
    for i, c in enumerate(counts):
        bh = (h - 2 * pad) * c / mx
        x = pad + i * bw
        y = h - pad - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" '
            f'height="{bh:.1f}" fill="{color}" rx="2">'
            f'<title>{i / buckets:.1f}-{(i + 1) / buckets:.1f}: {c}</title>'
            '</rect>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart">'
            f'<text x="{pad}" y="16" class="cap">n={dist["n"]} '
            f'mean={dist["mean"]}</text>'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" '
            'stroke="#555"/>' + "".join(bars) + "</svg>")


def _line_svg(series: list[dict]) -> str:
    w, h, pad = 620, 200, 40
    if not series:
        return '<svg viewBox="0 0 620 200" class="chart"></svg>'
    mx = max(p["cumulative_usd"] for p in series) or 1
    pts = []
    for i, p in enumerate(series):
        x = pad + i * (w - 2 * pad) / max(len(series) - 1, 1)
        y = h - pad - (h - 2 * pad) * p["cumulative_usd"] / mx
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart">'
        f'<text x="{pad}" y="16" class="cap">cumulative USD (max {mx:.4f})</text>'
        f'<polyline points="{poly}" fill="none" stroke="#4cc38a" '
        'stroke-width="2"/>' + "".join(
            f'<circle cx="{x}" cy="{y}" r="2.5" fill="#4cc38a"/>'
            for x, y in (p.split(",") for p in pts)) + "</svg>")


def _trace_table(tlist: list[dict]) -> str:
    rows = []
    for t in tlist[-50:]:  # most recent 50
        rows.append(
            "<tr>"
            f"<td class='mono'>{_esc(t['id'][:8])}</td>"
            f"<td>{_esc(t['prompt_version'] or '')}</td>"
            f"<td>{_esc(t['model'] or '')}</td>"
            f"<td class='num'>{t['input_tokens'] + t['output_tokens']}</td>"
            f"<td class='num'>{t['latency_ms']:.0f}</td>"
            f"<td class='num'>${t['cost_usd']:.6f}</td>"
            f"<td>{_esc((t['input'] or '')[:70])}</td>"
            "</tr>")
    return (
        "<table><thead><tr><th>trace</th><th>version</th><th>model</th>"
        "<th>tokens</th><th>latency ms</th><th>cost</th><th>prompt</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _comparison_panel(versions: dict) -> str:
    cards = []
    for v in sorted(versions):
        g = versions[v]
        cards.append(
            f"<div class='vcard'><h3>{_esc(v)}</h3>"
            f"<div>n = {g['n']}</div>"
            f"<div>avg cost <b>${g['avg_cost_usd']:.6f}</b> "
            f"(total ${g['total_cost_usd']:.6f})</div>"
            f"<div>avg latency <b>{g['avg_latency_ms']} ms</b></div>"
            f"<div>avg faithfulness <b>{g['avg_faithfulness']}</b></div>"
            f"<div>avg toxicity <b>{g['avg_toxicity']}</b></div></div>")
    note = ""
    if "v1" in versions and "v2" in versions:
        v1, v2 = versions["v1"], versions["v2"]
        cheaper = (v1["avg_cost_usd"] - v2["avg_cost_usd"])
        pct = (100 * cheaper / v1["avg_cost_usd"]) if v1["avg_cost_usd"] else 0
        df = (v1["avg_faithfulness"] or 0) - (v2["avg_faithfulness"] or 0)
        note = (f"<p class='takeaway'>v2 costs <b>{pct:.0f}% less</b> per "
                f"trace (${v1['avg_cost_usd']:.6f} → ${v2['avg_cost_usd']:.6f}) "
                f"but faithfulness drops by <b>{df:.3f}</b>. "
                "That is the tradeoff to A/B-gate before rollout.</p>")
    return "<div class='vrow'>" + "".join(cards) + "</div>" + note


CSS = """
body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;
color:#e6e9ef;margin:0;padding:24px}
h1{margin:0 0 4px} .sub{color:#8b93a1;margin-bottom:20px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}
.card{background:#161b22;border:1px solid #2a3140;border-radius:8px;
padding:14px 18px;min-width:150px}
.card .big{font-size:24px;font-weight:700;color:#58a6ff}
.card .lbl{font-size:12px;color:#8b93a1;text-transform:uppercase}
.panel{background:#161b22;border:1px solid #2a3140;border-radius:8px;
padding:18px;margin-bottom:24px}
.chart{width:100%;max-width:640px}.cap{fill:#8b93a1;font-size:11px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px;border-bottom:1px solid #2a3140}
th{color:#8b93a1;font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,monospace}.vrow{display:flex;gap:12px;flex-wrap:wrap}
.vcard{background:#0e1116;border:1px solid #2a3140;border-radius:8px;padding:14px;min-width:220px}
.takeaway{background:#0e2b1c;border:1px solid #2ea043;border-radius:8px;padding:12px;margin-top:12px}
"""

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Observability Dashboard</title><style>{css}</style></head>
<body>
<h1>LLM Observability Dashboard</h1>
<div class="sub">Generated from local SQLite traces (offline Langfuse shim) &mdash;
{total} traces &bull; prompt versions: {versions}</div>

<div class="cards">
<div class="card"><div class="lbl">traces</div><div class="big">{total}</div></div>
<div class="card"><div class="lbl">total cost</div><div class="big">${cost}</div></div>
<div class="card"><div class="lbl">avg latency</div><div class="big">{lat} ms</div></div>
</div>

<div class="panel"><h2>Prompt version comparison</h2>{compare}</div>

<div class="panel"><h2>Score distributions</h2>
<h3>faithfulness</h3>{fh}<h3>toxicity</h3>{th}</div>

<div class="panel"><h2>Cost over time</h2>{costsvg}</div>

<div class="panel"><h2>Recent traces</h2>{table}</div>

<footer class="sub">Built offline with stdlib SQLite. Swap in the real Langfuse
SDK via LANGFUSE_HOST + keys &mdash; same call sites.</footer>
</body></html>"""


def main() -> None:
    if not DB.exists():
        print(f"{DB} not found. Run: python src/seed_traces.py")
        raise SystemExit(1)
    s = summary(DB)
    tlist = traces(DB)
    page = HTML.format(
        css=CSS,
        total=s["total_traces"],
        cost=f"{s['total_cost_usd']:.6f}",
        lat=s["avg_latency_ms"],
        versions=", ".join(sorted(s["versions"])),
        compare=_comparison_panel(version_comparison(DB)),
        fh=_histogram_svg(s["faithfulness"], "#58a6ff"),
        th=_histogram_svg(s["toxicity"], "#f0883e"),
        costsvg=_line_svg(cost_over_time(DB)),
        table=_trace_table(tlist),
    )
    OUT.write_text(page)
    # also drop a machine-readable summary for tests/CI
    (ROOT / "reports" / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"Wrote {OUT} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
