"""Per-1k-token price table and cost estimation for traced generations.

Prices are illustrative reference values in USD; the point is attribution
plumbing, not quote accuracy. Override PRICES for your real rates.
"""

PRICES = {
    # model: (input_usd_per_1k, output_usd_per_1k)
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "claude-3-5-haiku": (0.00080, 0.00400),
    "claude-3-5-sonnet": (0.00300, 0.01500),
    "llama-3-8b": (0.00005, 0.00015),
}

DEFAULT_RATE = (0.00100, 0.00400)  # fallback for unknown models


def estimate_cost_usd(model: str, input_tokens: int,
                      output_tokens: int) -> float:
    """Estimate USD cost of one generation."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    in_rate, out_rate = PRICES.get(model, DEFAULT_RATE)
    return round(input_tokens / 1000 * in_rate
                 + output_tokens / 1000 * out_rate, 6)
