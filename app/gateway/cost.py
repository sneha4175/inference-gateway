"""Token counting and cost estimation.

Why track cost in a gateway at all? LLM calls are billed per *token*, not per
request, and cost is the number engineers/finance actually care about. A single
runaway RAG prompt can cost 100x a normal chat call. Centralising the maths here
means every route reports cost the same way.
"""

from __future__ import annotations

from app.schemas import Message, Usage
from app.util import count_tokens

# USD per 1,000 tokens as (prompt_price, completion_price).
# The mock model is free — that is what lets the whole service run at $0 offline.
# The real prices are illustrative published rates; keep them in one place so
# updating a price never means touching call sites.
PRICING: dict[str, tuple[float, float]] = {
    "mock-1": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


def prompt_tokens(messages: list[Message]) -> int:
    """Total tokens across all messages in a prompt."""
    return sum(count_tokens(m.content) for m in messages)


def estimate_cost(model: str, prompt_toks: int, completion_toks: int) -> float:
    """Cost in USD for a call. Unknown models are treated as free (0.0)."""
    p_price, c_price = PRICING.get(model, (0.0, 0.0))
    cost = (prompt_toks / 1000) * p_price + (completion_toks / 1000) * c_price
    return round(cost, 6)


def build_usage(model: str, prompt_toks: int, completion_text: str) -> Usage:
    """Assemble a Usage record from a prompt token count + the raw completion."""
    completion_toks = count_tokens(completion_text)
    return Usage(
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
        total_tokens=prompt_toks + completion_toks,
        cost_usd=estimate_cost(model, prompt_toks, completion_toks),
    )
