"""Shared helpers.

We deliberately keep ONE tokenizer for the whole service. Both cost tracking
(gateway half) and the hashing embedder (RAG half) need to split text into
tokens, and using the same rule everywhere keeps behaviour predictable and
testable.

NOTE: this is a *word/punctuation* tokenizer, not a real BPE tokenizer like
`tiktoken`. Real LLM providers bill on BPE tokens, so our token counts are an
approximation. That is an intentional trade-off: a genuine tokenizer would pull
in a heavy dependency and a model download, which breaks the "runs fully
offline" requirement. See README trade-offs.
"""

from __future__ import annotations

import re

# One token = a run of word characters OR a single punctuation char.
# Example: "hello, world!" -> ["hello", ",", "world", "!"]
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Split text into approximate tokens (deterministic, dependency-free)."""
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    return len(tokenize(text))
