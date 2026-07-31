"""Token counting and cost estimation are deterministic."""

from app.gateway.cost import estimate_cost
from app.schemas import Message
from app.gateway.cost import prompt_tokens
from app.util import count_tokens


def test_count_tokens_words_and_punctuation():
    # "hello, world!" -> hello , world !  == 4 tokens.
    assert count_tokens("hello, world!") == 4
    assert count_tokens("") == 0


def test_prompt_tokens_sums_messages():
    msgs = [Message(role="system", content="be nice"),      # 2
            Message(role="user", content="hi there")]         # 2
    assert prompt_tokens(msgs) == 4


def test_cost_is_zero_for_mock_and_unknown_models():
    assert estimate_cost("mock-1", 1000, 1000) == 0.0
    assert estimate_cost("some-unlisted-model", 1000, 1000) == 0.0


def test_cost_scales_with_tokens_for_priced_model():
    # gpt-4o-mini: 0.00015 prompt + 0.0006 completion per 1K tokens.
    assert estimate_cost("gpt-4o-mini", 1000, 0) == 0.00015
    assert estimate_cost("gpt-4o-mini", 0, 1000) == 0.0006
