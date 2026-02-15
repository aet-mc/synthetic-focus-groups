from __future__ import annotations

import asyncio

from analysis.sentiment import SentimentAnalyzer
from discussion.models import DiscussionMessage, DiscussionPhase, MessageRole


class _BatchLLM:
    def __init__(self, payload: str):
        self.payload = payload

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> str:
        del system_prompt, user_prompt, temperature, max_tokens
        return self.payload

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> str:
        return await self.complete(system_prompt, user_prompt, temperature, max_tokens)


def _message(idx: int, text: str) -> DiscussionMessage:
    return DiscussionMessage(
        role=MessageRole.PARTICIPANT,
        speaker_id=f"p{idx}",
        speaker_name=f"Person {idx}",
        content=text,
        phase=DiscussionPhase.REACTION,
        turn_number=idx,
    )


def test_batched_scoring_uses_llm_payload_when_lengths_match() -> None:
    analyzer = SentimentAnalyzer(_BatchLLM("[0.8, -0.2, 0.1]"))
    messages = [
        _message(1, "I love this concept and would buy it."),
        _message(2, "I worry about value and would avoid it."),
        _message(3, "I need more details first."),
    ]

    scores = asyncio.run(analyzer.analyze_batch(messages))

    assert scores == [0.8, -0.2, 0.1]


def test_bug5_batched_scoring_fallback_preserves_message_count() -> None:
    # BUG 5 regression guard: malformed batch payload should not drop scores.
    analyzer = SentimentAnalyzer(_BatchLLM("[0.9]"))
    messages = [
        _message(1, "I love it and would buy it."),
        _message(2, "I worry this is overpriced and risky."),
        _message(3, "It seems okay so far."),
    ]

    scores = asyncio.run(analyzer.analyze_batch(messages))

    assert len(scores) == len(messages)
    assert scores[0] > 0
    assert scores[1] < 0
