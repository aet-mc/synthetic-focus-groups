from __future__ import annotations

import hashlib
import json
import statistics

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionMessage, MessageRole
from persona_engine.models import Persona

from .models import SentimentTimeline
from .prompts import SENTIMENT_BATCH_PROMPT


class SentimentAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
        self._persona_map: dict[str, Persona] = {}

    def set_personas(self, personas: list[Persona]) -> None:
        self._persona_map = {persona.id: persona for persona in personas}

    async def analyze_message(self, message: DiscussionMessage) -> float:
        scores = await self.analyze_batch([message])
        return scores[0]

    async def analyze_batch(self, messages: list[DiscussionMessage]) -> list[float]:
        if not messages:
            return []

        if isinstance(self.llm, MockLLMClient):
            return [self._mock_score(message) for message in messages]

        responses = [f"{idx}. {message.content}" for idx, message in enumerate(messages)]
        prompt = SENTIMENT_BATCH_PROMPT.format(responses="\n".join(responses))
        raw = await self.llm.complete(
            system_prompt="You are a sentiment analyst. Return JSON only.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=700,
        )

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or len(parsed) != len(messages):
                raise ValueError("Invalid sentiment payload")
            return [self._clamp(float(value), -1.0, 1.0) for value in parsed]
        except (ValueError, TypeError, json.JSONDecodeError):
            return [self._heuristic_score(message.content) for message in messages]

    def compute_timeline(self, messages: list[DiscussionMessage], scores: list[float]) -> SentimentTimeline:
        if len(messages) != len(scores):
            raise ValueError("messages and scores length mismatch")

        by_phase_raw: dict[str, list[float]] = {}
        by_turn: list[float] = []
        for message, score in zip(messages, scores):
            phase_key = message.phase.value
            by_phase_raw.setdefault(phase_key, []).append(score)
            by_turn.append(score)

        by_phase = {
            phase: (sum(phase_scores) / len(phase_scores) if phase_scores else 0.0)
            for phase, phase_scores in by_phase_raw.items()
        }

        overall = sum(by_turn) / len(by_turn) if by_turn else 0.0
        trend = self._detect_trend(by_turn)

        return SentimentTimeline(by_phase=by_phase, by_turn=by_turn, overall=overall, trend=trend)

    def _mock_score(self, message: DiscussionMessage) -> float:
        if message.role != MessageRole.PARTICIPANT:
            return 0.0

        persona = self._persona_map.get(message.speaker_id)
        base = persona.opinion_valence if persona and persona.opinion_valence is not None else None
        if base is None and message.sentiment is not None:
            base = message.sentiment
        if base is None:
            base = self._heuristic_score(message.content)

        digest = hashlib.sha256(f"{message.speaker_id}:{message.turn_number}".encode("utf-8")).hexdigest()
        noise_bucket = int(digest[:2], 16) / 255.0
        noise = (noise_bucket - 0.5) * 0.16

        return self._clamp(float(base) + noise, -1.0, 1.0)

    @staticmethod
    def _heuristic_score(text: str) -> float:
        lowered = text.lower()
        pos_terms = ["like", "love", "good", "great", "helpful", "buy", "useful"]
        neg_terms = ["bad", "worry", "concern", "skeptical", "avoid", "overpriced", "risk"]
        pos = sum(1 for token in pos_terms if token in lowered)
        neg = sum(1 for token in neg_terms if token in lowered)
        total = pos + neg
        if total == 0:
            return 0.0
        return SentimentAnalyzer._clamp((pos - neg) / total, -1.0, 1.0)

    @staticmethod
    def _detect_trend(scores: list[float]) -> str:
        if len(scores) < 4:
            return "stable"

        window = max(1, len(scores) // 3)
        start = sum(scores[:window]) / window
        end = sum(scores[-window:]) / window
        delta = end - start
        volatility = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        if abs(delta) >= 0.15:
            return "improving" if delta > 0 else "declining"
        if volatility >= 0.35:
            return "volatile"
        return "stable"

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
