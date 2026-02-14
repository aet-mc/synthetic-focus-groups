from __future__ import annotations

import json

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionTranscript, MessageRole

from .models import ConceptScores
from .prompts import CONCEPT_SCORE_PROMPT


class ConceptScorer:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def score_concept(self, transcript: DiscussionTranscript, personas: list) -> ConceptScores:
        participant_messages: dict[str, list[str]] = {}
        for message in transcript.messages:
            if message.role != MessageRole.PARTICIPANT:
                continue
            participant_messages.setdefault(message.speaker_id, []).append(message.content)

        per_participant: dict[str, dict[str, float]] = {}
        for persona in personas:
            statements = participant_messages.get(persona.id, [])
            if isinstance(self.llm, MockLLMClient):
                scores = self._mock_scores_for_persona(persona)
            else:
                scores = await self._score_with_llm(persona, statements)
                if not scores:
                    scores = self._mock_scores_for_persona(persona)
            per_participant[persona.id] = scores

        aggregate = {
            metric: self._top2box(per_participant, metric)
            for metric in [
                "purchase_intent",
                "overall_appeal",
                "uniqueness",
                "relevance",
                "believability",
                "value_perception",
            ]
        }

        excitement = (
            aggregate["overall_appeal"] * 0.3
            + aggregate["uniqueness"] * 0.25
            + aggregate["purchase_intent"] * 0.25
            + aggregate["relevance"] * 0.2
        )

        return ConceptScores(
            purchase_intent=aggregate["purchase_intent"],
            overall_appeal=aggregate["overall_appeal"],
            uniqueness=aggregate["uniqueness"],
            relevance=aggregate["relevance"],
            believability=aggregate["believability"],
            value_perception=aggregate["value_perception"],
            excitement_score=round(excitement, 4),
            participant_scores=per_participant,
        )

    async def _score_with_llm(self, persona, statements: list[str]) -> dict[str, float]:
        prompt = CONCEPT_SCORE_PROMPT.format(
            persona=persona.model_dump_json(indent=2),
            statements="\n".join(f"- {line}" for line in statements) if statements else "- No statements",
        )
        raw = await self.llm.complete(
            system_prompt="You are a concept testing analyst. Return JSON only.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=250,
        )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        metrics = [
            "purchase_intent",
            "overall_appeal",
            "uniqueness",
            "relevance",
            "believability",
            "value_perception",
        ]
        normalized: dict[str, float] = {}
        for metric in metrics:
            if metric not in parsed:
                return {}
            normalized[metric] = round(self._clamp(float(parsed[metric]), 1.0, 5.0), 2)
        return normalized

    def _mock_scores_for_persona(self, persona) -> dict[str, float]:
        valence = 0.0 if persona.opinion_valence is None else float(persona.opinion_valence)
        base = 1.0 + ((valence + 1.0) / 2.0) * 4.0

        ocean = persona.psychographics.ocean
        openness = (ocean.openness - 50.0) / 50.0
        agreeableness = (ocean.agreeableness - 50.0) / 50.0
        conscientiousness = (ocean.conscientiousness - 50.0) / 50.0
        neuroticism = (ocean.neuroticism - 50.0) / 50.0
        risk_tolerance = (persona.consumer.risk_tolerance - 0.5) * 2.0
        price_sensitivity = (persona.consumer.price_sensitivity - 0.5) * 2.0

        scores = {
            "purchase_intent": base + (risk_tolerance * 0.4) - (price_sensitivity * 0.35),
            "overall_appeal": base + (openness * 0.3),
            "uniqueness": base + (openness * 0.6),
            "relevance": base + (conscientiousness * 0.25),
            "believability": base + (conscientiousness * 0.25) - (neuroticism * 0.3),
            "value_perception": base - (price_sensitivity * 0.5) + (agreeableness * 0.15),
        }

        return {
            metric: round(self._clamp(value, 1.0, 5.0), 2)
            for metric, value in scores.items()
        }

    @staticmethod
    def _top2box(participant_scores: dict[str, dict[str, float]], metric: str) -> float:
        if not participant_scores:
            return 0.0
        values = [scores[metric] for scores in participant_scores.values() if metric in scores]
        if not values:
            return 0.0
        top_two = sum(1 for score in values if score >= 4.0)
        return round(top_two / len(values), 4)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
