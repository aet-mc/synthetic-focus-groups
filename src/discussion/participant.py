from __future__ import annotations

import json
import random
import re
from typing import Iterable

from persona_engine.models import Persona

from .llm_client import LLMClient
from .models import DiscussionMessage, DiscussionPhase, MessageRole
from .prompts import (
    OPINION_SHIFT_DETECTION_PROMPT,
    PARTICIPANT_RESPONSE_PROMPT,
    PERSONA_SYSTEM_PROMPT,
)


class Participant:
    def __init__(self, persona: Persona, llm_client: LLMClient):
        self.persona = persona
        self.llm_client = llm_client
        self.times_spoken = 0

    def build_system_prompt(self) -> str:
        demo = self.persona.demographics
        ocean = self.persona.psychographics.ocean

        personality_bits = [
            self._describe_openness(ocean.openness),
            self._describe_conscientiousness(ocean.conscientiousness),
            self._describe_extraversion(ocean.extraversion),
            self._describe_agreeableness(ocean.agreeableness),
            self._describe_neuroticism(ocean.neuroticism),
        ]

        communication_style = (
            f"You usually communicate in a {self.persona.voice.communication_style} way with "
            f"{self.persona.voice.vocabulary_level} vocabulary. You're typically {self.persona.voice.verbosity} "
            f"in how much you say."
        )

        consumer_behavior = (
            f"You're generally {self._level_text(self.persona.consumer.price_sensitivity)} price-sensitive, "
            f"{self._level_text(self.persona.consumer.brand_loyalty)} brand-loyal, and "
            f"{self._level_text(self.persona.consumer.research_tendency)} likely to research before buying. "
            f"Your decision style is {self.persona.consumer.decision_style}."
        )

        location = demo.location.state
        if demo.location.metro_area:
            location = f"{demo.location.metro_area}, {demo.location.state}"

        return PERSONA_SYSTEM_PROMPT.format(
            name=self.persona.name,
            age=demo.age,
            occupation=demo.occupation,
            location=location,
            personality_description="\n".join(f"- {line}" for line in personality_bits),
            communication_style=communication_style,
            consumer_behavior=consumer_behavior,
            category_engagement=self.persona.consumer.category_engagement,
            initial_opinion=self.persona.initial_opinion or "No strong initial opinion.",
        )

    async def respond(
        self,
        moderator_question: str,
        discussion_context: list[DiscussionMessage],
        phase: DiscussionPhase,
    ) -> DiscussionMessage:
        system_prompt = self.build_system_prompt()
        context_text = self._format_context(discussion_context[-16:])

        user_prompt = PARTICIPANT_RESPONSE_PROMPT.format(
            phase=phase.value,
            context=context_text,
            question=moderator_question,
        )

        response_text = await self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.9,
            max_tokens=180,
        )

        sentiment = self._sentiment_from_text(response_text)
        changed_mind = self._heuristic_shift(response_text)

        self.times_spoken += 1

        return DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id=self.persona.id,
            speaker_name=self.persona.name,
            content=response_text,
            phase=phase,
            turn_number=len(discussion_context) + 1,
            sentiment=sentiment,
            changed_mind=changed_mind,
        )

    def should_speak(self, phase: DiscussionPhase, turn: int, question: str | None = None) -> bool:
        del phase, turn
        if question and self.persona.name.lower() in question.lower():
            return True

        extraversion = self.persona.psychographics.ocean.extraversion
        if extraversion >= 70:
            threshold = 0.85
        elif extraversion >= 40:
            threshold = 0.60
        else:
            threshold = 0.35

        return random.random() < threshold

    async def _detect_opinion_shift(self, response_text: str) -> bool:
        prompt = OPINION_SHIFT_DETECTION_PROMPT.format(
            initial_opinion=self.persona.initial_opinion or "No initial opinion",
            initial_valence=self.persona.opinion_valence,
            response=response_text,
        )
        raw = await self.llm_client.complete(
            system_prompt="You classify opinion shifts.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=60,
        )
        try:
            parsed = json.loads(raw)
            return bool(parsed.get("changed_mind", False))
        except json.JSONDecodeError:
            return self._heuristic_shift(response_text)

    def _heuristic_shift(self, response_text: str) -> bool:
        if self.persona.opinion_valence is None:
            return False
        sentiment = self._sentiment_from_text(response_text)
        if sentiment is None:
            return False
        return (self.persona.opinion_valence < -0.2 and sentiment > 0.25) or (
            self.persona.opinion_valence > 0.2 and sentiment < -0.25
        )

    @staticmethod
    def _format_context(messages: Iterable[DiscussionMessage]) -> str:
        lines: list[str] = []
        for message in messages:
            lines.append(f"{message.speaker_name}: {message.content}")
        return "\n".join(lines) if lines else "No prior discussion yet."

    @staticmethod
    def _sentiment_from_text(text: str) -> float | None:
        lowered = text.lower()
        positive = len(re.findall(r"\b(like|love|useful|good|great|buy|helpful|positive)\b", lowered))
        negative = len(re.findall(r"\b(dislike|hate|bad|worry|concern|avoid|negative|skeptical)\b", lowered))
        total = positive + negative
        if total == 0:
            return 0.0
        score = (positive - negative) / total
        return max(-1.0, min(1.0, float(score)))

    @staticmethod
    def _level_text(value: float) -> str:
        if value >= 0.75:
            return "very"
        if value >= 0.45:
            return "moderately"
        return "not very"

    @staticmethod
    def _describe_openness(score: float) -> str:
        if score >= 80:
            return (
                "You're naturally curious and enjoy experimenting with unfamiliar ideas and products."
            )
        if score >= 50:
            return "You balance practical choices with occasional curiosity for new options."
        return "You prefer familiar, proven options and are cautious about novelty."

    @staticmethod
    def _describe_conscientiousness(score: float) -> str:
        if score >= 80:
            return "You're organized and deliberate, and you plan purchases carefully."
        if score >= 50:
            return "You're reasonably structured but can be flexible when needed."
        return "You're more spontaneous and less focused on rigid planning."

    @staticmethod
    def _describe_extraversion(score: float) -> str:
        if score >= 70:
            return "You're socially energetic, quick to speak, and comfortable leading discussions."
        if score >= 40:
            return "You're socially balanced and speak up when you have something useful to add."
        return "You're reserved and usually concise unless directly asked."

    @staticmethod
    def _describe_agreeableness(score: float) -> str:
        if score >= 70:
            return "You're cooperative and inclined to find common ground with others."
        if score >= 30:
            return "You're polite but still willing to disagree when it matters."
        return "You're independent-minded and do not go along just to keep the peace."

    @staticmethod
    def _describe_neuroticism(score: float) -> str:
        if score >= 70:
            return (
                "You tend to worry about risks and downside scenarios, so reassurance matters."
            )
        if score >= 40:
            return "You notice risks but usually keep concerns in proportion."
        return "You're emotionally steady and not easily rattled by uncertainty."
