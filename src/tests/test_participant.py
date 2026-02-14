from __future__ import annotations

import asyncio
import random

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionPhase
from discussion.participant import Participant
from persona_engine.models import (
    ConsumerProfile,
    Demographics,
    Location,
    OceanScores,
    Persona,
    Psychographics,
    SchwartzValues,
    VoiceProfile,
)


def _make_persona(name: str, extraversion: float, agreeableness: float = 50.0) -> Persona:
    return Persona(
        id=f"id-{name}",
        name=name,
        demographics=Demographics(
            age=34,
            gender="female",
            income=90000,
            education="bachelor",
            occupation="product manager",
            location=Location(state="CA", metro_area="San Francisco", urbanicity="urban"),
            household_type="single",
            race_ethnicity="white",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=88,
                conscientiousness=62,
                extraversion=extraversion,
                agreeableness=agreeableness,
                neuroticism=72,
            ),
            vals_type="Thinker",
            schwartz_values=SchwartzValues(primary="self-direction", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.7,
            brand_loyalty=0.4,
            research_tendency=0.8,
            impulse_tendency=0.2,
            social_influence=0.6,
            risk_tolerance=0.3,
            category_engagement="high",
            decision_style="analytical",
        ),
        voice=VoiceProfile(
            vocabulary_level="high",
            verbosity="moderate",
            hedging_tendency=0.3,
            emotional_expressiveness=0.4,
            assertiveness=0.7,
            humor_tendency=0.2,
            communication_style="direct",
        ),
        initial_opinion="Interesting but uncertain.",
        opinion_valence=-0.2,
    )


def test_build_system_prompt_uses_natural_language_without_trait_numbers() -> None:
    participant = Participant(_make_persona("Emma", extraversion=82), MockLLMClient())
    prompt = participant.build_system_prompt()

    assert "curious" in prompt.lower()
    assert "independent" not in prompt.lower()  # agreeableness is neutral in this persona
    assert "88" not in prompt
    assert "82" not in prompt
    assert "72" not in prompt


def test_should_speak_high_extraversion_more_than_low_extraversion() -> None:
    high = Participant(_make_persona("HighE", extraversion=85), MockLLMClient())
    low = Participant(_make_persona("LowE", extraversion=20), MockLLMClient())

    random.seed(123)
    high_count = sum(high.should_speak(DiscussionPhase.EXPLORATION, i) for i in range(100))
    random.seed(123)
    low_count = sum(low.should_speak(DiscussionPhase.EXPLORATION, i) for i in range(100))

    assert high_count > low_count


def test_respond_returns_message_with_speaker_and_phase() -> None:
    participant = Participant(_make_persona("Ava", extraversion=78), MockLLMClient())

    message = asyncio.run(
        participant.respond(
            moderator_question="What comes to mind when you hear this concept?",
            discussion_context=[],
            phase=DiscussionPhase.EXPLORATION,
        )
    )

    assert message.speaker_id == "id-Ava"
    assert message.speaker_name == "Ava"
    assert message.phase == DiscussionPhase.EXPLORATION
    assert message.content
