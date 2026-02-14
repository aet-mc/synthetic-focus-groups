from __future__ import annotations

import asyncio

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig, DiscussionPhase
from discussion.moderator import Moderator
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


def _persona(idx: int, extraversion: float = 60.0) -> Persona:
    name = f"Person{idx}"
    return Persona(
        id=f"pid-{idx}",
        name=name,
        demographics=Demographics(
            age=25 + idx,
            gender="male" if idx % 2 == 0 else "female",
            income=50000 + (idx * 5000),
            education="bachelor",
            occupation="analyst",
            location=Location(state="TX", metro_area="Austin", urbanicity="urban"),
            household_type="single",
            race_ethnicity="mixed",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=55,
                conscientiousness=60,
                extraversion=extraversion,
                agreeableness=55,
                neuroticism=45,
            ),
            vals_type="Achiever",
            schwartz_values=SchwartzValues(primary="achievement", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.5,
            brand_loyalty=0.4,
            research_tendency=0.5,
            impulse_tendency=0.5,
            social_influence=0.5,
            risk_tolerance=0.5,
            category_engagement="medium",
            decision_style="balanced",
        ),
        voice=VoiceProfile(
            vocabulary_level="medium",
            verbosity="medium",
            hedging_tendency=0.4,
            emotional_expressiveness=0.4,
            assertiveness=0.5,
            humor_tendency=0.3,
            communication_style="conversational",
        ),
        initial_opinion="Neutral",
        opinion_valence=0.0,
    )


def test_generate_discussion_guide_has_10_questions() -> None:
    config = DiscussionConfig(product_concept="AI grocery planner", category="app")
    moderator = Moderator(config=config, llm_client=MockLLMClient())

    guide = asyncio.run(moderator.generate_discussion_guide())

    assert len(guide) == 10


def test_quiet_persona_gets_named_in_generated_question() -> None:
    config = DiscussionConfig(product_concept="AI grocery planner", category="app")
    moderator = Moderator(config=config, llm_client=MockLLMClient())

    question = asyncio.run(
        moderator.generate_question(
            phase=DiscussionPhase.EXPLORATION,
            transcript_so_far=[],
            quiet_personas=["Person3"],
        )
    )

    assert "Person3" in question


def test_select_respondents_returns_between_3_and_6() -> None:
    config = DiscussionConfig(product_concept="AI grocery planner", category="app")
    moderator = Moderator(config=config, llm_client=MockLLMClient())
    participants = [Participant(_persona(i, extraversion=65), MockLLMClient()) for i in range(8)]

    selected = moderator.select_respondents(
        participants=participants,
        question="What do you think of this concept?",
        phase=DiscussionPhase.DEEP_DIVE,
        turn=1,
    )

    assert 3 <= len(selected) <= 6
