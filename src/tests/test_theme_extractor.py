from __future__ import annotations

import asyncio

from analysis.theme_extractor import ThemeExtractor
from discussion.llm_client import MockLLMClient
from discussion.models import (
    DiscussionConfig,
    DiscussionMessage,
    DiscussionPhase,
    DiscussionTranscript,
    MessageRole,
)
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


def _persona(
    idx: int,
    valence: float,
    age: int,
    income: int,
    gender: str,
    openness: float,
    agreeableness: float,
) -> Persona:
    return Persona(
        id=f"p{idx}",
        name=f"Person {idx}",
        demographics=Demographics(
            age=age,
            gender=gender,
            income=income,
            education="bachelor",
            occupation="analyst",
            location=Location(state="CA", metro_area="San Jose", urbanicity="urban"),
            household_type="single",
            race_ethnicity="mixed",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=openness,
                conscientiousness=60,
                extraversion=50,
                agreeableness=agreeableness,
                neuroticism=45,
            ),
            vals_type="Thinker",
            schwartz_values=SchwartzValues(primary="achievement", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.5,
            brand_loyalty=0.4,
            research_tendency=0.7,
            impulse_tendency=0.2,
            social_influence=0.3,
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
            humor_tendency=0.2,
            communication_style="conversational",
        ),
        initial_opinion="Initial",
        opinion_valence=valence,
    )


def _transcript() -> DiscussionTranscript:
    personas = [
        _persona(1, 0.7, 29, 45_000, "female", 80, 65),
        _persona(2, -0.6, 42, 110_000, "male", 28, 35),
        _persona(3, 0.2, 36, 82_000, "female", 65, 55),
        _persona(4, -0.2, 58, 60_000, "male", 35, 40),
    ]
    config = DiscussionConfig(product_concept="AI meal planner", category="app")
    transcript = DiscussionTranscript(config=config, personas=personas)
    transcript.messages = [
        DiscussionMessage(
            role=MessageRole.MODERATOR,
            speaker_id="moderator",
            speaker_name="Moderator",
            content="Let's discuss the concept.",
            phase=DiscussionPhase.EXPLORATION,
            turn_number=1,
        ),
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id="p1",
            speaker_name="Person 1",
            content="I like the idea, and if setup is simple I would buy it quickly.",
            phase=DiscussionPhase.EXPLORATION,
            turn_number=2,
            sentiment=0.6,
        ),
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id="p2",
            speaker_name="Person 2",
            content="I worry the price is high and I need proof this is reliable.",
            phase=DiscussionPhase.DEEP_DIVE,
            turn_number=3,
            sentiment=-0.7,
        ),
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id="p3",
            speaker_name="Person 3",
            content="The feature set feels new compared with alternatives and that is appealing.",
            phase=DiscussionPhase.DEEP_DIVE,
            turn_number=4,
            sentiment=0.3,
        ),
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id="p4",
            speaker_name="Person 4",
            content="I compare options carefully, and value for money plus trust would decide it.",
            phase=DiscussionPhase.REACTION,
            turn_number=5,
            sentiment=-0.1,
        ),
    ]
    return transcript


def test_extract_themes_returns_multiple_themes() -> None:
    transcript = _transcript()
    extractor = ThemeExtractor(MockLLMClient())
    extractor.set_personas(transcript.personas)

    themes = asyncio.run(extractor.extract_themes(transcript, max_themes=7))

    assert len(themes) >= 3
    for theme in themes:
        assert 0.0 <= theme.prevalence <= 1.0
        assert theme.supporting_quotes
