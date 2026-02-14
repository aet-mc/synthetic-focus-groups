from __future__ import annotations

import asyncio

from analysis.models import ConceptScores, Theme
from analysis.segment_analyzer import SegmentAnalyzer
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
    age: int,
    income: int,
    gender: str,
    openness: float,
    agreeableness: float,
    vals_type: str,
    valence: float,
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
            location=Location(state="NY", metro_area="New York", urbanicity="urban"),
            household_type="single",
            race_ethnicity="mixed",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=openness,
                conscientiousness=60,
                extraversion=55,
                agreeableness=agreeableness,
                neuroticism=45,
            ),
            vals_type=vals_type,
            schwartz_values=SchwartzValues(primary="security", secondary="achievement"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.5,
            brand_loyalty=0.4,
            research_tendency=0.6,
            impulse_tendency=0.3,
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
        initial_opinion="initial",
        opinion_valence=valence,
    )


def _themes() -> list[Theme]:
    return [
        Theme(
            name="Value and Pricing",
            description="Cost/value feedback",
            prevalence=0.8,
            sentiment=-0.2,
            supporting_quotes=["Price is a concern."],
            participant_ids=["p1", "p2", "p3", "p4"],
            phase_distribution={"deep_dive": 4},
        ),
        Theme(
            name="Positive Momentum",
            description="Enthusiasm and intent",
            prevalence=0.5,
            sentiment=0.4,
            supporting_quotes=["I would buy this."],
            participant_ids=["p5", "p6"],
            phase_distribution={"reaction": 3},
        ),
    ]


def test_segment_analyzer_identifies_meaningful_segments() -> None:
    personas = [
        _persona(1, 28, 42_000, "female", 85, 65, "Experiencer", 0.7),
        _persona(2, 31, 48_000, "female", 78, 70, "Experiencer", 0.6),
        _persona(3, 47, 120_000, "male", 25, 30, "Thinker", -0.6),
        _persona(4, 54, 135_000, "male", 30, 35, "Thinker", -0.7),
        _persona(5, 39, 92_000, "female", 72, 60, "Achiever", 0.3),
        _persona(6, 58, 88_000, "male", 35, 45, "Believer", -0.2),
    ]

    config = DiscussionConfig(product_concept="AI budget planner", category="app")
    transcript = DiscussionTranscript(config=config, personas=personas)
    transcript.messages = [
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id=persona.id,
            speaker_name=persona.name,
            content=f"Participant {persona.id} gives detailed feedback with tradeoffs and purchase rationale.",
            phase=DiscussionPhase.REACTION,
            turn_number=idx + 1,
            sentiment=persona.opinion_valence,
        )
        for idx, persona in enumerate(personas)
    ]

    concept_scores = ConceptScores(
        purchase_intent=0.5,
        overall_appeal=0.5,
        uniqueness=0.5,
        relevance=0.5,
        believability=0.5,
        value_perception=0.5,
        excitement_score=0.5,
        participant_scores={
            "p1": {"purchase_intent": 5, "overall_appeal": 5, "uniqueness": 5, "relevance": 5, "believability": 4, "value_perception": 4},
            "p2": {"purchase_intent": 4, "overall_appeal": 4, "uniqueness": 4, "relevance": 4, "believability": 4, "value_perception": 4},
            "p3": {"purchase_intent": 1, "overall_appeal": 2, "uniqueness": 2, "relevance": 2, "believability": 2, "value_perception": 1},
            "p4": {"purchase_intent": 1, "overall_appeal": 1, "uniqueness": 1, "relevance": 1, "believability": 1, "value_perception": 1},
            "p5": {"purchase_intent": 4, "overall_appeal": 4, "uniqueness": 4, "relevance": 4, "believability": 4, "value_perception": 4},
            "p6": {"purchase_intent": 2, "overall_appeal": 2, "uniqueness": 2, "relevance": 2, "believability": 2, "value_perception": 2},
        },
    )

    analyzer = SegmentAnalyzer(MockLLMClient())
    segments = asyncio.run(
        analyzer.analyze_segments(
            transcript=transcript,
            personas=personas,
            concept_scores=concept_scores,
            themes=_themes(),
        )
    )

    assert len(segments) >= 2
    for segment in segments:
        assert segment.participant_ids


def test_segment_analyzer_excludes_non_distinct_segments() -> None:
    personas = [
        _persona(1, 30, 60_000, "female", 55, 55, "Achiever", 0.0),
        _persona(2, 34, 62_000, "male", 57, 53, "Achiever", 0.0),
        _persona(3, 38, 64_000, "female", 56, 54, "Achiever", 0.0),
        _persona(4, 42, 66_000, "male", 58, 52, "Achiever", 0.0),
    ]
    config = DiscussionConfig(product_concept="AI budget planner", category="app")
    transcript = DiscussionTranscript(config=config, personas=personas)
    transcript.messages = [
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id=persona.id,
            speaker_name=persona.name,
            content="Neutral statement about the concept.",
            phase=DiscussionPhase.EXPLORATION,
            turn_number=idx + 1,
            sentiment=0.0,
        )
        for idx, persona in enumerate(personas)
    ]

    concept_scores = ConceptScores(
        purchase_intent=1.0,
        overall_appeal=1.0,
        uniqueness=1.0,
        relevance=1.0,
        believability=1.0,
        value_perception=1.0,
        excitement_score=1.0,
        participant_scores={
            persona.id: {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            }
            for persona in personas
        },
    )

    analyzer = SegmentAnalyzer(MockLLMClient())
    segments = asyncio.run(
        analyzer.analyze_segments(
            transcript=transcript,
            personas=personas,
            concept_scores=concept_scores,
            themes=_themes(),
        )
    )

    assert segments == []
