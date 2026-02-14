from __future__ import annotations

import asyncio

from analysis.concept_scorer import ConceptScorer
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


def _persona(idx: int, valence: float, openness: float, price_sensitivity: float) -> Persona:
    return Persona(
        id=f"p{idx}",
        name=f"Persona {idx}",
        demographics=Demographics(
            age=25 + idx,
            gender="female" if idx % 2 else "male",
            income=45_000 + (idx * 15_000),
            education="bachelor",
            occupation="manager",
            location=Location(state="TX", metro_area="Austin", urbanicity="urban"),
            household_type="single",
            race_ethnicity="mixed",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=openness,
                conscientiousness=60,
                extraversion=50,
                agreeableness=55,
                neuroticism=40,
            ),
            vals_type="Achiever",
            schwartz_values=SchwartzValues(primary="achievement", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=price_sensitivity,
            brand_loyalty=0.4,
            research_tendency=0.6,
            impulse_tendency=0.3,
            social_influence=0.4,
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


def _transcript(personas: list[Persona]) -> DiscussionTranscript:
    config = DiscussionConfig(product_concept="AI nutrition coach", category="app")
    transcript = DiscussionTranscript(config=config, personas=personas)
    transcript.messages = [
        DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id=persona.id,
            speaker_name=persona.name,
            content="I can see the upside, but value and reliability matter to me.",
            phase=DiscussionPhase.REACTION,
            turn_number=idx + 1,
        )
        for idx, persona in enumerate(personas)
    ]
    return transcript


def test_concept_scores_populated_and_formula_correct() -> None:
    personas = [
        _persona(1, 0.8, 85, 0.2),
        _persona(2, 0.3, 65, 0.5),
        _persona(3, -0.4, 35, 0.8),
        _persona(4, -0.7, 20, 0.9),
    ]
    transcript = _transcript(personas)

    scorer = ConceptScorer(MockLLMClient())
    scores = asyncio.run(scorer.score_concept(transcript=transcript, personas=personas))

    metrics = [
        scores.purchase_intent,
        scores.overall_appeal,
        scores.uniqueness,
        scores.relevance,
        scores.believability,
        scores.value_perception,
    ]
    assert all(0.0 <= metric <= 1.0 for metric in metrics)

    expected = (
        scores.overall_appeal * 0.3
        + scores.uniqueness * 0.25
        + scores.purchase_intent * 0.25
        + scores.relevance * 0.2
    )
    assert scores.excitement_score == round(expected, 4)

    assert set(scores.participant_scores) == {persona.id for persona in personas}
    for participant_metrics in scores.participant_scores.values():
        assert set(participant_metrics) == {
            "purchase_intent",
            "overall_appeal",
            "uniqueness",
            "relevance",
            "believability",
            "value_perception",
        }
