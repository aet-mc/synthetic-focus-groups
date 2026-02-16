from __future__ import annotations

import asyncio

import pytest

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig, DiscussionPhase, MessageRole
from discussion.moderator import Moderator
from discussion.participant import Participant
from discussion.simulator import DiscussionSimulator
from persona_engine.demographics import STATE_REGION
from persona_engine.generator import PersonaGenerator
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
                openness=55, conscientiousness=60, extraversion=extraversion,
                agreeableness=55, neuroticism=45,
            ),
            vals_type="Achiever",
            schwartz_values=SchwartzValues(primary="achievement", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.5, brand_loyalty=0.4, research_tendency=0.5,
            impulse_tendency=0.5, social_influence=0.5, risk_tolerance=0.5,
            category_engagement="medium", decision_style="balanced",
        ),
        voice=VoiceProfile(
            vocabulary_level="medium", verbosity="medium", hedging_tendency=0.4,
            emotional_expressiveness=0.4, assertiveness=0.5, humor_tendency=0.3,
            communication_style="conversational",
        ),
        initial_opinion="Neutral",
        opinion_valence=0.0,
    )


# --- Validation tests ---

def test_num_personas_min_validation() -> None:
    with pytest.raises(ValueError, match="num_personas must be between 4 and 48"):
        DiscussionConfig(product_concept="test", category="test", num_personas=2)


def test_num_personas_max_validation() -> None:
    with pytest.raises(ValueError, match="num_personas must be between 4 and 48"):
        DiscussionConfig(product_concept="test", category="test", num_personas=50)


def test_num_personas_valid_range() -> None:
    for n in [4, 8, 16, 24, 48]:
        config = DiscussionConfig(product_concept="test", category="test", num_personas=n)
        assert config.num_personas == n


# --- Auto-scaling tests ---

def test_auto_scale_small_pool() -> None:
    config = DiscussionConfig(product_concept="test", category="test", num_personas=6)
    sim = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    sim._auto_scale_config()
    assert config.max_responses_per_question == 5
    assert config.questions_per_phase == 2


def test_auto_scale_medium_pool() -> None:
    config = DiscussionConfig(product_concept="test", category="test", num_personas=12)
    sim = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    sim._auto_scale_config()
    assert config.max_responses_per_question == 7


def test_auto_scale_large_pool() -> None:
    config = DiscussionConfig(product_concept="test", category="test", num_personas=20)
    sim = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    sim._auto_scale_config()
    assert config.max_responses_per_question == 8
    assert config.questions_per_phase == 3


def test_auto_scale_xlarge_pool() -> None:
    config = DiscussionConfig(product_concept="test", category="test", num_personas=30)
    sim = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    sim._auto_scale_config()
    assert config.max_responses_per_question == 10
    assert config.questions_per_phase == 3


# --- Persona generation diversity tests ---

def test_persona_generation_16_has_age_diversity() -> None:
    gen = PersonaGenerator(seed=42)
    personas = gen.generate(n=16, product_concept="test widget", category="gadgets")
    age_brackets = {gen._age_bracket(p.demographics.age) for p in personas}
    assert len(age_brackets) >= 3, f"Only {len(age_brackets)} age brackets: {age_brackets}"


def test_persona_generation_24_has_region_diversity() -> None:
    gen = PersonaGenerator(seed=42)
    personas = gen.generate(n=24, product_concept="test widget", category="gadgets")
    regions = {STATE_REGION.get(p.demographics.location.state, "unknown") for p in personas}
    assert len(regions) >= 4, f"Only {len(regions)} regions: {regions}"


# --- Respondent selection with large pools ---

def test_select_respondents_large_pool_includes_quiet() -> None:
    """For >12 participants, quiet personas should be represented."""
    config = DiscussionConfig(product_concept="test", category="test", num_personas=16,
                              max_responses_per_question=7)
    moderator = Moderator(config=config, llm_client=MockLLMClient())

    # Create participants with varying extraversion
    participants = []
    for i in range(16):
        ext = 20 + i * 5  # 20 to 95
        participants.append(Participant(_persona(i, extraversion=ext), MockLLMClient()))

    # Run selection multiple times
    quiet_ids = {p.persona.id for p in participants if p.persona.psychographics.ocean.extraversion < 50}

    total_quiet_selected = 0
    total_selected = 0
    for turn in range(6):
        selected = moderator.select_respondents(
            participants=participants,
            question="What do you think?",
            phase=DiscussionPhase.DEEP_DIVE,
            turn=turn,
        )
        total_selected += len(selected)
        total_quiet_selected += sum(1 for p in selected if p.persona.id in quiet_ids)

    # Over 6 rounds, quiet personas should appear at least a few times
    assert total_quiet_selected >= 3, f"Quiet personas only selected {total_quiet_selected} times in {total_selected} total"


# --- Full simulation with 16 personas ---

def test_simulation_16_personas() -> None:
    config = DiscussionConfig(
        product_concept="Smart water bottle",
        category="fitness",
        num_personas=16,
    )
    simulator = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    transcript = asyncio.run(simulator.run())

    participant_ids = {
        m.speaker_id for m in transcript.messages if m.role == MessageRole.PARTICIPANT
    }
    expected_ids = {p.id for p in transcript.personas}
    # All personas should have spoken at least once
    assert expected_ids.issubset(participant_ids)
    assert len(transcript.personas) == 16
