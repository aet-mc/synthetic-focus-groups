from __future__ import annotations

import asyncio
import json

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionMessage, DiscussionPhase, MessageRole
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


def _make_persona(name: str = "TestUser", valence: float = -0.5) -> Persona:
    return Persona(
        id=f"id-{name}",
        name=name,
        demographics=Demographics(
            age=30,
            gender="female",
            income=80000,
            education="bachelor",
            occupation="engineer",
            location=Location(state="CA", metro_area="San Francisco", urbanicity="urban"),
            household_type="single",
            race_ethnicity="white",
        ),
        psychographics=Psychographics(
            ocean=OceanScores(
                openness=70, conscientiousness=60, extraversion=60,
                agreeableness=50, neuroticism=40,
            ),
            vals_type="Achiever",
            schwartz_values=SchwartzValues(primary="achievement", secondary="security"),
        ),
        consumer=ConsumerProfile(
            price_sensitivity=0.5, brand_loyalty=0.4, research_tendency=0.6,
            impulse_tendency=0.3, social_influence=0.5, risk_tolerance=0.4,
            category_engagement="medium", decision_style="analytical",
        ),
        voice=VoiceProfile(
            vocabulary_level="medium", verbosity="moderate", hedging_tendency=0.3,
            emotional_expressiveness=0.4, assertiveness=0.5, humor_tendency=0.2,
            communication_style="direct",
        ),
        initial_opinion="Skeptical about this product.",
        opinion_valence=valence,
    )


def _make_context(n: int = 4) -> list[DiscussionMessage]:
    msgs = []
    for i in range(n):
        msgs.append(DiscussionMessage(
            role=MessageRole.PARTICIPANT,
            speaker_id=f"id-other-{i}",
            speaker_name=f"Other{i}",
            content=f"I think this product looks promising in area {i}.",
            phase=DiscussionPhase.DEEP_DIVE,
            turn_number=i + 1,
        ))
    return msgs


class ShiftMockLLMClient(MockLLMClient):
    """Mock that returns controllable shift results."""

    def __init__(self, shift_response: dict | str | None = None):
        super().__init__()
        self._shift_response = shift_response

    async def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.9, max_tokens: int = 300) -> str:
        if "classify opinion shifts" in system_prompt.lower() or "shifted their opinion" in user_prompt.lower():
            if isinstance(self._shift_response, dict):
                return json.dumps(self._shift_response)
            if isinstance(self._shift_response, str):
                return self._shift_response
            return '{"reasoning": "no shift", "changed_mind": false, "shift_magnitude": "none", "new_valence": -0.5}'
        return await super().complete(system_prompt, user_prompt, temperature, max_tokens)


def test_opinion_shift_detected_via_llm() -> None:
    """LLM detects a shift and valence is updated."""
    persona = _make_persona(valence=-0.5)
    client = ShiftMockLLMClient({
        "reasoning": "Participant became positive",
        "changed_mind": True,
        "shift_magnitude": "moderate",
        "new_valence": 0.3,
    })
    participant = Participant(persona=persona, llm_client=client)
    context = _make_context()

    msg = asyncio.run(participant.respond(
        moderator_question="What do you think now?",
        discussion_context=context,
        phase=DiscussionPhase.DEEP_DIVE,
    ))

    assert msg.changed_mind is True
    assert persona.opinion_valence == 0.3


def test_opinion_shift_not_detected() -> None:
    """No shift means valence stays the same."""
    persona = _make_persona(valence=-0.5)
    client = ShiftMockLLMClient({
        "reasoning": "No change",
        "changed_mind": False,
        "shift_magnitude": "none",
        "new_valence": -0.5,
    })
    participant = Participant(persona=persona, llm_client=client)

    msg = asyncio.run(participant.respond(
        moderator_question="Any thoughts?",
        discussion_context=_make_context(),
        phase=DiscussionPhase.REACTION,
    ))

    assert msg.changed_mind is False
    assert persona.opinion_valence == -0.5


def test_opinion_shift_fallback_to_heuristic() -> None:
    """Invalid JSON falls back to heuristic."""
    persona = _make_persona(valence=-0.5)
    client = ShiftMockLLMClient("not valid json at all")
    participant = Participant(persona=persona, llm_client=client)

    # Should not crash; heuristic runs instead
    msg = asyncio.run(participant.respond(
        moderator_question="What do you think?",
        discussion_context=_make_context(),
        phase=DiscussionPhase.SYNTHESIS,
    ))

    assert isinstance(msg.changed_mind, bool)


def test_no_shift_detection_in_warmup() -> None:
    """Warmup phase should skip shift detection entirely."""
    persona = _make_persona(valence=-0.5)
    # This client would detect a shift if called
    client = ShiftMockLLMClient({
        "reasoning": "shift",
        "changed_mind": True,
        "shift_magnitude": "significant",
        "new_valence": 0.8,
    })
    participant = Participant(persona=persona, llm_client=client)

    msg = asyncio.run(participant.respond(
        moderator_question="Tell us about yourself.",
        discussion_context=[],
        phase=DiscussionPhase.WARMUP,
    ))

    assert msg.changed_mind is False
    assert persona.opinion_valence == -0.5  # Unchanged


def test_no_shift_detection_in_exploration() -> None:
    """Exploration phase should also skip shift detection."""
    persona = _make_persona(valence=0.3)
    client = ShiftMockLLMClient({
        "reasoning": "shift",
        "changed_mind": True,
        "shift_magnitude": "moderate",
        "new_valence": -0.5,
    })
    participant = Participant(persona=persona, llm_client=client)

    msg = asyncio.run(participant.respond(
        moderator_question="What comes to mind?",
        discussion_context=[],
        phase=DiscussionPhase.EXPLORATION,
    ))

    assert msg.changed_mind is False
    assert persona.opinion_valence == 0.3


def test_cumulative_valence_drift() -> None:
    """Multiple shifts should track cumulative drift."""
    persona = _make_persona(valence=-0.6)

    # First shift
    client1 = ShiftMockLLMClient({
        "reasoning": "slight positive",
        "changed_mind": True,
        "shift_magnitude": "slight",
        "new_valence": -0.2,
    })
    p1 = Participant(persona=persona, llm_client=client1)
    asyncio.run(p1.respond("Q1?", _make_context(), DiscussionPhase.DEEP_DIVE))
    assert persona.opinion_valence == -0.2

    # Second shift
    client2 = ShiftMockLLMClient({
        "reasoning": "now positive",
        "changed_mind": True,
        "shift_magnitude": "moderate",
        "new_valence": 0.3,
    })
    p2 = Participant(persona=persona, llm_client=client2)
    asyncio.run(p2.respond("Q2?", _make_context(), DiscussionPhase.REACTION))
    assert persona.opinion_valence == 0.3
