from __future__ import annotations

import asyncio

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig, DiscussionPhase, MessageRole
from discussion.simulator import DiscussionSimulator


def test_full_simulation_with_mock_llm() -> None:
    config = DiscussionConfig(
        product_concept="AI meal planner",
        category="app",
        num_personas=8,
    )
    simulator = DiscussionSimulator(config=config, llm_client=MockLLMClient())

    transcript = asyncio.run(simulator.run())

    phases_seen = {message.phase for message in transcript.messages}
    assert set(config.phases).issubset(phases_seen)

    participant_ids = {
        message.speaker_id
        for message in transcript.messages
        if message.role == MessageRole.PARTICIPANT
    }
    expected_ids = {persona.id for persona in transcript.personas}
    assert expected_ids.issubset(participant_ids)

    moderator_indices = [
        i for i, message in enumerate(transcript.messages) if message.role == MessageRole.MODERATOR
    ]
    participant_indices = [
        i for i, message in enumerate(transcript.messages) if message.role == MessageRole.PARTICIPANT
    ]
    assert moderator_indices
    assert participant_indices
    assert transcript.messages[0].role == MessageRole.MODERATOR

    found_bridge = False
    for index in range(1, len(transcript.messages) - 1):
        if (
            transcript.messages[index - 1].role == MessageRole.PARTICIPANT
            and transcript.messages[index].role == MessageRole.MODERATOR
            and transcript.messages[index + 1].role == MessageRole.PARTICIPANT
        ):
            found_bridge = True
            break
    assert found_bridge

    total_messages = len(transcript.messages)
    assert 40 <= total_messages <= 65

    assert DiscussionPhase.WARMUP in phases_seen
    assert DiscussionPhase.SYNTHESIS in phases_seen
