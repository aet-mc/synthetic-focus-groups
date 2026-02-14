from __future__ import annotations

from discussion.models import (
    DiscussionConfig,
    DiscussionMessage,
    DiscussionPhase,
    DiscussionTranscript,
    MessageRole,
)
from discussion.transcript import TranscriptFormatter


def _sample_transcript() -> DiscussionTranscript:
    config = DiscussionConfig(product_concept="Smart bottle", category="fitness")
    transcript = DiscussionTranscript(config=config, personas=[])
    transcript.messages.extend(
        [
            DiscussionMessage(
                role=MessageRole.MODERATOR,
                speaker_id="moderator",
                speaker_name="Moderator",
                content="Welcome everyone.",
                phase=DiscussionPhase.WARMUP,
                turn_number=1,
            ),
            DiscussionMessage(
                role=MessageRole.PARTICIPANT,
                speaker_id="p1",
                speaker_name="Emma",
                content="I like gadgets like this.",
                phase=DiscussionPhase.WARMUP,
                turn_number=2,
                sentiment=0.8,
            ),
            DiscussionMessage(
                role=MessageRole.PARTICIPANT,
                speaker_id="p2",
                speaker_name="David",
                content="I worry about the price.",
                phase=DiscussionPhase.DEEP_DIVE,
                turn_number=3,
                sentiment=-0.7,
                changed_mind=True,
            ),
        ]
    )
    return transcript


def test_to_markdown_includes_phase_headers() -> None:
    transcript = _sample_transcript()
    markdown = TranscriptFormatter.to_markdown(transcript)

    assert "## Phase: Warmup" in markdown
    assert "## Phase: Deep Dive" in markdown
    assert "**Moderator:**" in markdown


def test_summary_stats_returns_expected_counts() -> None:
    transcript = _sample_transcript()
    stats = TranscriptFormatter.summary_stats(transcript)

    assert stats["total_messages_by_role"]["participant"] == 2
    assert stats["messages_per_phase"]["warmup"] == 2
    assert stats["messages_per_participant"]["Emma"] == 1
    assert stats["opinion_shifts_detected"] == 1
