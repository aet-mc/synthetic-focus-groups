from __future__ import annotations

import json
from collections import Counter, defaultdict

from .models import DiscussionTranscript, MessageRole


class TranscriptFormatter:
    @staticmethod
    def to_markdown(transcript: DiscussionTranscript) -> str:
        lines: list[str] = []
        current_phase = None

        for message in transcript.messages:
            if message.phase != current_phase:
                current_phase = message.phase
                title = current_phase.value.replace("_", " ").title()
                lines.append(f"## Phase: {title}")
                lines.append("")

            lines.append(f"**{message.speaker_name}:** {message.content}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def to_json(transcript: DiscussionTranscript) -> str:
        return transcript.model_dump_json(indent=2)

    @staticmethod
    def summary_stats(transcript: DiscussionTranscript) -> dict:
        by_role = Counter(m.role.value for m in transcript.messages)
        by_phase = Counter(m.phase.value for m in transcript.messages)
        by_participant = Counter(
            m.speaker_name for m in transcript.messages if m.role == MessageRole.PARTICIPANT
        )

        sentiments_by_phase: dict[str, list[float]] = defaultdict(list)
        for message in transcript.messages:
            if message.sentiment is not None:
                sentiments_by_phase[message.phase.value].append(message.sentiment)

        avg_sentiment = {
            phase: (sum(values) / len(values) if values else 0.0)
            for phase, values in sentiments_by_phase.items()
        }

        opinion_shifts = sum(
            1
            for message in transcript.messages
            if message.role == MessageRole.PARTICIPANT and message.changed_mind
        )

        most_active = by_participant.most_common(1)[0][0] if by_participant else None
        least_active = by_participant.most_common()[-1][0] if by_participant else None

        return {
            "total_messages_by_role": dict(by_role),
            "messages_per_phase": dict(by_phase),
            "messages_per_participant": dict(by_participant),
            "average_sentiment_by_phase": avg_sentiment,
            "opinion_shifts_detected": opinion_shifts,
            "most_active_participant": most_active,
            "least_active_participant": least_active,
        }
