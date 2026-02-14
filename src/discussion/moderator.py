from __future__ import annotations

import random
from collections import Counter

from .llm_client import LLMClient
from .models import DiscussionConfig, DiscussionMessage, DiscussionPhase, MessageRole
from .participant import Participant
from .prompts import MODERATOR_QUESTION_PROMPT


class Moderator:
    def __init__(self, config: DiscussionConfig, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.discussion_guide: list[str] = []
        self._rng = random.Random(13)

    async def generate_discussion_guide(self) -> list[str]:
        guide: list[str] = []
        transcript_stub: list[DiscussionMessage] = []
        for phase in self.config.phases:
            for _ in range(self.config.questions_per_phase):
                question = await self.generate_question(
                    phase=phase,
                    transcript_so_far=transcript_stub,
                    quiet_personas=[],
                )
                guide.append(question)
        self.discussion_guide = guide
        return guide

    async def generate_question(
        self,
        phase: DiscussionPhase,
        transcript_so_far: list[DiscussionMessage],
        quiet_personas: list[str],
    ) -> str:
        summary = self._summarize_recent(transcript_so_far)
        user_prompt = MODERATOR_QUESTION_PROMPT.format(
            phase=phase.value,
            product_concept=self.config.product_concept,
            category=self.config.category,
            stimulus=self.config.stimulus_material or "None",
            summary=summary,
            quiet_personas=", ".join(quiet_personas) if quiet_personas else "None",
        )

        question = await self.llm_client.complete(
            system_prompt="You are a skilled focus group moderator.",
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=120,
        )

        if quiet_personas:
            present = any(name.lower() in question.lower() for name in quiet_personas)
            if not present:
                question = f"{question} {quiet_personas[0]}, I want your take as well."

        return question

    async def generate_followup(
        self,
        last_response: DiscussionMessage,
        phase: DiscussionPhase,
    ) -> str | None:
        del phase
        sentiment = abs(last_response.sentiment or 0.0)
        trigger = self._rng.random() < 0.2 or last_response.changed_mind or sentiment >= 0.75
        if not trigger:
            return None
        return (
            f"{last_response.speaker_name}, can you expand on that specific point a bit more?"
        )

    def select_respondents(
        self,
        participants: list[Participant],
        question: str,
        phase: DiscussionPhase,
        turn: int,
    ) -> list[Participant]:
        max_pick = min(6, self.config.max_responses_per_question)
        target = min(max_pick, max(3, int(round((3 + max_pick) / 2))))

        chosen: list[Participant] = []
        chosen_ids: set[str] = set()

        # Directly addressed participants should always respond.
        for participant in participants:
            if participant.persona.name.lower() in question.lower():
                chosen.append(participant)
                chosen_ids.add(participant.persona.id)

        # Ensure participants who have not spoken yet get opportunities.
        unspeaking = [p for p in participants if p.times_spoken == 0 and p.persona.id not in chosen_ids]
        unspeaking.sort(key=lambda p: p.persona.psychographics.ocean.extraversion, reverse=True)
        for participant in unspeaking:
            if len(chosen) >= target:
                break
            chosen.append(participant)
            chosen_ids.add(participant.persona.id)

        remaining = [p for p in participants if p.persona.id not in chosen_ids]
        self._rng.shuffle(remaining)
        for participant in remaining:
            if len(chosen) >= target:
                break
            if participant.should_speak(phase=phase, turn=turn, question=question):
                chosen.append(participant)
                chosen_ids.add(participant.persona.id)

        if len(chosen) < 3:
            for participant in remaining:
                if participant.persona.id in chosen_ids:
                    continue
                chosen.append(participant)
                chosen_ids.add(participant.persona.id)
                if len(chosen) >= 3:
                    break

        return chosen[:max_pick]

    @staticmethod
    def _summarize_recent(messages: list[DiscussionMessage]) -> str:
        if not messages:
            return "No discussion yet."

        participant_msgs = [m for m in messages if m.role == MessageRole.PARTICIPANT]
        if not participant_msgs:
            return "Moderator has introduced the session; participant viewpoints are pending."

        sentiment_counts = Counter(
            "positive" if (m.sentiment or 0) > 0.2 else "negative" if (m.sentiment or 0) < -0.2 else "neutral"
            for m in participant_msgs[-10:]
        )
        latest = participant_msgs[-3:]
        latest_snippets = " ".join(f"{m.speaker_name}: {m.content[:80]}" for m in latest)
        return (
            f"Recent tone -> positive: {sentiment_counts['positive']}, neutral: {sentiment_counts['neutral']}, "
            f"negative: {sentiment_counts['negative']}. Recent comments: {latest_snippets}"
        )
