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
        self._cumulative_speak_counts: dict[str, int] = {}
        self._recent_question_speakers: list[set[str]] = []  # last N questions' speaker sets

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
        max_pick = min(self.config.max_responses_per_question, len(participants))
        target = min(max_pick, max(3, int(round((3 + max_pick) / 2))))

        # Initialize cumulative counts for new participants
        for p in participants:
            if p.persona.id not in self._cumulative_speak_counts:
                self._cumulative_speak_counts[p.persona.id] = 0

        # Build recent-speaker set (last 3 questions)
        recent_speakers: set[str] = set()
        for s in self._recent_question_speakers[-3:]:
            recent_speakers.update(s)

        chosen: list[Participant] = []
        chosen_ids: set[str] = set()

        # Directly addressed participants should always respond.
        for participant in participants:
            if participant.persona.name.lower() in question.lower():
                chosen.append(participant)
                chosen_ids.add(participant.persona.id)

        # For large pools (>12): reserve slots for quiet personas first
        use_pools = len(participants) > 12
        quiet_pool: set[str] = set()
        if use_pools:
            sorted_by_ext = sorted(
                participants, key=lambda p: p.persona.psychographics.ocean.extraversion
            )
            quiet_cutoff = int(len(sorted_by_ext) * 0.4)
            quiet_pool = set(p.persona.id for p in sorted_by_ext[:quiet_cutoff])
            min_quiet = max(1, int(target * 0.3))

            quiet_chosen = sum(1 for p in chosen if p.persona.id in quiet_pool)
            quiet_remaining = [
                p for p in participants
                if p.persona.id in quiet_pool and p.persona.id not in chosen_ids
            ]
            quiet_remaining.sort(
                key=lambda p: (
                    0 if p.persona.id not in recent_speakers else 1,
                    self._cumulative_speak_counts.get(p.persona.id, 0),
                )
            )
            for participant in quiet_remaining:
                if quiet_chosen >= min_quiet:
                    break
                if len(chosen) >= max_pick:
                    break
                chosen.append(participant)
                chosen_ids.add(participant.persona.id)
                quiet_chosen += 1

        # Ensure participants who have not spoken yet get opportunities.
        unspeaking = [p for p in participants if p.times_spoken == 0 and p.persona.id not in chosen_ids]
        unspeaking.sort(key=lambda p: p.persona.psychographics.ocean.extraversion, reverse=True)
        for participant in unspeaking:
            if len(chosen) >= target:
                break
            chosen.append(participant)
            chosen_ids.add(participant.persona.id)

        # Fill remaining with diversity bonus scoring
        remaining = [p for p in participants if p.persona.id not in chosen_ids]

        def _selection_score(p: Participant) -> float:
            score = self._rng.random()
            # Diversity bonus: not spoken in last 3 questions
            if p.persona.id not in recent_speakers:
                score += 2.0
            # Penalize frequent speakers
            score -= self._cumulative_speak_counts.get(p.persona.id, 0) * 0.3
            return score

        remaining.sort(key=_selection_score, reverse=True)
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

        result = chosen[:max_pick]

        # Track cumulative counts
        this_question_speakers: set[str] = set()
        for p in result:
            self._cumulative_speak_counts[p.persona.id] = (
                self._cumulative_speak_counts.get(p.persona.id, 0) + 1
            )
            this_question_speakers.add(p.persona.id)
        self._recent_question_speakers.append(this_question_speakers)

        return result

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
