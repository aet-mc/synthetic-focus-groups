from __future__ import annotations

from persona_engine.generator import PersonaGenerator

from .llm_client import LLMClient, MockLLMClient
from .models import DiscussionConfig, DiscussionMessage, DiscussionPhase, DiscussionTranscript, MessageRole
from .moderator import Moderator
from .participant import Participant


class DiscussionSimulator:
    def __init__(self, config: DiscussionConfig, llm_client: LLMClient | None = None):
        self.config = config
        self.llm_client = llm_client or MockLLMClient(model=config.model)

    async def run(self) -> DiscussionTranscript:
        generator = PersonaGenerator(seed=42)
        personas = generator.generate(
            n=self.config.num_personas,
            product_concept=self.config.product_concept,
            category=self.config.category,
        )

        participants = [Participant(persona=persona, llm_client=self.llm_client) for persona in personas]
        moderator = Moderator(config=self.config, llm_client=self.llm_client)

        transcript = DiscussionTranscript(config=self.config, personas=personas)

        for phase in self.config.phases:
            await self._run_phase(phase, participants, moderator, transcript)

        # Guarantee every persona has at least one contribution.
        missing = [p for p in participants if p.times_spoken == 0]
        if missing:
            recovery_question = "Before we close, I want to hear from anyone we have not heard much from yet."
            transcript.messages.append(
                DiscussionMessage(
                    role=MessageRole.MODERATOR,
                    speaker_id="moderator",
                    speaker_name="Moderator",
                    content=recovery_question,
                    phase=DiscussionPhase.SYNTHESIS,
                    turn_number=len(transcript.messages) + 1,
                )
            )
            for participant in missing:
                response = await participant.respond(
                    moderator_question=recovery_question,
                    discussion_context=transcript.messages,
                    phase=DiscussionPhase.SYNTHESIS,
                )
                transcript.messages.append(response)

        return transcript

    async def _run_phase(
        self,
        phase: DiscussionPhase,
        participants: list[Participant],
        moderator: Moderator,
        transcript: DiscussionTranscript,
    ) -> None:
        for question_idx in range(self.config.questions_per_phase):
            quiet_personas = self._quiet_persona_names(transcript, participants)
            question = await moderator.generate_question(
                phase=phase,
                transcript_so_far=transcript.messages,
                quiet_personas=quiet_personas,
            )

            transcript.messages.append(
                DiscussionMessage(
                    role=MessageRole.MODERATOR,
                    speaker_id="moderator",
                    speaker_name="Moderator",
                    content=question,
                    phase=phase,
                    turn_number=len(transcript.messages) + 1,
                )
            )

            respondents = moderator.select_respondents(
                participants=participants,
                question=question,
                phase=phase,
                turn=question_idx,
            )

            last_response: DiscussionMessage | None = None
            for respondent in respondents:
                response = await respondent.respond(
                    moderator_question=question,
                    discussion_context=transcript.messages,
                    phase=phase,
                )
                transcript.messages.append(response)
                last_response = response

            # Keep follow-ups occasional: at most one per question.
            if last_response is not None:
                followup = await moderator.generate_followup(last_response=last_response, phase=phase)
                if followup:
                    target = next(
                        (p for p in participants if p.persona.id == last_response.speaker_id),
                        None,
                    )
                    if target is not None:
                        transcript.messages.append(
                            DiscussionMessage(
                                role=MessageRole.MODERATOR,
                                speaker_id="moderator",
                                speaker_name="Moderator",
                                content=followup,
                                phase=phase,
                                turn_number=len(transcript.messages) + 1,
                                replied_to=target.persona.id,
                            )
                        )
                        followup_response = await target.respond(
                            moderator_question=followup,
                            discussion_context=transcript.messages,
                            phase=phase,
                        )
                        transcript.messages.append(followup_response)

    @staticmethod
    def _quiet_persona_names(
        transcript: DiscussionTranscript, participants: list[Participant], limit: int = 2
    ) -> list[str]:
        counts = {participant.persona.id: 0 for participant in participants}
        names = {participant.persona.id: participant.persona.name for participant in participants}

        for message in transcript.messages:
            if message.role == MessageRole.PARTICIPANT:
                counts[message.speaker_id] = counts.get(message.speaker_id, 0) + 1

        sorted_ids = sorted(counts, key=lambda pid: counts[pid])
        return [names[pid] for pid in sorted_ids[:limit]]
