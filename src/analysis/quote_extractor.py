from __future__ import annotations

from discussion.models import DiscussionTranscript, MessageRole
from persona_engine.models import Persona

from .models import QuoteCollection, Theme


class QuoteExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self._persona_map: dict[str, Persona] = {}

    def set_personas(self, personas: list[Persona]) -> None:
        self._persona_map = {persona.id: persona for persona in personas}

    async def extract_quotes(self, transcript: DiscussionTranscript, themes: list[Theme]) -> QuoteCollection:
        del themes

        participant_messages = [
            message
            for message in transcript.messages
            if message.role == MessageRole.PARTICIPANT
        ]
        if not participant_messages:
            return QuoteCollection(positive=[], negative=[], surprising=[], most_impactful=[])

        long_messages = [message for message in participant_messages if len(message.content.split()) > 20]
        candidates = long_messages if long_messages else [m for m in participant_messages if len(m.content.split()) > 8]

        scored: list[tuple[float, dict, float]] = []
        for message in candidates:
            valence = self._message_valence(message)
            quote_payload = {
                "quote": message.content,
                "speaker_name": message.speaker_name,
                "context": message.phase.value,
            }
            impact = abs(valence) + (len(message.content.split()) / 40.0) + (0.4 if message.changed_mind else 0.0)
            scored.append((impact, quote_payload, valence))

        scored.sort(key=lambda item: item[0], reverse=True)

        positive = [payload for _, payload, valence in scored if valence >= 0.2][:5]
        negative = [payload for _, payload, valence in scored if valence <= -0.2][:5]
        surprising = [
            payload
            for _, payload, _ in scored
            if (
                "but" in payload["quote"].lower()
                or "however" in payload["quote"].lower()
                or "though" in payload["quote"].lower()
            )
            or any(message.changed_mind and message.content == payload["quote"] for message in participant_messages)
        ][:5]

        most_impactful = [payload for _, payload, _ in scored[:5]]

        return QuoteCollection(
            positive=positive,
            negative=negative,
            surprising=surprising,
            most_impactful=most_impactful,
        )

    def _message_valence(self, message) -> float:
        if message.sentiment is not None:
            return float(message.sentiment)

        persona = self._persona_map.get(message.speaker_id)
        if persona and persona.opinion_valence is not None:
            return float(persona.opinion_valence)

        lowered = message.content.lower()
        if any(token in lowered for token in ("love", "great", "buy", "useful", "good")):
            return 0.4
        if any(token in lowered for token in ("worry", "concern", "avoid", "bad", "skeptical")):
            return -0.4
        return 0.0
