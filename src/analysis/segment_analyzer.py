from __future__ import annotations

from discussion.models import DiscussionTranscript, MessageRole
from persona_engine.models import Persona

from .models import ConceptScores, SegmentInsight, Theme


class SegmentAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_segments(
        self,
        transcript: DiscussionTranscript,
        personas: list[Persona],
        concept_scores: ConceptScores,
        themes: list[Theme],
    ) -> list[SegmentInsight]:
        del self.llm

        persona_map = {persona.id: persona for persona in personas}
        sentiment_by_participant: dict[str, list[float]] = {}
        quote_by_participant: dict[str, str] = {}

        for message in transcript.messages:
            if message.role != MessageRole.PARTICIPANT:
                continue
            score = self._message_sentiment(message, persona_map)
            sentiment_by_participant.setdefault(message.speaker_id, []).append(score)
            current = quote_by_participant.get(message.speaker_id)
            if current is None or len(message.content) > len(current):
                quote_by_participant[message.speaker_id] = message.content

        overall_sentiment_values = [
            value for values in sentiment_by_participant.values() for value in values
        ]
        overall_sentiment = (
            sum(overall_sentiment_values) / len(overall_sentiment_values)
            if overall_sentiment_values
            else 0.0
        )
        overall_purchase_intent = concept_scores.purchase_intent

        candidate_segments = self._build_segments(personas)

        insights: list[tuple[float, SegmentInsight]] = []
        for segment_name, segment_type, participant_ids in candidate_segments:
            if len(participant_ids) < 2:
                continue

            segment_sentiments: list[float] = []
            for participant_id in participant_ids:
                segment_sentiments.extend(sentiment_by_participant.get(participant_id, []))
            avg_sentiment = (
                sum(segment_sentiments) / len(segment_sentiments)
                if segment_sentiments
                else 0.0
            )

            purchase_intent = self._segment_top2box(
                participant_ids=participant_ids,
                participant_scores=concept_scores.participant_scores,
                metric="purchase_intent",
            )

            sentiment_delta = avg_sentiment - overall_sentiment
            purchase_delta = purchase_intent - overall_purchase_intent
            if abs(sentiment_delta) <= 0.10 and abs(purchase_delta) <= 0.10:
                continue

            key_themes = self._segment_themes(participant_ids, themes)
            quote = self._representative_quote(participant_ids, quote_by_participant)
            difference = self._difference_text(sentiment_delta=sentiment_delta, purchase_delta=purchase_delta)

            insight = SegmentInsight(
                segment_name=segment_name,
                segment_type=segment_type,
                participant_ids=participant_ids,
                avg_sentiment=round(avg_sentiment, 4),
                purchase_intent=round(purchase_intent, 4),
                key_themes=key_themes,
                distinguishing_quote=quote,
                differs_from_overall=difference,
            )
            magnitude = max(abs(sentiment_delta), abs(purchase_delta))
            insights.append((magnitude, insight))

        insights.sort(key=lambda pair: pair[0], reverse=True)
        return [insight for _, insight in insights]

    def _build_segments(self, personas: list[Persona]) -> list[tuple[str, str, list[str]]]:
        segments: list[tuple[str, str, list[str]]] = []

        def ids_for(filter_fn) -> list[str]:
            return [persona.id for persona in personas if filter_fn(persona)]

        segments.extend(
            [
                ("Age: Under 35", "demographic", ids_for(lambda p: p.demographics.age < 35)),
                (
                    "Age: 35-55",
                    "demographic",
                    ids_for(lambda p: 35 <= p.demographics.age <= 55),
                ),
                ("Age: Over 55", "demographic", ids_for(lambda p: p.demographics.age > 55)),
                (
                    "Income: Under $50K",
                    "demographic",
                    ids_for(lambda p: p.demographics.income < 50_000),
                ),
                (
                    "Income: $50K-$100K",
                    "demographic",
                    ids_for(lambda p: 50_000 <= p.demographics.income <= 100_000),
                ),
                (
                    "Income: Over $100K",
                    "demographic",
                    ids_for(lambda p: p.demographics.income > 100_000),
                ),
                (
                    "Psychographic: High Openness",
                    "psychographic",
                    ids_for(lambda p: p.psychographics.ocean.openness >= 60),
                ),
                (
                    "Psychographic: Low Openness",
                    "psychographic",
                    ids_for(lambda p: p.psychographics.ocean.openness < 40),
                ),
                (
                    "Psychographic: High Agreeableness",
                    "psychographic",
                    ids_for(lambda p: p.psychographics.ocean.agreeableness >= 60),
                ),
                (
                    "Psychographic: Low Agreeableness",
                    "psychographic",
                    ids_for(lambda p: p.psychographics.ocean.agreeableness < 40),
                ),
            ]
        )

        genders = sorted({persona.demographics.gender for persona in personas})
        for gender in genders:
            segments.append(
                (
                    f"Gender: {gender.title()}",
                    "demographic",
                    ids_for(lambda p, g=gender: p.demographics.gender == g),
                )
            )

        vals_types = sorted({persona.psychographics.vals_type for persona in personas})
        for vals_type in vals_types:
            segments.append(
                (
                    f"VALS: {vals_type}",
                    "psychographic",
                    ids_for(lambda p, v=vals_type: p.psychographics.vals_type == v),
                )
            )

        return segments

    @staticmethod
    def _segment_top2box(
        participant_ids: list[str], participant_scores: dict[str, dict[str, float]], metric: str
    ) -> float:
        values = [
            participant_scores[participant_id][metric]
            for participant_id in participant_ids
            if participant_id in participant_scores and metric in participant_scores[participant_id]
        ]
        if not values:
            return 0.0
        top_two = sum(1 for value in values if value >= 4.0)
        return top_two / len(values)

    @staticmethod
    def _segment_themes(participant_ids: list[str], themes: list[Theme]) -> list[str]:
        ranked: list[tuple[float, str]] = []
        id_set = set(participant_ids)
        for theme in themes:
            overlap = len(id_set.intersection(theme.participant_ids)) / max(1, len(id_set))
            if overlap > 0:
                ranked.append((overlap, theme.name))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [name for _, name in ranked[:3]]

    @staticmethod
    def _representative_quote(participant_ids: list[str], quote_by_participant: dict[str, str]) -> str:
        candidates = [quote_by_participant[pid] for pid in participant_ids if pid in quote_by_participant]
        if not candidates:
            return "No representative quote available."
        return max(candidates, key=len)

    @staticmethod
    def _difference_text(sentiment_delta: float, purchase_delta: float) -> str | None:
        parts: list[str] = []
        if abs(purchase_delta) > 0.10:
            direction = "higher" if purchase_delta > 0 else "lower"
            parts.append(f"purchase intent is {direction} by {abs(purchase_delta):.0%}")
        if abs(sentiment_delta) > 0.10:
            direction = "more positive" if sentiment_delta > 0 else "more negative"
            parts.append(f"sentiment is {direction} by {abs(sentiment_delta):.2f}")
        if not parts:
            return None
        return "; ".join(parts)

    @staticmethod
    def _message_sentiment(message, persona_map: dict[str, Persona]) -> float:
        if message.sentiment is not None:
            return float(message.sentiment)
        persona = persona_map.get(message.speaker_id)
        if persona and persona.opinion_valence is not None:
            return float(persona.opinion_valence)
        return 0.0
