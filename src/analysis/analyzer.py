from __future__ import annotations

import json
from collections import Counter

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig, DiscussionTranscript, MessageRole

from .concept_scorer import ConceptScorer
from .models import AnalysisReport, ConceptScores, SentimentTimeline, Theme
from .prompts import EXECUTIVE_SUMMARY_PROMPT, RECOMMENDATION_PROMPT
from .quote_extractor import QuoteExtractor
from .segment_analyzer import SegmentAnalyzer
from .sentiment import SentimentAnalyzer
from .theme_extractor import ThemeExtractor


class AnalysisEngine:
    def __init__(self, llm_client=None):
        self.llm = llm_client or MockLLMClient()
        self.sentiment_analyzer = SentimentAnalyzer(self.llm)
        self.theme_extractor = ThemeExtractor(self.llm)
        self.concept_scorer = ConceptScorer(self.llm)
        self.quote_extractor = QuoteExtractor(self.llm)
        self.segment_analyzer = SegmentAnalyzer(self.llm)

    async def analyze(self, transcript: DiscussionTranscript) -> AnalysisReport:
        personas = transcript.personas or []
        self.sentiment_analyzer.set_personas(personas)
        self.theme_extractor.set_personas(personas)
        self.quote_extractor.set_personas(personas)

        participant_messages = [
            message
            for message in transcript.messages
            if message.role == MessageRole.PARTICIPANT
        ]

        sentiment_scores = await self.sentiment_analyzer.analyze_batch(participant_messages)
        for message, score in zip(participant_messages, sentiment_scores):
            message.sentiment = score
        sentiment_timeline = self.sentiment_analyzer.compute_timeline(participant_messages, sentiment_scores)

        themes = await self.theme_extractor.extract_themes(transcript=transcript)
        concept_scores = await self.concept_scorer.score_concept(transcript=transcript, personas=personas)
        quotes = await self.quote_extractor.extract_quotes(transcript=transcript, themes=themes)
        segment_insights = await self.segment_analyzer.analyze_segments(
            transcript=transcript,
            personas=personas,
            concept_scores=concept_scores,
            themes=themes,
        )

        top_concerns = self._derive_concerns(themes)
        top_opportunities = self._derive_opportunities(themes)
        suggested_improvements = [
            f"Address {concern.lower()} with clearer product execution."
            for concern in top_concerns
        ][:5]

        executive_summary = await self._generate_executive_summary(
            concept_scores=concept_scores,
            themes=themes,
            sentiment=sentiment_timeline,
            config=transcript.config,
        )
        recommendation, confidence_level = await self._generate_recommendation(
            concept_scores=concept_scores,
            themes=themes,
            concerns=top_concerns,
        )

        return AnalysisReport(
            executive_summary=executive_summary,
            recommendation=recommendation,
            confidence_level=confidence_level,
            concept_scores=concept_scores,
            themes=themes,
            sentiment_timeline=sentiment_timeline,
            quotes=quotes,
            segment_insights=segment_insights,
            top_concerns=top_concerns,
            top_opportunities=top_opportunities,
            suggested_improvements=suggested_improvements,
            num_participants=len(personas),
            num_messages=len(transcript.messages),
            phases_completed=[phase.value for phase in transcript.config.phases],
        )

    async def _generate_executive_summary(
        self,
        concept_scores: ConceptScores,
        themes: list[Theme],
        sentiment: SentimentTimeline,
        config: DiscussionConfig,
    ) -> str:
        if isinstance(self.llm, MockLLMClient):
            lead_theme = themes[0].name if themes else "overall concept response"
            return (
                f"This synthetic focus group evaluated a {config.category} concept: {config.product_concept}. "
                f"Overall sentiment was {sentiment.overall:.2f} with a {sentiment.trend} trajectory across phases. "
                f"The most prevalent theme was {lead_theme}. "
                f"Purchase intent Top-2-Box was {concept_scores.purchase_intent:.0%}, and the composite excitement score reached {concept_scores.excitement_score:.0%}."
            )

        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            concept_scores=concept_scores.model_dump_json(indent=2),
            themes=json.dumps([theme.model_dump() for theme in themes], indent=2),
            sentiment=sentiment.model_dump_json(indent=2),
            config=config.model_dump_json(indent=2),
        )
        return await self.llm.complete(
            system_prompt="You are an expert market research analyst.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=220,
        )

    async def _generate_recommendation(
        self,
        concept_scores: ConceptScores,
        themes: list[Theme],
        concerns: list[str],
    ) -> tuple[str, str]:
        excitement = concept_scores.excitement_score

        if excitement > 0.65:
            recommendation = "GO: Strong concept, proceed to development"
        elif excitement >= 0.45:
            recommendation = "ITERATE: Promising but needs refinement"
        else:
            recommendation = "NO-GO: Concept needs fundamental rethinking"

        confidence = self._confidence_level(concept_scores=concept_scores, themes=themes)

        if not isinstance(self.llm, MockLLMClient):
            prompt = RECOMMENDATION_PROMPT.format(
                concept_scores=concept_scores.model_dump_json(indent=2),
                themes=json.dumps([theme.model_dump() for theme in themes], indent=2),
                concerns=json.dumps(concerns, indent=2),
            )
            raw = await self.llm.complete(
                system_prompt="You are an objective product strategy advisor. Return JSON only.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=180,
            )
            try:
                parsed = json.loads(raw)
                recommendation = str(parsed.get("recommendation", recommendation))
                parsed_conf = str(parsed.get("confidence_level", confidence)).lower()
                if parsed_conf in {"high", "medium", "low"}:
                    confidence = parsed_conf
            except (TypeError, json.JSONDecodeError):
                pass

        return recommendation, confidence

    def _confidence_level(self, concept_scores: ConceptScores, themes: list[Theme]) -> str:
        participant_scores = concept_scores.participant_scores
        purchase_values = [
            scores["purchase_intent"]
            for scores in participant_scores.values()
            if "purchase_intent" in scores
        ]
        if not purchase_values:
            return "low"

        mean = sum(purchase_values) / len(purchase_values)
        variance = sum((value - mean) ** 2 for value in purchase_values) / len(purchase_values)
        stdev = variance**0.5

        dominant_theme_prevalence = themes[0].prevalence if themes else 0.0
        if stdev > 1.15:
            return "low"
        if stdev < 0.6 and dominant_theme_prevalence >= 0.6:
            return "high"
        if len(purchase_values) >= 8:
            return "medium"
        return "low"

    @staticmethod
    def _derive_concerns(themes: list[Theme]) -> list[str]:
        negatives = sorted(
            [theme for theme in themes if theme.sentiment < 0],
            key=lambda theme: theme.prevalence,
            reverse=True,
        )
        concerns = [theme.name for theme in negatives[:5]]
        if concerns:
            return concerns
        return ["Limited evidence of strong objections"]

    @staticmethod
    def _derive_opportunities(themes: list[Theme]) -> list[str]:
        positives = sorted(
            [theme for theme in themes if theme.sentiment >= 0],
            key=lambda theme: theme.prevalence,
            reverse=True,
        )
        opportunities = [theme.name for theme in positives[:5]]
        if opportunities:
            return opportunities
        return ["Clarify value proposition in follow-up testing"]
