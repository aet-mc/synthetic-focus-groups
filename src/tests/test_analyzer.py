from __future__ import annotations

import asyncio

from analysis.analyzer import AnalysisEngine
from analysis.models import ConceptScores
from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig
from discussion.simulator import DiscussionSimulator


def _concept_scores(excitement: float) -> ConceptScores:
    return ConceptScores(
        purchase_intent=0.7,
        overall_appeal=0.7,
        uniqueness=0.7,
        relevance=0.7,
        believability=0.6,
        value_perception=0.6,
        excitement_score=excitement,
        participant_scores={
            "p1": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p2": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p3": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p4": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p5": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p6": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p7": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
            "p8": {
                "purchase_intent": 4,
                "overall_appeal": 4,
                "uniqueness": 4,
                "relevance": 4,
                "believability": 4,
                "value_perception": 4,
            },
        },
    )


def test_full_analysis_pipeline_returns_complete_report() -> None:
    config = DiscussionConfig(product_concept="AI meal planner", category="app", num_personas=8)
    simulator = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    transcript = asyncio.run(simulator.run())

    engine = AnalysisEngine(llm_client=MockLLMClient())
    report = asyncio.run(engine.analyze(transcript))

    assert report.executive_summary
    assert report.recommendation
    assert report.confidence_level in {"high", "medium", "low"}

    assert report.concept_scores is not None
    assert report.themes
    assert report.sentiment_timeline.by_turn
    assert report.quotes.most_impactful
    assert report.segment_insights

    assert report.top_concerns
    assert report.top_opportunities
    assert report.suggested_improvements

    assert report.num_participants == len(transcript.personas)
    assert report.num_messages == len(transcript.messages)
    assert report.phases_completed == [phase.value for phase in transcript.config.phases]


def test_recommendation_logic_go_and_no_go() -> None:
    engine = AnalysisEngine(llm_client=MockLLMClient())

    high_rec, _ = asyncio.run(
        engine._generate_recommendation(
            concept_scores=_concept_scores(excitement=0.72),
            themes=[],
            concerns=[],
        )
    )
    low_rec, _ = asyncio.run(
        engine._generate_recommendation(
            concept_scores=_concept_scores(excitement=0.31),
            themes=[],
            concerns=[],
        )
    )

    assert high_rec.startswith("GO:")
    assert low_rec.startswith("NO-GO:")
