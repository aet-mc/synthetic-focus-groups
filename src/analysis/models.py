from __future__ import annotations

from pydantic import BaseModel


class Theme(BaseModel):
    name: str
    description: str
    prevalence: float
    sentiment: float
    supporting_quotes: list[str]
    participant_ids: list[str]
    phase_distribution: dict[str, int]


class ConceptScores(BaseModel):
    """Standard concept testing metrics, each on 1-5 scale then converted to Top-2-Box %."""

    purchase_intent: float
    overall_appeal: float
    uniqueness: float
    relevance: float
    believability: float
    value_perception: float
    excitement_score: float
    participant_scores: dict[str, dict[str, float]]


class SentimentTimeline(BaseModel):
    """Sentiment progression through the discussion."""

    by_phase: dict[str, float]
    by_turn: list[float]
    overall: float
    trend: str


class QuoteCollection(BaseModel):
    """Curated impactful quotes organized by sentiment and theme."""

    positive: list[dict]
    negative: list[dict]
    surprising: list[dict]
    most_impactful: list[dict]


class SegmentInsight(BaseModel):
    """Insights broken down by a demographic or psychographic segment."""

    segment_name: str
    segment_type: str
    participant_ids: list[str]
    avg_sentiment: float
    purchase_intent: float
    key_themes: list[str]
    distinguishing_quote: str
    differs_from_overall: str | None


class AnalysisReport(BaseModel):
    """Complete analysis output."""

    executive_summary: str
    recommendation: str
    confidence_level: str
    concept_scores: ConceptScores
    themes: list[Theme]
    sentiment_timeline: SentimentTimeline
    quotes: QuoteCollection
    segment_insights: list[SegmentInsight]
    top_concerns: list[str]
    top_opportunities: list[str]
    suggested_improvements: list[str]
    num_participants: int
    num_messages: int
    phases_completed: list[str]
