from __future__ import annotations

from pydantic import BaseModel

from analysis.models import ConceptScores


class ScorecardResult(BaseModel):
    """Quality scorecard for a single focus group run."""

    metric_independence: float  # 0-1, avg per-person spread normalized
    opinion_diversity: float  # 0-1, stdev of PI scores normalized
    score_distribution_shape: str  # "clustered" | "moderate" | "polarized"
    score_distribution_stdev: float
    sentiment_score_alignment: float  # -1 to 1 (Pearson correlation)
    discussion_quality: float  # 0-1, composite
    participation_balance: float  # 0-1, 1 = perfectly balanced
    mind_change_rate: float  # 0-1
    overall_grade: str  # A/B/C/D
    issues: list[str]  # human-readable warnings


class RunResult(BaseModel):
    """Summary of a single focus group run."""

    seed: int
    concept_scores: ConceptScores
    recommendation: str
    confidence_level: str
    num_messages: int
    themes: list[str]  # theme names only
    scorecard: ScorecardResult


class ConsistencyReport(BaseModel):
    """Cross-run stability analysis."""

    runs: list[RunResult]
    scorecards: list[ScorecardResult]
    # Cross-run metrics
    score_cv: dict[str, float]  # metric name → coefficient of variation
    theme_overlap: float  # Jaccard similarity across runs
    recommendation_consistent: bool
    stability_grade: str  # A/B/C/D
    combined_grade: str  # worst of quality + stability
    summary: str  # human-readable paragraph
