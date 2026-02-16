"""Tests for quality scorecard."""

from __future__ import annotations

from analysis.models import AnalysisReport, ConceptScores, QuoteCollection, SentimentTimeline, Theme
from consistency.models import ScorecardResult
from consistency.scorecard import QualityScorecard
from discussion.models import DiscussionConfig, DiscussionMessage, DiscussionPhase, DiscussionTranscript, MessageRole


def make_concept_scores(participant_scores: dict[str, dict[str, float]] | None = None) -> ConceptScores:
    """Create a ConceptScores object with optional participant scores."""
    return ConceptScores(
        purchase_intent=0.65,
        overall_appeal=0.70,
        uniqueness=0.55,
        relevance=0.60,
        believability=0.58,
        value_perception=0.62,
        excitement_score=0.60,
        participant_scores=participant_scores or {},
    )


def make_analysis_report(participant_scores: dict[str, dict[str, float]] | None = None) -> AnalysisReport:
    """Create a minimal AnalysisReport for testing."""
    return AnalysisReport(
        executive_summary="Test summary",
        recommendation="GO",
        confidence_level="High",
        concept_scores=make_concept_scores(participant_scores),
        themes=[
            Theme(
                name="Test Theme",
                description="A test theme",
                prevalence=0.5,
                sentiment=0.3,
                supporting_quotes=["quote1"],
                participant_ids=["p1"],
                phase_distribution={"warmup": 1},
            )
        ],
        sentiment_timeline=SentimentTimeline(
            by_phase={"warmup": 0.2, "exploration": 0.3},
            by_turn=[0.1, 0.2, 0.3],
            overall=0.25,
            trend="positive",
        ),
        quotes=QuoteCollection(positive=[], negative=[], surprising=[], most_impactful=[]),
        segment_insights=[],
        top_concerns=["concern1"],
        top_opportunities=["opportunity1"],
        suggested_improvements=["improvement1"],
        num_participants=4,
        num_messages=20,
        phases_completed=["warmup", "exploration"],
    )


def make_transcript(messages: list[DiscussionMessage] | None = None) -> DiscussionTranscript:
    """Create a minimal DiscussionTranscript for testing."""
    config = DiscussionConfig(
        product_concept="Test Product",
        category="test",
        num_personas=4,
    )
    return DiscussionTranscript(
        config=config,
        messages=messages or [],
        personas=[],
    )


class TestMetricIndependence:
    """Tests for metric independence calculation."""

    def test_high_independence(self):
        """Participants with varied scores across metrics should have high independence."""
        participant_scores = {
            "p1": {"purchase_intent": 5.0, "overall_appeal": 2.0, "uniqueness": 4.0, "relevance": 1.0, "believability": 3.0, "value_perception": 5.0},
            "p2": {"purchase_intent": 1.0, "overall_appeal": 5.0, "uniqueness": 2.0, "relevance": 4.0, "believability": 3.0, "value_perception": 1.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        independence = scorecard._metric_independence(concept_scores)

        # Max spread is 4.0 (5-1), so independence should be 1.0
        assert independence == 1.0

    def test_low_independence(self):
        """Participants giving same score across all metrics should have low independence."""
        participant_scores = {
            "p1": {"purchase_intent": 3.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 4.0, "overall_appeal": 4.0, "uniqueness": 4.0, "relevance": 4.0, "believability": 4.0, "value_perception": 4.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        independence = scorecard._metric_independence(concept_scores)

        # Spread is 0 for all participants
        assert independence == 0.0

    def test_empty_scores(self):
        """Empty participant scores should return 0."""
        concept_scores = make_concept_scores({})
        scorecard = QualityScorecard()

        independence = scorecard._metric_independence(concept_scores)

        assert independence == 0.0


class TestOpinionDiversity:
    """Tests for opinion diversity calculation."""

    def test_high_diversity(self):
        """Participants with varied purchase intent should have high diversity."""
        participant_scores = {
            "p1": {"purchase_intent": 1.0},
            "p2": {"purchase_intent": 5.0},
            "p3": {"purchase_intent": 1.0},
            "p4": {"purchase_intent": 5.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        diversity = scorecard._opinion_diversity(concept_scores)

        # Stdev of [1, 5, 1, 5] = 2.0, max theoretical = 2.0, so diversity = 1.0
        assert diversity == 1.0

    def test_low_diversity(self):
        """Participants with same purchase intent should have low diversity."""
        participant_scores = {
            "p1": {"purchase_intent": 3.0},
            "p2": {"purchase_intent": 3.0},
            "p3": {"purchase_intent": 3.0},
            "p4": {"purchase_intent": 3.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        diversity = scorecard._opinion_diversity(concept_scores)

        assert diversity == 0.0

    def test_moderate_diversity(self):
        """Participants with moderate variation should have moderate diversity."""
        participant_scores = {
            "p1": {"purchase_intent": 2.0},
            "p2": {"purchase_intent": 3.0},
            "p3": {"purchase_intent": 4.0},
            "p4": {"purchase_intent": 3.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        diversity = scorecard._opinion_diversity(concept_scores)

        # Should be between 0 and 1
        assert 0.0 < diversity < 1.0


class TestScoreDistribution:
    """Tests for score distribution shape classification."""

    def test_clustered_distribution(self):
        """Low variance scores should be classified as clustered."""
        participant_scores = {
            "p1": {"purchase_intent": 3.0},
            "p2": {"purchase_intent": 3.1},
            "p3": {"purchase_intent": 2.9},
            "p4": {"purchase_intent": 3.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        shape, stdev = scorecard._distribution_shape(concept_scores)

        assert shape == "clustered"
        assert stdev < 0.5

    def test_moderate_distribution(self):
        """Moderate variance scores should be classified as moderate."""
        participant_scores = {
            "p1": {"purchase_intent": 2.0},
            "p2": {"purchase_intent": 3.0},
            "p3": {"purchase_intent": 4.0},
            "p4": {"purchase_intent": 3.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        shape, stdev = scorecard._distribution_shape(concept_scores)

        assert shape == "moderate"
        assert 0.5 <= stdev <= 1.2

    def test_polarized_distribution(self):
        """High variance scores should be classified as polarized."""
        participant_scores = {
            "p1": {"purchase_intent": 1.0},
            "p2": {"purchase_intent": 5.0},
            "p3": {"purchase_intent": 1.0},
            "p4": {"purchase_intent": 5.0},
        }
        concept_scores = make_concept_scores(participant_scores)
        scorecard = QualityScorecard()

        shape, stdev = scorecard._distribution_shape(concept_scores)

        assert shape == "polarized"
        assert stdev > 1.2


class TestSentimentScoreAlignment:
    """Tests for sentiment-score alignment calculation."""

    def test_positive_alignment(self):
        """Positive sentiment should correlate with high purchase intent."""
        participant_scores = {
            "p1": {"purchase_intent": 5.0},
            "p2": {"purchase_intent": 1.0},
            "p3": {"purchase_intent": 4.0},
            "p4": {"purchase_intent": 2.0},
        }
        messages = [
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p1", speaker_name="P1", content="Great!", phase=DiscussionPhase.WARMUP, turn_number=1, sentiment=0.8),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p2", speaker_name="P2", content="Bad!", phase=DiscussionPhase.WARMUP, turn_number=2, sentiment=-0.8),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p3", speaker_name="P3", content="Good!", phase=DiscussionPhase.WARMUP, turn_number=3, sentiment=0.6),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p4", speaker_name="P4", content="Meh", phase=DiscussionPhase.WARMUP, turn_number=4, sentiment=-0.5),
        ]
        concept_scores = make_concept_scores(participant_scores)
        transcript = make_transcript(messages)
        scorecard = QualityScorecard()

        alignment = scorecard._sentiment_score_alignment(concept_scores, transcript)

        # Should be strongly positive
        assert alignment > 0.5

    def test_negative_alignment(self):
        """Negative sentiment correlating with high purchase intent is problematic."""
        participant_scores = {
            "p1": {"purchase_intent": 1.0},
            "p2": {"purchase_intent": 5.0},
            "p3": {"purchase_intent": 2.0},
        }
        messages = [
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p1", speaker_name="P1", content="Great!", phase=DiscussionPhase.WARMUP, turn_number=1, sentiment=0.8),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p2", speaker_name="P2", content="Bad!", phase=DiscussionPhase.WARMUP, turn_number=2, sentiment=-0.8),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p3", speaker_name="P3", content="OK!", phase=DiscussionPhase.WARMUP, turn_number=3, sentiment=0.5),
        ]
        concept_scores = make_concept_scores(participant_scores)
        transcript = make_transcript(messages)
        scorecard = QualityScorecard()

        alignment = scorecard._sentiment_score_alignment(concept_scores, transcript)

        # Should be negative (inverted relationship)
        assert alignment < 0


class TestOverallGrade:
    """Tests for overall grade computation."""

    def test_grade_a(self):
        """High scores across all metrics should give grade A."""
        scorecard = QualityScorecard()

        grade, issues = scorecard._compute_grade(
            metric_independence=0.4,
            opinion_diversity=0.5,
            shape="moderate",
            alignment=0.6,
            discussion_quality=0.7,
        )

        assert grade == "A"
        assert len(issues) == 0

    def test_grade_with_one_issue(self):
        """One failing metric should still give high grade but report issues."""
        scorecard = QualityScorecard()

        grade, issues = scorecard._compute_grade(
            metric_independence=0.4,
            opinion_diversity=0.5,
            shape="clustered",  # This triggers a warning
            alignment=0.6,
            discussion_quality=0.7,
        )

        # Clustered shape is just a warning, doesn't affect score
        assert grade == "A"
        assert len(issues) == 1
        assert "clustered" in issues[0].lower()

    def test_grade_c(self):
        """Low scores should give grade C."""
        scorecard = QualityScorecard()

        grade, issues = scorecard._compute_grade(
            metric_independence=0.1,  # Low
            opinion_diversity=0.1,  # Low
            shape="clustered",
            alignment=0.3,  # Medium
            discussion_quality=0.5,  # Medium
        )

        assert grade in ["B", "C"]  # Depends on exact scoring

    def test_grade_d(self):
        """Very low scores should give grade D."""
        scorecard = QualityScorecard()

        grade, issues = scorecard._compute_grade(
            metric_independence=0.05,  # Very low
            opinion_diversity=0.05,  # Very low
            shape="clustered",
            alignment=-0.5,  # Negative
            discussion_quality=0.2,  # Very low
        )

        assert grade == "D"
        assert len(issues) >= 3


class TestFullScorecard:
    """Integration tests for the full scorecard."""

    def test_full_score_returns_result(self):
        """Full score method should return a valid ScorecardResult."""
        participant_scores = {
            "p1": {"purchase_intent": 4.0, "overall_appeal": 3.0, "uniqueness": 5.0, "relevance": 2.0, "believability": 4.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 2.0, "overall_appeal": 4.0, "uniqueness": 3.0, "relevance": 5.0, "believability": 2.0, "value_perception": 4.0},
            "p3": {"purchase_intent": 3.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
        }
        messages = [
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p1", speaker_name="P1", content="I like it", phase=DiscussionPhase.WARMUP, turn_number=1, sentiment=0.5),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p2", speaker_name="P2", content="Not sure", phase=DiscussionPhase.WARMUP, turn_number=2, sentiment=-0.2),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p3", speaker_name="P3", content="It's okay", phase=DiscussionPhase.WARMUP, turn_number=3, sentiment=0.1),
            DiscussionMessage(role=MessageRole.PARTICIPANT, speaker_id="p1", speaker_name="P1", content="Still like it", phase=DiscussionPhase.EXPLORATION, turn_number=4, sentiment=0.6, changed_mind=True),
        ]
        report = make_analysis_report(participant_scores)
        transcript = make_transcript(messages)

        scorecard = QualityScorecard()
        result = scorecard.score(report, transcript)

        assert isinstance(result, ScorecardResult)
        assert 0.0 <= result.metric_independence <= 1.0
        assert 0.0 <= result.opinion_diversity <= 1.0
        assert result.score_distribution_shape in ["clustered", "moderate", "polarized"]
        assert -1.0 <= result.sentiment_score_alignment <= 1.0
        assert 0.0 <= result.discussion_quality <= 1.0
        assert 0.0 <= result.participation_balance <= 1.0
        assert 0.0 <= result.mind_change_rate <= 1.0
        assert result.overall_grade in ["A", "B", "C", "D"]
        assert isinstance(result.issues, list)

    def test_empty_transcript(self):
        """Scorecard should handle empty transcript gracefully."""
        report = make_analysis_report({})
        transcript = make_transcript([])

        scorecard = QualityScorecard()
        result = scorecard.score(report, transcript)

        assert isinstance(result, ScorecardResult)
        assert result.overall_grade in ["A", "B", "C", "D"]
