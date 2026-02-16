"""Tests for the quality scorecard and consistency scoring system."""
from __future__ import annotations

from analysis.models import AnalysisReport, ConceptScores, QuoteCollection, SentimentTimeline, Theme
from consistency.scorecard import QualityScorecard
from discussion.models import DiscussionConfig, DiscussionMessage, DiscussionPhase, DiscussionTranscript, MessageRole


def _make_concept_scores(participant_scores: dict[str, dict[str, float]]) -> ConceptScores:
    """Build ConceptScores from raw participant data."""
    metrics = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception"]
    agg = {}
    for m in metrics:
        vals = [s[m] for s in participant_scores.values() if m in s]
        top2 = sum(1 for v in vals if v >= 3.5) / len(vals) if vals else 0
        agg[m] = round(top2, 4)
    excitement = agg["overall_appeal"] * 0.3 + agg["uniqueness"] * 0.25 + agg["purchase_intent"] * 0.25 + agg["relevance"] * 0.2
    return ConceptScores(
        **agg,
        excitement_score=round(excitement, 4),
        participant_scores=participant_scores,
    )


def _make_transcript(messages_data: list[dict], config: DiscussionConfig | None = None) -> DiscussionTranscript:
    """Build a minimal transcript from message dicts."""
    if config is None:
        config = DiscussionConfig(product_concept="Test product", category="test", num_personas=4)
    messages = []
    for i, md in enumerate(messages_data):
        messages.append(DiscussionMessage(
            role=md.get("role", MessageRole.PARTICIPANT),
            speaker_id=md.get("speaker_id", "p1"),
            speaker_name=md.get("speaker_name", "Test"),
            content=md.get("content", "test message"),
            phase=md.get("phase", DiscussionPhase.WARMUP),
            turn_number=md.get("turn_number", i + 1),
            sentiment=md.get("sentiment"),
            changed_mind=md.get("changed_mind", False),
        ))
    return DiscussionTranscript(config=config, messages=messages, personas=[])


class TestMetricIndependence:
    def test_high_independence(self):
        """Participants with divergent metrics should score high."""
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 1.0, "overall_appeal": 3.0, "uniqueness": 5.0, "relevance": 2.0, "believability": 4.0, "value_perception": 1.5},
            "p2": {"purchase_intent": 4.0, "overall_appeal": 2.0, "uniqueness": 4.5, "relevance": 1.0, "believability": 3.0, "value_perception": 5.0},
        })
        result = sc._metric_independence(scores)
        assert result >= 0.75, f"Expected high independence, got {result}"

    def test_low_independence(self):
        """Participants with flat scores should score low."""
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 3.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 2.0, "overall_appeal": 2.0, "uniqueness": 2.0, "relevance": 2.0, "believability": 2.0, "value_perception": 2.0},
        })
        result = sc._metric_independence(scores)
        assert result == 0.0, f"Expected 0.0 independence, got {result}"

    def test_empty_scores(self):
        sc = QualityScorecard()
        scores = _make_concept_scores({})
        assert sc._metric_independence(scores) == 0.0


class TestOpinionDiversity:
    def test_high_diversity(self):
        """Spread of PI from 1 to 5 should be high diversity."""
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 1.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 5.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
        })
        result = sc._opinion_diversity(scores)
        assert result >= 0.9, f"Expected high diversity, got {result}"

    def test_low_diversity(self):
        """Everyone at 3.0 should be 0 diversity."""
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 3.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 3.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
        })
        result = sc._opinion_diversity(scores)
        assert result == 0.0


class TestDistributionShape:
    def test_clustered(self):
        sc = QualityScorecard()
        scores = _make_concept_scores({
            f"p{i}": {"purchase_intent": 3.0 + (i * 0.1), "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0}
            for i in range(4)
        })
        shape, stdev = sc._distribution_shape(scores)
        assert shape == "clustered"

    def test_polarized(self):
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 1.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p2": {"purchase_intent": 1.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p3": {"purchase_intent": 5.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p4": {"purchase_intent": 5.0, "overall_appeal": 3.0, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
        })
        shape, stdev = sc._distribution_shape(scores)
        assert shape == "polarized"


class TestParticipationBalance:
    def test_balanced(self):
        sc = QualityScorecard()
        transcript = _make_transcript([
            {"speaker_id": "p1", "content": "msg1"},
            {"speaker_id": "p2", "content": "msg2"},
            {"speaker_id": "p1", "content": "msg3"},
            {"speaker_id": "p2", "content": "msg4"},
        ])
        result = sc._participation_balance(transcript)
        assert result >= 0.95, f"Expected near-perfect balance, got {result}"

    def test_imbalanced(self):
        sc = QualityScorecard()
        transcript = _make_transcript([
            {"speaker_id": "p1", "content": f"msg{i}"} for i in range(10)
        ] + [{"speaker_id": "p2", "content": "msg"}])
        result = sc._participation_balance(transcript)
        assert result < 0.7, f"Expected low balance, got {result}"


class TestMindChangeRate:
    def test_with_changes(self):
        sc = QualityScorecard()
        transcript = _make_transcript([
            {"speaker_id": "p1", "content": "msg1", "changed_mind": True},
            {"speaker_id": "p2", "content": "msg2", "changed_mind": False},
            {"speaker_id": "p3", "content": "msg3", "changed_mind": True},
            {"speaker_id": "p4", "content": "msg4", "changed_mind": False},
        ])
        result = sc._mind_change_rate(transcript)
        assert result == 0.5

    def test_no_changes(self):
        sc = QualityScorecard()
        transcript = _make_transcript([
            {"speaker_id": "p1", "content": "msg1"},
            {"speaker_id": "p2", "content": "msg2"},
        ])
        result = sc._mind_change_rate(transcript)
        assert result == 0.0


class TestOverallGrade:
    def test_grade_a(self):
        sc = QualityScorecard()
        grade, issues = sc._compute_grade(
            metric_independence=0.3,
            opinion_diversity=0.4,
            shape="moderate",
            alignment=0.6,
            discussion_quality=0.7,
        )
        assert grade == "A"

    def test_grade_d(self):
        sc = QualityScorecard()
        grade, issues = sc._compute_grade(
            metric_independence=0.05,
            opinion_diversity=0.05,
            shape="clustered",
            alignment=-0.3,
            discussion_quality=0.2,
        )
        assert grade == "D"
        assert len(issues) >= 3

    def test_issues_flagged(self):
        sc = QualityScorecard()
        grade, issues = sc._compute_grade(
            metric_independence=0.1,
            opinion_diversity=0.1,
            shape="clustered",
            alignment=0.6,
            discussion_quality=0.7,
        )
        assert any("metric independence" in issue.lower() for issue in issues)
        assert any("opinion diversity" in issue.lower() for issue in issues)


class TestPearson:
    def test_perfect_positive(self):
        result = QualityScorecard._pearson([(1, 1), (2, 2), (3, 3)])
        assert abs(result - 1.0) < 0.01

    def test_perfect_negative(self):
        result = QualityScorecard._pearson([(1, 3), (2, 2), (3, 1)])
        assert abs(result - (-1.0)) < 0.01

    def test_no_correlation(self):
        result = QualityScorecard._pearson([(1, 2), (2, 1), (3, 2)])
        assert abs(result) < 0.5

    def test_too_few_pairs(self):
        assert QualityScorecard._pearson([(1, 1)]) == 0.0


class TestFullScorecard:
    def test_end_to_end(self):
        """Full scorecard computation with realistic data."""
        sc = QualityScorecard()
        scores = _make_concept_scores({
            "p1": {"purchase_intent": 4.0, "overall_appeal": 4.5, "uniqueness": 3.5, "relevance": 4.0, "believability": 3.0, "value_perception": 3.5},
            "p2": {"purchase_intent": 1.5, "overall_appeal": 2.0, "uniqueness": 4.0, "relevance": 2.0, "believability": 3.5, "value_perception": 2.0},
            "p3": {"purchase_intent": 3.0, "overall_appeal": 3.5, "uniqueness": 3.0, "relevance": 3.0, "believability": 3.0, "value_perception": 3.0},
            "p4": {"purchase_intent": 2.0, "overall_appeal": 2.5, "uniqueness": 3.5, "relevance": 2.5, "believability": 2.5, "value_perception": 2.0},
        })
        report = AnalysisReport(
            executive_summary="Test",
            recommendation="ITERATE: test",
            confidence_level="medium",
            concept_scores=scores,
            themes=[Theme(name="Test Theme", description="desc", prevalence=0.5, sentiment=0.1, supporting_quotes=["q1"], participant_ids=["p1", "p2"], phase_distribution={"initial_reactions": 2})],
            sentiment_timeline=SentimentTimeline(by_phase={"initial_reactions": 0.1}, by_turn=[0.1], overall=0.1, trend="stable"),
            quotes=QuoteCollection(positive=[], negative=[], surprising=[], most_impactful=[]),
            segment_insights=[],
            top_concerns=["concern1"],
            top_opportunities=["opp1"],
            suggested_improvements=["imp1"],
            num_participants=4,
            num_messages=20,
            phases_completed=["initial_reactions"],
        )
        transcript = _make_transcript([
            {"speaker_id": "p1", "content": "I love it", "sentiment": 0.8},
            {"speaker_id": "p2", "content": "Not for me", "sentiment": -0.5},
            {"speaker_id": "p3", "content": "It's okay", "sentiment": 0.1},
            {"speaker_id": "p4", "content": "Meh", "sentiment": -0.2, "changed_mind": True},
        ])

        result = sc.score(report, transcript)
        assert result.overall_grade in ("A", "B", "C", "D")
        assert 0 <= result.metric_independence <= 1
        assert 0 <= result.opinion_diversity <= 1
        assert result.score_distribution_shape in ("clustered", "moderate", "polarized")
        assert 0 <= result.participation_balance <= 1
        assert 0 <= result.mind_change_rate <= 1
        assert isinstance(result.issues, list)
