from __future__ import annotations

import math

from analysis.models import AnalysisReport, ConceptScores
from discussion.models import DiscussionTranscript, MessageRole

from .models import ScorecardResult


class QualityScorecard:
    """Computes quality metrics for a single focus group run."""

    METRICS = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception"]

    def score(self, report: AnalysisReport, transcript: DiscussionTranscript) -> ScorecardResult:
        metric_independence = self._metric_independence(report.concept_scores)
        opinion_diversity = self._opinion_diversity(report.concept_scores)
        shape, stdev = self._distribution_shape(report.concept_scores)
        alignment = self._sentiment_score_alignment(report.concept_scores, transcript)
        participation_balance = self._participation_balance(transcript)
        mind_change_rate = self._mind_change_rate(transcript)
        discussion_quality = (participation_balance * 0.4 + mind_change_rate * 0.3 + opinion_diversity * 0.3)

        grade, issues = self._compute_grade(
            metric_independence=metric_independence,
            opinion_diversity=opinion_diversity,
            shape=shape,
            alignment=alignment,
            discussion_quality=discussion_quality,
        )

        return ScorecardResult(
            metric_independence=round(metric_independence, 4),
            opinion_diversity=round(opinion_diversity, 4),
            score_distribution_shape=shape,
            score_distribution_stdev=round(stdev, 4),
            sentiment_score_alignment=round(alignment, 4),
            discussion_quality=round(discussion_quality, 4),
            participation_balance=round(participation_balance, 4),
            mind_change_rate=round(mind_change_rate, 4),
            overall_grade=grade,
            issues=issues,
        )

    def _metric_independence(self, concept_scores: ConceptScores) -> float:
        """Average per-person spread across metrics, normalized to 0-1. Higher = more independent."""
        spreads: list[float] = []
        for scores in concept_scores.participant_scores.values():
            vals = [scores.get(m, 3.0) for m in self.METRICS]
            if vals:
                spreads.append(max(vals) - min(vals))
        if not spreads:
            return 0.0
        avg_spread = sum(spreads) / len(spreads)
        return min(avg_spread / 4.0, 1.0)  # normalize: 4.0 spread = perfect independence

    def _opinion_diversity(self, concept_scores: ConceptScores) -> float:
        """Diversity of purchase_intent scores across participants. 0 = everyone agrees, 1 = max spread."""
        pi_values = [scores.get("purchase_intent", 3.0) for scores in concept_scores.participant_scores.values()]
        if len(pi_values) < 2:
            return 0.0
        mean = sum(pi_values) / len(pi_values)
        variance = sum((v - mean) ** 2 for v in pi_values) / len(pi_values)
        stdev = math.sqrt(variance)
        # Max theoretical stdev on 1-5 scale is 2.0 (half at 1, half at 5)
        return min(stdev / 2.0, 1.0)

    def _distribution_shape(self, concept_scores: ConceptScores) -> tuple[str, float]:
        """Classify the PI score distribution shape."""
        pi_values = [scores.get("purchase_intent", 3.0) for scores in concept_scores.participant_scores.values()]
        if len(pi_values) < 2:
            return "clustered", 0.0
        mean = sum(pi_values) / len(pi_values)
        variance = sum((v - mean) ** 2 for v in pi_values) / len(pi_values)
        stdev = math.sqrt(variance)
        if stdev < 0.5:
            shape = "clustered"
        elif stdev > 1.2:
            shape = "polarized"
        else:
            shape = "moderate"
        return shape, stdev

    def _sentiment_score_alignment(self, concept_scores: ConceptScores, transcript: DiscussionTranscript) -> float:
        """Pearson correlation between avg transcript sentiment and purchase_intent per participant."""
        # Collect avg sentiment per participant from transcript messages
        sentiment_by_participant: dict[str, list[float]] = {}
        for message in transcript.messages:
            if message.role != MessageRole.PARTICIPANT:
                continue
            if message.sentiment is not None:
                sentiment_by_participant.setdefault(message.speaker_id, []).append(float(message.sentiment))

        if not sentiment_by_participant:
            return 0.0

        pairs: list[tuple[float, float]] = []
        for pid, sentiments in sentiment_by_participant.items():
            if pid in concept_scores.participant_scores and "purchase_intent" in concept_scores.participant_scores[pid]:
                avg_sentiment = sum(sentiments) / len(sentiments)
                pi = concept_scores.participant_scores[pid]["purchase_intent"]
                pairs.append((avg_sentiment, pi))

        if len(pairs) < 3:
            return 0.0

        return self._pearson(pairs)

    def _participation_balance(self, transcript: DiscussionTranscript) -> float:
        """1 - Gini coefficient of message counts per participant. 1 = perfectly balanced."""
        counts: dict[str, int] = {}
        for message in transcript.messages:
            if message.role == MessageRole.PARTICIPANT:
                counts[message.speaker_id] = counts.get(message.speaker_id, 0) + 1

        if len(counts) < 2:
            return 1.0

        values = sorted(counts.values())
        n = len(values)
        total = sum(values)
        if total == 0:
            return 1.0

        # Gini coefficient
        cumulative = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(values):
            cumulative += v
            weighted_sum += (i + 1) * v
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, 1.0 - gini))

    def _mind_change_rate(self, transcript: DiscussionTranscript) -> float:
        """Fraction of participants who changed their mind during the discussion."""
        participants: set[str] = set()
        changers: set[str] = set()
        for message in transcript.messages:
            if message.role != MessageRole.PARTICIPANT:
                continue
            participants.add(message.speaker_id)
            if message.changed_mind:
                changers.add(message.speaker_id)

        if not participants:
            return 0.0
        return len(changers) / len(participants)

    def _compute_grade(
        self,
        metric_independence: float,
        opinion_diversity: float,
        shape: str,
        alignment: float,
        discussion_quality: float,
    ) -> tuple[str, list[str]]:
        """Compute overall grade A/B/C/D with issue list."""
        issues: list[str] = []
        score = 0.0

        # Metric independence (0-25 pts)
        if metric_independence >= 0.25:
            score += 25
        elif metric_independence >= 0.15:
            score += 15
        else:
            score += 5
            issues.append(f"Low metric independence ({metric_independence:.2f}) — scores may be correlated")

        # Opinion diversity (0-25 pts)
        if opinion_diversity >= 0.3:
            score += 25
        elif opinion_diversity >= 0.15:
            score += 15
        else:
            score += 5
            issues.append(f"Low opinion diversity ({opinion_diversity:.2f}) — participants may be too similar")

        # Sentiment-score alignment (0-25 pts)
        if alignment >= 0.5:
            score += 25
        elif alignment >= 0.2:
            score += 15
        elif alignment >= 0.0:
            score += 10
        else:
            score += 5
            issues.append(f"Negative sentiment-score alignment ({alignment:.2f}) — scores contradict discussion tone")

        # Discussion quality (0-25 pts)
        if discussion_quality >= 0.6:
            score += 25
        elif discussion_quality >= 0.4:
            score += 15
        else:
            score += 5
            issues.append(f"Low discussion quality ({discussion_quality:.2f}) — participation may be unbalanced")

        # Shape warnings (don't affect score, just flagged)
        if shape == "clustered":
            issues.append("Score distribution is clustered — limited differentiation between participants")

        if score >= 80:
            return "A", issues
        if score >= 60:
            return "B", issues
        if score >= 40:
            return "C", issues
        return "D", issues

    @staticmethod
    def _pearson(pairs: list[tuple[float, float]]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(pairs)
        if n < 2:
            return 0.0
        xs, ys = zip(*pairs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)
