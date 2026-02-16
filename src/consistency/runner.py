from __future__ import annotations

import math

from analysis.analyzer import AnalysisEngine
from discussion.models import DiscussionConfig
from discussion.simulator import DiscussionSimulator

from .models import ConsistencyReport, RunResult
from .scorecard import QualityScorecard


class ConsistencyRunner:
    """Runs the same concept multiple times with different seeds to measure stability."""

    def __init__(self, llm_client, num_runs: int = 3, seeds: list[int] | None = None):
        self.llm = llm_client
        self.num_runs = num_runs
        self.seeds = seeds or [42, 123, 777]

    async def run(self, config: DiscussionConfig) -> ConsistencyReport:
        scorecard_engine = QualityScorecard()
        runs: list[RunResult] = []

        for i in range(self.num_runs):
            seed = self.seeds[i] if i < len(self.seeds) else self.seeds[0] + i * 111
            run_config = config.model_copy(update={"seed": seed})

            sim = DiscussionSimulator(config=run_config, llm_client=self.llm)
            transcript = await sim.run()

            engine = AnalysisEngine(llm_client=self.llm)
            report = await engine.analyze(transcript)

            sc = scorecard_engine.score(report, transcript)

            runs.append(RunResult(
                seed=seed,
                concept_scores=report.concept_scores,
                recommendation=report.recommendation,
                confidence_level=report.confidence_level,
                num_messages=len(transcript.messages),
                themes=[t.name for t in report.themes],
                scorecard=sc,
            ))

        scorecards = [r.scorecard for r in runs]

        # Cross-run stability metrics
        score_cv = self._compute_score_cv(runs)
        theme_overlap = self._compute_theme_overlap(runs)
        rec_consistent = len({r.recommendation.split(":")[0].strip().upper() for r in runs}) == 1

        stability_grade = self._stability_grade(score_cv, theme_overlap, rec_consistent)

        # Combined grade: worst of avg quality + stability
        quality_grades = [sc.overall_grade for sc in scorecards]
        avg_quality = self._worst_grade(quality_grades)
        combined = self._worst_grade([avg_quality, stability_grade])

        summary = self._generate_summary(runs, score_cv, theme_overlap, rec_consistent, stability_grade, combined)

        return ConsistencyReport(
            runs=runs,
            scorecards=scorecards,
            score_cv=score_cv,
            theme_overlap=round(theme_overlap, 4),
            recommendation_consistent=rec_consistent,
            stability_grade=stability_grade,
            combined_grade=combined,
            summary=summary,
        )

    def _compute_score_cv(self, runs: list[RunResult]) -> dict[str, float]:
        """Coefficient of variation for each metric across runs."""
        metrics = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception", "excitement_score"]
        cv_map: dict[str, float] = {}
        for metric in metrics:
            values = [getattr(r.concept_scores, metric) for r in runs]
            mean = sum(values) / len(values) if values else 0
            if mean == 0:
                cv_map[metric] = 0.0
                continue
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stdev = math.sqrt(variance)
            cv_map[metric] = round(stdev / mean, 4) if mean != 0 else 0.0
        return cv_map

    def _compute_theme_overlap(self, runs: list[RunResult]) -> float:
        """Average pairwise Jaccard similarity of theme names across runs."""
        if len(runs) < 2:
            return 1.0
        similarities: list[float] = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                set_a = {t.lower() for t in runs[i].themes}
                set_b = {t.lower() for t in runs[j].themes}
                if not set_a and not set_b:
                    similarities.append(1.0)
                    continue
                union = set_a | set_b
                intersection = set_a & set_b
                similarities.append(len(intersection) / len(union) if union else 0.0)
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _stability_grade(self, score_cv: dict[str, float], theme_overlap: float, rec_consistent: bool) -> str:
        """Grade cross-run stability."""
        score = 0.0
        avg_cv = sum(score_cv.values()) / len(score_cv) if score_cv else 1.0

        # CV scoring (0-40 pts)
        if avg_cv < 0.15:
            score += 40
        elif avg_cv < 0.30:
            score += 25
        else:
            score += 10

        # Theme overlap (0-30 pts)
        if theme_overlap >= 0.5:
            score += 30
        elif theme_overlap >= 0.25:
            score += 20
        else:
            score += 5

        # Recommendation consistency (0-30 pts)
        if rec_consistent:
            score += 30
        else:
            score += 5

        if score >= 80:
            return "A"
        if score >= 60:
            return "B"
        if score >= 40:
            return "C"
        return "D"

    @staticmethod
    def _worst_grade(grades: list[str]) -> str:
        """Return the worst grade from a list."""
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        worst = max(order.get(g, 3) for g in grades)
        return {0: "A", 1: "B", 2: "C", 3: "D"}[worst]

    def _generate_summary(self, runs: list[RunResult], score_cv: dict[str, float], theme_overlap: float, rec_consistent: bool, stability_grade: str, combined_grade: str) -> str:
        avg_cv = sum(score_cv.values()) / len(score_cv) if score_cv else 0
        cv_label = "high" if avg_cv < 0.15 else "moderate" if avg_cv < 0.30 else "low"
        rec_text = "consistent" if rec_consistent else "inconsistent"
        recs = [r.recommendation.split(":")[0].strip() for r in runs]
        pis = [f"{r.concept_scores.purchase_intent:.0%}" for r in runs]

        return (
            f"Across {len(runs)} independent runs (seeds: {', '.join(str(r.seed) for r in runs)}), "
            f"score stability was {cv_label} (avg CV: {avg_cv:.0%}). "
            f"Purchase intent ranged from {min(pis)} to {max(pis)}. "
            f"Theme overlap was {theme_overlap:.0%} (Jaccard). "
            f"Recommendations were {rec_text} ({', '.join(recs)}). "
            f"Stability grade: {stability_grade}. Combined grade: {combined_grade}."
        )
