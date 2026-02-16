from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyzer import AnalysisEngine
from consistency.runner import ConsistencyRunner
from consistency.scorecard import QualityScorecard
from discussion.llm_client import LLMClient, MockLLMClient
from discussion.models import DiscussionConfig
from discussion.simulator import DiscussionSimulator
from report.generator import ReportGenerator


def print_scorecard_summary(scorecard, title: str = "Quality Scorecard") -> None:
    """Print a scorecard summary to stdout."""
    print(f"\n  {title}")
    print(f"  {'─' * 40}")
    print(f"  Overall Grade: {scorecard.overall_grade}")
    print(f"  Metric Independence: {scorecard.metric_independence:.2f}")
    print(f"  Opinion Diversity: {scorecard.opinion_diversity:.2f}")
    print(f"  Score Distribution: {scorecard.score_distribution_shape} (σ={scorecard.score_distribution_stdev:.2f})")
    print(f"  Sentiment Alignment: {scorecard.sentiment_score_alignment:.2f}")
    print(f"  Discussion Quality: {scorecard.discussion_quality:.2f}")
    print(f"  Participation Balance: {scorecard.participation_balance:.2f}")
    print(f"  Mind Change Rate: {scorecard.mind_change_rate:.2f}")
    if scorecard.issues:
        print(f"  Issues:")
        for issue in scorecard.issues:
            print(f"    ⚠ {issue}")


def print_consistency_summary(consistency_report) -> None:
    """Print consistency report summary to stdout."""
    print(f"\n  Consistency Report ({len(consistency_report.runs)} runs)")
    print(f"  {'─' * 40}")
    print(f"  Combined Grade: {consistency_report.combined_grade}")
    print(f"  Stability Grade: {consistency_report.stability_grade}")
    print(f"  Recommendation Consistent: {'Yes' if consistency_report.recommendation_consistent else 'No'}")
    print(f"  Theme Overlap: {consistency_report.theme_overlap:.0%}")

    if consistency_report.score_cv:
        print(f"  Score CV:")
        for metric, cv in consistency_report.score_cv.items():
            stability = "✓" if cv < 0.15 else "~" if cv < 0.30 else "✗"
            print(f"    {stability} {metric}: {cv:.1%}")

    print(f"\n  {consistency_report.summary}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a complete synthetic focus group study and generate an HTML report."
    )
    parser.add_argument("product_concept", help="Product concept to evaluate")
    parser.add_argument("--category", default="general", help="Product category")
    parser.add_argument("--output", default="report.html", help="Output HTML file")
    parser.add_argument("--participants", type=int, default=8, help="Number of personas")
    parser.add_argument("--personas", type=int, default=None, help="Number of personas (alias for --participants)")
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Exclude full transcript appendix",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no API calls, deterministic output)",
    )
    parser.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "deepseek", "nvidia", "openrouter", "google", "moonshotai"],
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model ID for the provider",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for persona generation (for reproducibility)",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Run 3 seeds instead of 1 and generate consistency report",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for consistency check (default: 3)",
    )
    args = parser.parse_args()

    # --personas overrides --participants
    if args.personas is not None:
        args.participants = args.personas

    # Build LLM client
    if args.mock:
        llm = MockLLMClient()
        print("Using mock LLM (no API calls)")
    else:
        llm = LLMClient(provider=args.provider, model=args.model)
        print(f"Using {args.provider} / {llm.model}")

    config = DiscussionConfig(
        product_concept=args.product_concept,
        category=args.category,
        num_personas=args.participants,
        seed=args.seed if args.seed is not None else 42,
    )

    print(f"\n{'='*60}")
    print(f"  Synthetic Focus Group: {args.product_concept}")
    print(f"  Category: {args.category} | Participants: {args.participants}")
    if args.consistency:
        print(f"  Mode: Consistency Check ({args.num_runs} runs)")
    print(f"{'='*60}\n")

    if args.consistency:
        # Run consistency check
        print(f"Running consistency check with {args.num_runs} seeds...")
        runner = ConsistencyRunner(llm_client=llm, num_runs=args.num_runs)
        consistency_report = await runner.run_consistency_check(config)

        print_consistency_summary(consistency_report)

        # Generate report from first run (or best run)
        # For now, use first run's data
        if consistency_report.runs:
            first_run = consistency_report.runs[0]
            first_scorecard = consistency_report.scorecards[0] if consistency_report.scorecards else None
            print(f"\n  Using run with seed {first_run.seed} for report generation")

            # Re-run first seed to get full data for report
            run_config = config.model_copy(update={"seed": first_run.seed})
            simulator = DiscussionSimulator(config=run_config, llm_client=llm)
            transcript = await simulator.run()
            analyzer = AnalysisEngine(llm_client=llm)
            report = await analyzer.analyze(transcript)

            print("Generating report...")
            report_generator = ReportGenerator()
            output_path = report_generator.save_html(
                report=report,
                transcript=transcript,
                personas=transcript.personas,
                output_path=args.output,
                include_transcript=not args.no_transcript,
                scorecard=first_scorecard,
            )

            print(f"\n{'='*60}")
            print("  ✅ COMPLETE")
            print(f"  Combined Grade: {consistency_report.combined_grade}")
            print(f"  Recommendation: {first_run.recommendation}")
            print(f"  Report: {Path(output_path).resolve()}")
            print(f"{'='*60}")
    else:
        # Single run mode
        # 1. Simulate discussion
        print("Phase 1/3: Running focus group discussion...")
        simulator = DiscussionSimulator(config=config, llm_client=llm)
        transcript = await simulator.run()
        print(f"  ✓ {len(transcript.messages)} messages across {len(set(m.phase for m in transcript.messages))} phases")

        # 2. Analyze
        print("Phase 2/3: Analyzing transcript...")
        analyzer = AnalysisEngine(llm_client=llm)
        report = await analyzer.analyze(transcript)
        print(f"  ✓ {len(report.themes)} themes, excitement score: {report.concept_scores.excitement_score:.0%}")

        # 2b. Compute quality scorecard
        scorecard = QualityScorecard().score(report, transcript)
        print_scorecard_summary(scorecard)

        # 3. Generate report
        print("\nPhase 3/3: Generating report...")
        report_generator = ReportGenerator()
        output_path = report_generator.save_html(
            report=report,
            transcript=transcript,
            personas=transcript.personas,
            output_path=args.output,
            include_transcript=not args.no_transcript,
            scorecard=scorecard,
        )

        print(f"\n{'='*60}")
        print("  ✅ COMPLETE")
        print(f"  Quality Grade: {scorecard.overall_grade}")
        print(f"  Recommendation: {report.recommendation}")
        print(f"  Purchase Intent: {report.concept_scores.purchase_intent:.0%}")
        print(f"  Excitement Score: {report.concept_scores.excitement_score:.0%}")
        print(f"  Report: {Path(output_path).resolve()}")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
