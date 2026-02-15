from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyzer import AnalysisEngine
from discussion.llm_client import LLMClient, MockLLMClient
from discussion.models import DiscussionConfig
from discussion.simulator import DiscussionSimulator
from report.generator import ReportGenerator


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a complete synthetic focus group study and generate an HTML report."
    )
    parser.add_argument("product_concept", help="Product concept to evaluate")
    parser.add_argument("--category", default="general", help="Product category")
    parser.add_argument("--output", default="report.html", help="Output HTML file")
    parser.add_argument("--participants", type=int, default=8, help="Number of personas")
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
    args = parser.parse_args()

    # Build LLM client
    if args.mock:
        llm = MockLLMClient()
        print(f"Using mock LLM (no API calls)")
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
    print(f"{'='*60}\n")

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

    # 3. Generate report
    print("Phase 3/3: Generating report...")
    report_generator = ReportGenerator()
    output_path = report_generator.save_html(
        report=report,
        transcript=transcript,
        personas=transcript.personas,
        output_path=args.output,
        include_transcript=not args.no_transcript,
    )

    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETE")
    print(f"  Recommendation: {report.recommendation}")
    print(f"  Purchase Intent: {report.concept_scores.purchase_intent:.0%}")
    print(f"  Excitement Score: {report.concept_scores.excitement_score:.0%}")
    print(f"  Report: {Path(output_path).resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
