from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyzer import AnalysisEngine
from discussion.llm_client import MockLLMClient
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
    args = parser.parse_args()

    config = DiscussionConfig(
        product_concept=args.product_concept,
        category=args.category,
        num_personas=args.participants,
    )

    simulator = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    transcript = await simulator.run()

    analyzer = AnalysisEngine(llm_client=MockLLMClient())
    report = await analyzer.analyze(transcript)

    report_generator = ReportGenerator()
    output_path = report_generator.save_html(
        report=report,
        transcript=transcript,
        personas=transcript.personas,
        output_path=args.output,
        include_transcript=not args.no_transcript,
    )

    print("Synthetic focus group completed")
    print(f"Product concept: {args.product_concept}")
    print(f"Category: {args.category}")
    print(f"Participants: {len(transcript.personas)}")
    print(f"Messages: {len(transcript.messages)}")
    print(f"Recommendation: {report.recommendation}")
    print(f"Report saved: {Path(output_path).resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
