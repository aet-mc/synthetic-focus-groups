#!/usr/bin/env python3
"""Debug: run a small focus group and print raw scores."""
import asyncio
import json
import sys

sys.stdout.reconfigure(line_buffering=True)

from discussion.models import DiscussionConfig
from discussion.llm_client import LLMClient
from discussion.simulator import DiscussionSimulator
from analysis.analyzer import AnalysisEngine


async def main():
    print("Starting 4-persona diagnostic...", flush=True)
    config = DiscussionConfig(
        product_concept="Coffee subscription: freshly roasted specialty beans from a different small-batch roaster each month. 16 dollars per month, free shipping, skip anytime.",
        category="beverages",
        num_personas=4,
        questions_per_phase=1,
    )
    client = LLMClient(provider="google", model="gemini-2.0-flash")
    sim = DiscussionSimulator(config=config, llm_client=client)

    print("Running discussion...", flush=True)
    transcript = await sim.run()
    print(f"Discussion done: {len(transcript.messages)} messages", flush=True)

    # Print last few participant messages
    participant_msgs = [m for m in transcript.messages if m.role.value == "participant"]
    print(f"\nSample participant messages ({len(participant_msgs)} total):", flush=True)
    for m in participant_msgs[-4:]:
        print(f"  {m.speaker_name}: {m.content[:120]}...", flush=True)

    print("\nRunning analysis...", flush=True)
    engine = AnalysisEngine(llm_client=client)
    result = await engine.analyze(transcript)

    print(f"\n=== RAW PARTICIPANT SCORES ===", flush=True)
    for pid, scores in result.concept_scores.participant_scores.items():
        name = next((p.name for p in transcript.personas if p.id == pid), pid[:8])
        print(f"  {name}: {json.dumps(scores)}", flush=True)

    print(f"\n=== AGGREGATE (top-2-box >= 3.5) ===", flush=True)
    print(f"  Purchase Intent: {result.concept_scores.purchase_intent:.0%}", flush=True)
    print(f"  Overall Appeal:  {result.concept_scores.overall_appeal:.0%}", flush=True)
    print(f"  Uniqueness:      {result.concept_scores.uniqueness:.0%}", flush=True)
    print(f"  Relevance:       {result.concept_scores.relevance:.0%}", flush=True)
    print(f"  Believability:   {result.concept_scores.believability:.0%}", flush=True)
    print(f"  Value:           {result.concept_scores.value_perception:.0%}", flush=True)
    print(f"  Excitement:      {result.concept_scores.excitement_score:.0%}", flush=True)


asyncio.run(main())
