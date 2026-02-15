from __future__ import annotations

import asyncio
from pathlib import Path

from analysis.analyzer import AnalysisEngine
from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionConfig
from discussion.simulator import DiscussionSimulator
from report.generator import ReportGenerator


def _build_pipeline_output():
    config = DiscussionConfig(
        product_concept="AI meal planning app",
        category="food_tech",
        num_personas=8,
    )
    simulator = DiscussionSimulator(config=config, llm_client=MockLLMClient())
    transcript = asyncio.run(simulator.run())

    engine = AnalysisEngine(llm_client=MockLLMClient())
    report = asyncio.run(engine.analyze(transcript))
    return transcript, report, transcript.personas


def test_generate_html_returns_valid_html() -> None:
    transcript, report, personas = _build_pipeline_output()
    generator = ReportGenerator()

    html = generator.generate_html(report=report, transcript=transcript, personas=personas)

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_output_contains_all_section_headers() -> None:
    transcript, report, personas = _build_pipeline_output()
    generator = ReportGenerator()
    html = generator.generate_html(report=report, transcript=transcript, personas=personas)

    expected_sections = [
        "Executive Summary",
        "Concept Scores",
        "Key Themes",
        "Sentiment Analysis",
        "Participant Profiles",
        "Segment Insights",
        "Concerns &amp; Opportunities",
        "Key Quotes",
        "Methodology",
    ]
    for section in expected_sections:
        assert section in html


def test_output_contains_svg_elements() -> None:
    transcript, report, personas = _build_pipeline_output()
    generator = ReportGenerator()
    html = generator.generate_html(report=report, transcript=transcript, personas=personas)

    assert "<svg" in html
    assert "</svg>" in html


def test_output_contains_participant_names_and_summary() -> None:
    transcript, report, personas = _build_pipeline_output()
    generator = ReportGenerator()
    html = generator.generate_html(report=report, transcript=transcript, personas=personas)

    for persona in personas[:3]:
        assert persona.name in html

    assert report.executive_summary in html


def test_save_html_creates_file_on_disk(tmp_path: Path) -> None:
    transcript, report, personas = _build_pipeline_output()
    generator = ReportGenerator()

    output_path = tmp_path / "focus_group_report.html"
    saved = generator.save_html(
        report=report,
        transcript=transcript,
        personas=personas,
        output_path=str(output_path),
    )

    assert Path(saved).exists()
    assert Path(saved).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
