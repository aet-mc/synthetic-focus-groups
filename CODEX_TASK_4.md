# Codex Task 4: Build Report Generator

## Context

The persona engine (`src/persona_engine/`), discussion simulator (`src/discussion/`), and analysis engine (`src/analysis/`) are complete. The analysis engine produces an `AnalysisReport` object with executive summary, concept scores, themes, sentiment timeline, quotes, segment insights, concerns, opportunities, and recommendations.

Now build the **report generator** — transforms AnalysisReport into a professional, self-contained HTML report that clients can open in any browser and print to PDF.

## What to Build

### Project Structure (add to existing)

```
src/
├── persona_engine/          # ALREADY BUILT - don't modify
├── discussion/              # ALREADY BUILT - don't modify
├── analysis/                # ALREADY BUILT - don't modify
├── report/
│   ├── __init__.py
│   ├── generator.py         # ReportGenerator class
│   ├── charts.py            # Pure-Python inline SVG chart generation
│   ├── templates/
│   │   └── report.html      # Jinja2 HTML template (self-contained, all CSS inline)
│   └── models.py            # Report-specific models if needed
├── tests/
│   ├── ... (existing)
│   ├── test_report_generator.py
│   └── test_charts.py
```

## Detailed Specifications

### charts.py — Pure-Python SVG Chart Generation

Generate inline SVG strings directly in Python. NO external charting libraries. The SVG must be self-contained (embeddable in HTML).

```python
class ChartGenerator:
    """Generate inline SVG charts for the report. All methods return SVG strings."""
    
    @staticmethod
    def horizontal_bar_chart(
        labels: list[str],
        values: list[float],
        max_value: float = 1.0,
        title: str = "",
        width: int = 600,
        height: int | None = None,  # Auto-calculate from num bars
        colors: list[str] | None = None,
        show_values: bool = True,
        value_format: str = "{:.0%}",  # Format string for displayed values
    ) -> str:
        """Horizontal bar chart. Used for concept scores (Top-2-Box %), theme prevalence.
        
        Each bar: label on left, colored bar, value on right.
        Colors default to a professional blue gradient.
        Bar width proportional to value/max_value.
        """
    
    @staticmethod
    def sentiment_line_chart(
        phases: list[str],
        values: list[float],
        title: str = "Sentiment by Phase",
        width: int = 600,
        height: int = 250,
    ) -> str:
        """Line chart showing sentiment progression across phases.
        
        X-axis: phase names
        Y-axis: -1 to +1 sentiment
        Zero line highlighted
        Color: green above 0, red below 0
        """
    
    @staticmethod
    def donut_chart(
        labels: list[str],
        values: list[float],
        title: str = "",
        width: int = 250,
        height: int = 250,
        colors: list[str] | None = None,
    ) -> str:
        """Donut/ring chart. Used for overall recommendation (GO/ITERATE/NO-GO visual).
        Also used for opinion distribution pie.
        """
    
    @staticmethod
    def score_gauge(
        value: float,
        max_value: float = 1.0,
        label: str = "",
        width: int = 150,
        height: int = 100,
    ) -> str:
        """Semi-circular gauge chart for a single metric.
        Green zone: >0.65, Yellow: 0.45-0.65, Red: <0.45.
        Used for excitement score display.
        """
    
    @staticmethod
    def participant_grid(
        personas: list,  # Persona objects
        width: int = 700,
    ) -> str:
        """Visual grid showing participant avatars/icons with key demographics.
        Each participant: circle with initials, age, occupation below.
        Color-coded by opinion valence (green=positive, red=negative, gray=neutral).
        """
```

**SVG styling rules:**
- Use a professional color palette: `#2563EB` (primary blue), `#10B981` (green/positive), `#EF4444` (red/negative), `#F59E0B` (yellow/neutral), `#6B7280` (gray)
- Font: system-ui, -apple-system, sans-serif (available everywhere)
- All text rendered as SVG `<text>` elements (no font embedding needed)
- Round corners on bars (rx="4")
- Clean, minimal design — think McKinsey/Bain report quality

### templates/report.html — Jinja2 Template

A **single self-contained HTML file** with ALL CSS inline (no external dependencies). Must look professional when opened in any browser.

#### Report Structure (sections in order):

```
1. COVER PAGE
   - Study title: "Focus Group Report: [Product Concept]"
   - Date
   - "Powered by SynthFocus" branding
   - Study type: "Synthetic Focus Group — AI-Powered Market Research"

2. EXECUTIVE SUMMARY (1 page)
   - 3-5 sentence summary
   - Recommendation badge: GO (green) / ITERATE (yellow) / NO-GO (red)
   - Excitement Score gauge chart
   - 3 key stats in big numbers (e.g., "62% Purchase Intent", "5 Key Themes", "8 Participants")

3. CONCEPT SCORES (1 page)
   - Horizontal bar chart of all 6 metrics (Top-2-Box %)
   - Each metric with a brief interpretation sentence
   - Color-coded: green (>65%), yellow (45-65%), red (<45%)

4. KEY THEMES (1-2 pages)
   - Each theme as a card with:
     - Theme name (bold header)
     - Prevalence bar (e.g., "6 of 8 participants")
     - Sentiment indicator (positive/negative/mixed)
     - Description
     - 2-3 supporting verbatim quotes in italic blockquotes with speaker attribution

5. SENTIMENT ANALYSIS (1 page)
   - Sentiment line chart (by phase)
   - Trend description
   - Phase-by-phase narrative

6. PARTICIPANT PROFILES (1 page)
   - Participant grid visualization
   - Table: Name, Age, Gender, Income Range, Education, VALS Type, Initial vs Final Opinion
   - Note: "Personas generated from US Census demographic distributions"

7. SEGMENT INSIGHTS (1 page)
   - Each meaningful segment as a card:
     - Segment name
     - Purchase intent for this segment
     - How it differs from overall
     - Representative quote

8. CONCERNS & OPPORTUNITIES (1 page)
   - Two-column layout
   - Left: Top concerns (red bullets)
   - Right: Top opportunities (green bullets)
   - Below: Suggested improvements (blue bullets)

9. KEY QUOTES (1 page)
   - Most impactful quotes in large pull-quote format
   - Positive, negative, and surprising quotes
   - Speaker attribution with demographics

10. METHODOLOGY (half page)
    - "This study used AI-generated personas grounded in US Census demographic data"
    - "Personality profiles based on Five-Factor (OCEAN) model with age/gender-adjusted distributions"
    - "Discussion facilitated by AI moderator using 5-phase qualitative research methodology"
    - "Analysis includes thematic coding (Braun & Clarke framework), concept scoring, and segment analysis"
    - Participant count, phase count, total messages
    - Disclaimer: "Synthetic research is designed to complement, not replace, traditional research methods"

11. APPENDIX: FULL TRANSCRIPT (optional, collapsible)
    - Toggle-able section (JavaScript collapse/expand)
    - Full discussion formatted with speaker names and phase headers
```

#### CSS Requirements:
- Print-friendly: `@media print` styles, page breaks between sections
- A4/Letter page size awareness
- Professional typography: clean headers, readable body text
- Color palette consistent with charts
- Quote styling: left border, italic, gray background
- Responsive: looks good on screen AND prints well
- Dark header/footer with branding
- Card-based layout for themes and segments

### generator.py — Report Generator

```python
class ReportGenerator:
    def __init__(self):
        self.chart_gen = ChartGenerator()
        self.env = Environment(loader=FileSystemLoader(templates_dir))
    
    def generate_html(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,  # Persona objects
        include_transcript: bool = True,
    ) -> str:
        """Generate complete self-contained HTML report.
        
        1. Pre-generate all SVG charts
        2. Prepare template context
        3. Render Jinja2 template
        4. Return HTML string
        """
    
    def save_html(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,
        output_path: str,
        include_transcript: bool = True,
    ) -> str:
        """Generate and save HTML to file. Returns file path."""
    
    def _prepare_context(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,
    ) -> dict:
        """Prepare all template variables including pre-rendered SVG charts"""
```

### Testing

**test_report_generator.py:**
- Test generate_html returns valid HTML string
- Test output contains all section headers (Executive Summary, Concept Scores, etc.)
- Test output contains SVG elements (charts rendered)
- Test output contains participant names from the personas
- Test output contains the executive summary text from the report
- Test save_html creates a file on disk

**test_charts.py:**
- Test horizontal_bar_chart returns valid SVG string with `<svg` and `</svg>` tags
- Test bar chart contains all labels
- Test sentiment_line_chart has correct number of data points
- Test donut_chart renders without error
- Test score_gauge color is green for value > 0.65, yellow for 0.45-0.65, red for < 0.45

## End-to-End Demo

Also create `src/demo.py` — a script that runs the ENTIRE pipeline end-to-end:

```python
"""
Demo script: Run a complete synthetic focus group study.

Usage: python3 -m src.demo "Your product concept here" --category food_tech --output report.html
"""

import asyncio
import argparse

async def main():
    # 1. Generate personas
    # 2. Run discussion simulation (with MockLLMClient for demo)
    # 3. Analyze transcript
    # 4. Generate HTML report
    # 5. Save to output file
    # Print summary to stdout
```

## Dependencies

Add to requirements.txt:
- jinja2>=3.1 (already installed on system)

No weasyprint needed — HTML-only output.

## Important

- The HTML must be 100% self-contained: all CSS inline, all SVGs inline, no external requests
- Professional quality — this is what clients see. It needs to look like a $5,000 McKinsey deliverable
- Charts must be clean and readable when printed
- Use the existing MockLLMClient for all tests — no real API calls
- Import from existing modules, don't modify them
- The demo.py should work with `python3 -m src.demo "AI meal planning app" --category food_tech`
