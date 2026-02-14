# Codex Task 3: Build Analysis Engine

## Context

The persona engine (`src/persona_engine/`) and discussion simulator (`src/discussion/`) are complete. After a focus group simulation runs, we have a `DiscussionTranscript` with ~50-65 messages across 5 phases.

Now build the **analysis engine** — the module that transforms raw transcripts into actionable market research insights. This is what clients pay for.

## What to Build

### Project Structure (add to existing)

```
src/
├── persona_engine/          # ALREADY BUILT - don't modify
├── discussion/              # ALREADY BUILT - don't modify
├── analysis/
│   ├── __init__.py
│   ├── analyzer.py          # Main AnalysisEngine orchestrator
│   ├── theme_extractor.py   # Braun & Clarke thematic analysis via LLM
│   ├── sentiment.py         # Per-message and aggregate sentiment scoring
│   ├── concept_scorer.py    # Standard concept testing metrics (purchase intent, appeal, etc.)
│   ├── quote_extractor.py   # Pull impactful verbatim quotes from transcript
│   ├── segment_analyzer.py  # Break insights by demographic/psychographic segments
│   ├── prompts.py           # Analysis-specific LLM prompt templates
│   └── models.py            # Analysis data models
├── tests/
│   ├── ... (existing)
│   ├── test_analyzer.py
│   ├── test_theme_extractor.py
│   ├── test_concept_scorer.py
│   └── test_segment_analyzer.py
```

## Detailed Specifications

### models.py — Analysis Data Models

```python
from pydantic import BaseModel, Field

class Theme(BaseModel):
    name: str  # Short theme name, e.g. "Price Sensitivity"
    description: str  # 1-2 sentence description
    prevalence: float  # 0-1, what fraction of participants mentioned this
    sentiment: float  # -1 to 1, overall sentiment of this theme
    supporting_quotes: list[str]  # 3-5 verbatim quotes from transcript
    participant_ids: list[str]  # Which personas contributed to this theme
    phase_distribution: dict[str, int]  # How many mentions per phase

class ConceptScores(BaseModel):
    """Standard concept testing metrics, each on 1-5 scale then converted to Top-2-Box %"""
    purchase_intent: float  # Top-2-Box % (definitely + probably would buy)
    overall_appeal: float  # Top-2-Box %
    uniqueness: float  # Top-2-Box %
    relevance: float  # Top-2-Box % (solves a real need)
    believability: float  # Top-2-Box %
    value_perception: float  # Top-2-Box % (worth the price)
    
    # Derived scores
    excitement_score: float  # Composite: (appeal × 0.3 + uniqueness × 0.25 + purchase_intent × 0.25 + relevance × 0.2)
    
    # Per-participant raw scores (for segment analysis)
    participant_scores: dict[str, dict[str, float]]  # participant_id -> {metric: 1-5 score}

class SentimentTimeline(BaseModel):
    """Sentiment progression through the discussion"""
    by_phase: dict[str, float]  # phase -> average sentiment
    by_turn: list[float]  # sentiment per message
    overall: float  # Overall average
    trend: str  # "improving", "declining", "stable", "volatile"

class QuoteCollection(BaseModel):
    """Curated impactful quotes organized by sentiment and theme"""
    positive: list[dict]  # [{quote, speaker_name, context}]
    negative: list[dict]
    surprising: list[dict]  # Unexpected or insightful quotes
    most_impactful: list[dict]  # Top 5 most quotable moments

class SegmentInsight(BaseModel):
    """Insights broken down by a demographic or psychographic segment"""
    segment_name: str  # e.g. "High Income ($80K+)", "Low Agreeableness (Contrarians)"
    segment_type: str  # "demographic" or "psychographic"
    participant_ids: list[str]
    avg_sentiment: float
    purchase_intent: float  # Top-2-Box for this segment
    key_themes: list[str]  # Theme names most relevant to this segment
    distinguishing_quote: str  # Best quote that captures this segment's view
    differs_from_overall: str | None  # How this segment differs from the group (or None)

class AnalysisReport(BaseModel):
    """Complete analysis output"""
    # Executive Summary
    executive_summary: str  # 3-5 sentence high-level takeaway
    recommendation: str  # Clear go/no-go/iterate recommendation
    confidence_level: str  # "high", "medium", "low"
    
    # Core Metrics
    concept_scores: ConceptScores
    
    # Themes
    themes: list[Theme]  # Ordered by prevalence (most common first)
    
    # Sentiment
    sentiment_timeline: SentimentTimeline
    
    # Quotes
    quotes: QuoteCollection
    
    # Segments
    segment_insights: list[SegmentInsight]
    
    # Concerns & Opportunities
    top_concerns: list[str]  # Top 3-5 barriers/concerns raised
    top_opportunities: list[str]  # Top 3-5 opportunities/positives identified
    suggested_improvements: list[str]  # Specific product improvement suggestions from participants
    
    # Metadata
    num_participants: int
    num_messages: int
    phases_completed: list[str]
```

### theme_extractor.py — Braun & Clarke Thematic Analysis

Implements a simplified version of Braun & Clarke's 6-phase framework using LLM:

```python
class ThemeExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def extract_themes(
        self,
        transcript: DiscussionTranscript,
        max_themes: int = 7,
    ) -> list[Theme]:
        """
        Phase 1: Familiarization — feed full transcript to LLM
        Phase 2: Initial coding — ask LLM to code each message with tags
        Phase 3: Search for themes — cluster codes into candidate themes
        Phase 4: Review themes — refine, merge overlapping themes
        Phase 5: Define and name — create final theme definitions
        Phase 6: (Report writing is handled elsewhere)
        
        In practice, we do this in 2 LLM calls:
        1. Send transcript → get initial codes for each message (list of tags)
        2. Send codes → get clustered themes with descriptions
        
        Then compute prevalence, sentiment, supporting quotes programmatically.
        """
```

**Implementation approach:**
- Call 1: Send full transcript text. Ask LLM to assign 1-3 topic codes per participant message. Return as JSON: `[{message_index, codes: [str]}]`
- Call 2: Send the list of all codes. Ask LLM to cluster into max_themes themes, name each, and write a 1-2 sentence description. Return as JSON.
- Programmatic: For each theme, find all messages tagged with its constituent codes. Count participants (prevalence), compute average sentiment, extract top quotes.

### sentiment.py — Sentiment Analysis

```python
class SentimentAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def analyze_message(self, message: DiscussionMessage) -> float:
        """Score a single message from -1 (very negative) to 1 (very positive)"""
    
    async def analyze_batch(self, messages: list[DiscussionMessage]) -> list[float]:
        """Score all messages in a single LLM call for efficiency.
        Send all participant messages in one prompt, get back JSON array of scores."""
    
    def compute_timeline(self, messages: list[DiscussionMessage], scores: list[float]) -> SentimentTimeline:
        """Compute phase-by-phase and turn-by-turn sentiment, detect trend"""
```

**For the MockLLMClient path (testing):** Derive sentiment from the persona's pre-seeded opinion_valence ± small random noise. This makes test results deterministic and meaningful.

### concept_scorer.py — Standard Concept Testing Metrics

```python
class ConceptScorer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def score_concept(
        self,
        transcript: DiscussionTranscript,
        personas: list,  # Persona objects
    ) -> ConceptScores:
        """
        For each participant, analyze their statements across the discussion to infer:
        - Purchase intent (1-5 scale: definitely not → definitely would)
        - Overall appeal (1-5)
        - Uniqueness (1-5: nothing new → very unique)
        - Relevance (1-5: no need → solves real problem)
        - Believability (1-5: skeptical → fully believe)
        - Value perception (1-5: overpriced → great value)
        
        Then convert to Top-2-Box: % scoring 4 or 5.
        Compute excitement_score composite.
        """
```

**For MockLLMClient:** Derive scores from persona's opinion_valence mapped to 1-5 scale:
- valence -1.0 to -0.5 → score 1-2
- valence -0.5 to 0.0 → score 2-3
- valence 0.0 to 0.5 → score 3-4
- valence 0.5 to 1.0 → score 4-5
Add trait-specific adjustments (e.g., high openness → +0.5 uniqueness score).

### quote_extractor.py — Verbatim Quote Extraction

```python
class QuoteExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def extract_quotes(
        self,
        transcript: DiscussionTranscript,
        themes: list[Theme],
    ) -> QuoteCollection:
        """
        From the transcript, pull out the most impactful verbatim quotes.
        Categorize as positive, negative, surprising.
        Select top 5 most quotable moments (would look great in a report/presentation).
        """
```

**For MockLLMClient:** Select quotes by length (>20 words) and assign to categories based on the persona's opinion_valence.

### segment_analyzer.py — Segment Breakdowns

```python
class SegmentAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def analyze_segments(
        self,
        transcript: DiscussionTranscript,
        personas: list,
        concept_scores: ConceptScores,
        themes: list[Theme],
    ) -> list[SegmentInsight]:
        """
        Break analysis by meaningful segments:
        
        Demographic segments (auto-detect which are meaningful):
        - Age groups (under 35 / 35-55 / over 55)
        - Income levels (under $50K / $50-100K / over $100K)
        - Gender
        
        Psychographic segments:
        - High vs Low Openness (early adopters vs skeptics)
        - High vs Low Agreeableness (conformists vs contrarians)
        - VALS type groupings
        
        For each segment:
        - Calculate segment-specific concept scores
        - Identify themes that over-index in this segment
        - Find the most representative quote
        - Note how this segment differs from the overall
        
        Only include segments with meaningful differences (>10% deviation from overall).
        """
```

### analyzer.py — Main Orchestrator

```python
class AnalysisEngine:
    def __init__(self, llm_client=None):
        """If llm_client is None, use MockLLMClient"""
    
    async def analyze(
        self,
        transcript: DiscussionTranscript,
    ) -> AnalysisReport:
        """
        Full analysis pipeline:
        1. Run sentiment analysis on all messages
        2. Extract themes (Braun & Clarke via LLM)
        3. Score concept metrics (purchase intent, appeal, etc.)
        4. Extract impactful quotes
        5. Analyze by segments
        6. Generate executive summary
        7. Generate recommendation
        8. Compile into AnalysisReport
        """
    
    async def _generate_executive_summary(
        self,
        concept_scores: ConceptScores,
        themes: list[Theme],
        sentiment: SentimentTimeline,
        config: DiscussionConfig,
    ) -> str:
        """3-5 sentence summary of the key findings"""
    
    async def _generate_recommendation(
        self,
        concept_scores: ConceptScores,
        themes: list[Theme],
        concerns: list[str],
    ) -> tuple[str, str]:
        """Returns (recommendation_text, confidence_level)
        
        Recommendation logic:
        - excitement_score > 0.65 → "GO: Strong concept, proceed to development"
        - excitement_score 0.45-0.65 → "ITERATE: Promising but needs refinement"
        - excitement_score < 0.45 → "NO-GO: Concept needs fundamental rethinking"
        
        confidence_level based on:
        - 8 participants, all phases → "medium" (synthetic study)
        - Clear consensus → "high"
        - Very split opinions → "low"
        """
```

### prompts.py — Analysis Prompt Templates

All prompts for analysis LLM calls. Key prompts:

1. **THEME_CODING_PROMPT** — "Analyze this focus group transcript. For each participant message, assign 1-3 topic codes..."
2. **THEME_CLUSTERING_PROMPT** — "Given these topic codes from a focus group, cluster them into 5-7 major themes..."
3. **SENTIMENT_BATCH_PROMPT** — "Score each of these focus group responses from -1 (very negative) to 1 (very positive)..."
4. **CONCEPT_SCORE_PROMPT** — "Based on this participant's statements throughout the discussion, score their reaction on these metrics..."
5. **EXECUTIVE_SUMMARY_PROMPT** — "Write a 3-5 sentence executive summary of these focus group findings..."
6. **RECOMMENDATION_PROMPT** — "Based on these focus group results, provide a clear GO/ITERATE/NO-GO recommendation..."

## Testing Requirements

All tests must work with MockLLMClient (no real API calls).

**test_analyzer.py:**
- Test full analysis pipeline produces complete AnalysisReport
- Verify all required fields are populated
- Verify recommendation logic: mock high scores → GO, mock low scores → NO-GO

**test_theme_extractor.py:**
- Test that themes are extracted (at least 3 themes)
- Test theme prevalence is between 0 and 1
- Test each theme has supporting quotes

**test_concept_scorer.py:**
- Test all 6 metrics are populated
- Test Top-2-Box scores are between 0 and 1
- Test excitement_score calculation matches formula
- Test per-participant scores exist for each persona

**test_segment_analyzer.py:**
- Test that at least 2 segments are identified
- Test each segment has participant_ids
- Test segments with no meaningful difference are excluded

## Important

- The analysis engine is LLM-heavy but must be fully testable with MockLLMClient
- For mock mode: derive all scores deterministically from persona profiles (opinion_valence, OCEAN scores)
- For real mode: use LLM calls with structured JSON output parsing
- Import from existing modules: `from discussion.models import DiscussionTranscript, DiscussionMessage` and `from persona_engine.models import Persona`
- Keep LLM calls minimal: batch where possible (e.g., score all sentiments in one call)
- The AnalysisReport is what gets turned into the client-facing report (next task)
- Use the existing `LLMClient` / `MockLLMClient` from `src/discussion/llm_client.py`

## Requirements

No new dependencies needed. Use existing pydantic, numpy, httpx.
