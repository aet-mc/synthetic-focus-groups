# Codex Task 2: Build Moderator Agent + Discussion Simulator

## Context

The persona engine is complete in `src/persona_engine/`. It generates grounded, diverse personas with demographics, OCEAN psychographics, VALS types, consumer profiles, voice profiles, and pre-seeded opinions.

Now build the **moderator agent** and **discussion simulator** — the core of the focus group experience.

## What to Build

### Project Structure (add to existing)

```
src/
├── persona_engine/          # ALREADY BUILT - don't modify
├── discussion/
│   ├── __init__.py
│   ├── moderator.py         # Moderator agent (manages discussion flow)
│   ├── simulator.py         # Multi-agent discussion loop
│   ├── participant.py       # Individual participant agent (wraps persona + LLM)
│   ├── transcript.py        # Transcript recording and formatting
│   ├── prompts.py           # All LLM prompt templates
│   └── models.py            # Discussion-specific data models
├── tests/
│   ├── ... (existing)
│   ├── test_moderator.py
│   ├── test_simulator.py
│   ├── test_participant.py
│   └── test_transcript.py
```

## Detailed Specifications

### models.py — Discussion Data Models

```python
from pydantic import BaseModel
from enum import Enum

class DiscussionPhase(str, Enum):
    WARMUP = "warmup"           # Ice-breaker, get people talking
    EXPLORATION = "exploration"  # Open-ended, what comes to mind
    DEEP_DIVE = "deep_dive"     # Specific feature/concept probing
    REACTION = "reaction"       # Direct reaction to stimulus (product concept, ad, etc.)
    SYNTHESIS = "synthesis"     # Wrap-up, final thoughts, purchase intent

class MessageRole(str, Enum):
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    SYSTEM = "system"  # Internal tracking only

class DiscussionMessage(BaseModel):
    role: MessageRole
    speaker_id: str  # persona.id or "moderator"
    speaker_name: str
    content: str
    phase: DiscussionPhase
    turn_number: int
    replied_to: str | None = None  # speaker_id they're responding to
    sentiment: float | None = None  # -1 to 1, detected from content
    changed_mind: bool = False  # Did this response indicate opinion shift

class DiscussionConfig(BaseModel):
    product_concept: str  # The product/concept being discussed
    category: str  # Product category
    stimulus_material: str | None = None  # Ad copy, product description, etc.
    num_personas: int = 8
    phases: list[DiscussionPhase] = [
        DiscussionPhase.WARMUP,
        DiscussionPhase.EXPLORATION,
        DiscussionPhase.DEEP_DIVE,
        DiscussionPhase.REACTION,
        DiscussionPhase.SYNTHESIS,
    ]
    questions_per_phase: int = 2
    max_responses_per_question: int = 5  # Not everyone speaks to every question
    temperature: float = 0.9  # For participant responses
    model: str = "anthropic/claude-sonnet-4-20250514"  # Default LLM

class DiscussionTranscript(BaseModel):
    config: DiscussionConfig
    messages: list[DiscussionMessage] = []
    personas: list  # List of Persona objects (from persona_engine)
```

### prompts.py — All Prompt Templates

Create well-crafted prompt templates for:

1. **PERSONA_SYSTEM_PROMPT** — Sets up LLM as a specific persona. Must include:
   - Name, age, occupation, location
   - Personality description derived from OCEAN scores (natural language, not numbers)
   - Communication style from VoiceProfile
   - Consumer behavior tendencies
   - Category engagement level
   - Pre-seeded opinion (their private initial reaction)
   - Rules: stay in character, don't be artificially agreeable, express real disagreement if your personality would

2. **MODERATOR_QUESTION_PROMPT** — Generates the next moderator question based on:
   - Current phase
   - What's been said so far (summary)
   - Which personas haven't spoken much (need to be drawn out)
   - Phase-specific question style:
     - WARMUP: "Tell us about yourself and your experience with [category]"
     - EXPLORATION: Open-ended, "What comes to mind when you hear [concept]?"
     - DEEP_DIVE: Probing specific features, pricing, concerns
     - REACTION: Direct stimulus response, "Having seen this, would you...?"
     - SYNTHESIS: "If this were available tomorrow, would you buy it? Why or why not?"

3. **PARTICIPANT_RESPONSE_PROMPT** — Generates a participant's response. Must include:
   - The persona system prompt
   - The full discussion so far (or recent context window)
   - The moderator's current question
   - Instructions to:
     - Respond naturally in 1-4 sentences (not essays)
     - React to what others have said (agree, disagree, build on)
     - Stay in character per their personality
     - High-agreeableness: more likely to agree with others
     - Low-agreeableness: more likely to challenge/disagree
     - High-extraversion: longer responses, more assertive
     - Low-extraversion: shorter, may need direct questions to speak up

4. **OPINION_SHIFT_DETECTION_PROMPT** — After each response, detect if the persona shifted their opinion from their pre-seeded position. Returns a boolean + new valence if shifted.

### participant.py — Individual Participant Agent

```python
class Participant:
    def __init__(self, persona: Persona, llm_client):
        """Wraps a persona with LLM capability"""
    
    def build_system_prompt(self) -> str:
        """Convert persona profile into natural language system prompt"""
    
    async def respond(
        self,
        moderator_question: str,
        discussion_context: list[DiscussionMessage],
        phase: DiscussionPhase,
    ) -> DiscussionMessage:
        """Generate a response as this persona"""
    
    def should_speak(self, phase: DiscussionPhase, turn: int) -> bool:
        """Probabilistic: high-E personas speak more often, low-E less.
        Returns True/False based on extraversion + randomness.
        High E: 85% chance to speak
        Medium E: 60% chance
        Low E: 35% chance
        But: if directly addressed by moderator, always speak"""
```

**CRITICAL for participant.py:** The `build_system_prompt()` method must translate OCEAN scores into NATURAL LANGUAGE personality descriptions. Do NOT include raw numbers. Examples:
- High Openness (80+): "You're naturally curious and love trying new things. You're drawn to novel products and aren't afraid of the unfamiliar."
- Low Agreeableness (<30): "You're independent-minded and don't go along with the group just to be polite. If you disagree, you say so directly."
- High Neuroticism (70+): "You tend to worry about purchases — what if it doesn't work? What if there's a better option? You research carefully and need reassurance."

### moderator.py — Moderator Agent

```python
class Moderator:
    def __init__(self, config: DiscussionConfig, llm_client):
        self.config = config
        self.discussion_guide: list[str] = []  # Pre-generated questions
    
    async def generate_discussion_guide(self) -> list[str]:
        """Pre-generate all questions for the discussion (2 per phase × 5 phases = 10 questions).
        Uses LLM to create context-appropriate questions."""
    
    async def generate_question(
        self,
        phase: DiscussionPhase,
        transcript_so_far: list[DiscussionMessage],
        quiet_personas: list[str],  # Names of personas who haven't spoken much
    ) -> str:
        """Generate the next question, optionally drawing out quiet participants"""
    
    async def generate_followup(
        self,
        last_response: DiscussionMessage,
        phase: DiscussionPhase,
    ) -> str | None:
        """Optionally generate a follow-up probe if the response was interesting.
        Returns None if no follow-up needed (most of the time).
        Follow up ~20% of the time on interesting/surprising responses."""
    
    def select_respondents(
        self,
        participants: list[Participant],
        question: str,
        phase: DiscussionPhase,
        turn: int,
    ) -> list[Participant]:
        """Select which participants respond to this question.
        Uses should_speak() probability + ensures quiet ones get drawn in.
        Typically 3-6 out of 8 respond per question (not everyone every time)."""
```

### simulator.py — Discussion Orchestrator

```python
class DiscussionSimulator:
    def __init__(self, config: DiscussionConfig, llm_client=None):
        """If llm_client is None, use a MockLLMClient for testing"""
    
    async def run(self) -> DiscussionTranscript:
        """Full discussion simulation pipeline:
        
        1. Generate personas (using PersonaGenerator)
        2. Create Participant agents for each persona
        3. Create Moderator
        4. For each phase:
           a. Moderator asks question
           b. Select respondents (3-6 per question)
           c. Each respondent generates response (sequentially, so they see prior responses)
           d. Moderator optionally follows up
           e. Repeat for questions_per_phase questions
        5. Return complete transcript
        
        Total expected: ~10 questions × ~4 responses each = ~40-50 participant messages
        Plus ~10-15 moderator messages = ~50-65 total messages
        """
    
    async def _run_phase(
        self,
        phase: DiscussionPhase,
        participants: list[Participant],
        moderator: Moderator,
        transcript: DiscussionTranscript,
    ) -> None:
        """Run a single phase of the discussion"""
```

### transcript.py — Transcript Formatting

```python
class TranscriptFormatter:
    @staticmethod
    def to_markdown(transcript: DiscussionTranscript) -> str:
        """Format transcript as readable markdown.
        Include phase headers, speaker names, and content.
        Example:
        
        ## Phase: Warm-Up
        
        **Moderator:** Welcome everyone! Let's start by...
        
        **Emma (27, Marketing Coordinator):** Thanks! I've been...
        
        **David (64, Retired Engineer):** Well, in my experience...
        """
    
    @staticmethod
    def to_json(transcript: DiscussionTranscript) -> str:
        """Export transcript as JSON"""
    
    @staticmethod
    def summary_stats(transcript: DiscussionTranscript) -> dict:
        """Return summary statistics:
        - Total messages by role
        - Messages per phase
        - Messages per participant
        - Average sentiment by phase
        - Opinion shifts detected
        - Most/least active participants
        """
```

### LLM Client

Create a simple async LLM client abstraction:

```python
class LLMClient:
    """Async wrapper for LLM API calls"""
    
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 300,
    ) -> str:
        """Call LLM and return response text"""

class MockLLMClient(LLMClient):
    """Returns deterministic responses for testing.
    Generates varied mock responses based on persona name + question content.
    Not random — uses hash of inputs for reproducibility."""
```

Put the LLM client in `src/discussion/llm_client.py`.

### Testing

**test_moderator.py:**
- Test that discussion guide generates 10 questions (2 per phase × 5 phases)
- Test that quiet personas get drawn out (moderator mentions them by name)
- Test select_respondents returns 3-6 participants

**test_simulator.py:**
- Test full simulation with MockLLMClient
- Verify transcript has messages from all 5 phases
- Verify all personas spoke at least once
- Verify moderator messages exist between participant messages
- Verify total message count is reasonable (40-65)

**test_participant.py:**
- Test build_system_prompt contains natural language personality (no raw numbers)
- Test should_speak probability: high-E persona speaks more often over 100 trials
- Test response includes speaker_id and correct phase

**test_transcript.py:**
- Test to_markdown produces valid markdown with phase headers
- Test summary_stats returns correct message counts

## Requirements

Add to existing requirements.txt:
- httpx (for async HTTP to LLM APIs)
- No other new deps needed

## Important

- Use `async/await` throughout — the discussion loop is inherently sequential (each response depends on prior context)
- MockLLMClient is critical — all tests must work WITHOUT any real API calls
- The MockLLMClient should generate realistic-looking varied responses (use persona name + question hash to vary output)
- Keep responses SHORT in prompts: instruct 1-4 sentences, not paragraphs
- The moderator should feel natural — not robotic "Question 1:", but conversational transitions
- Import PersonaGenerator from the existing persona_engine module: `from persona_engine.generator import PersonaGenerator`
