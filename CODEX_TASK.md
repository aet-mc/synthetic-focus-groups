# Codex Task: Build Persona Engine

## What to Build

A Python persona engine for synthetic focus groups. This is the core module that generates grounded, diverse consumer personas for AI-powered market research simulations.

## Project Structure

```
src/
├── persona_engine/
│   ├── __init__.py
│   ├── generator.py        # Main PersonaGenerator class
│   ├── demographics.py     # Census-based demographic sampling
│   ├── psychographics.py   # OCEAN, VALS, Schwartz value assignment
│   ├── consumer_behavior.py # Purchase style, price sensitivity, brand loyalty derivation
│   ├── voice.py            # Communication style derivation (vocab, verbosity, hedging)
│   ├── diversity.py        # Diversity checker & enforcer
│   ├── models.py           # Pydantic data models for Persona, Demographics, Psychographics, etc.
│   └── opinion_seeder.py   # Pre-seed independent opinions before group discussion
├── tests/
│   ├── test_generator.py
│   ├── test_demographics.py
│   ├── test_psychographics.py
│   ├── test_diversity.py
│   └── test_opinion_seeder.py
└── requirements.txt
```

## Detailed Specifications

### models.py — Pydantic Data Models

```python
class Demographics(BaseModel):
    age: int  # 18-85
    gender: str  # "male", "female", "non-binary"
    income: int  # Annual household income in USD
    education: str  # "high_school", "some_college", "bachelors", "masters", "doctorate"
    occupation: str  # Free text, realistic for demographics
    location: Location  # state, metro_area, urban/suburban/rural
    household_type: str  # "single", "married_no_kids", "married_with_kids", "single_parent"
    race_ethnicity: str  # Census categories

class Location(BaseModel):
    state: str
    metro_area: str | None
    urbanicity: str  # "urban", "suburban", "rural"

class OceanScores(BaseModel):
    openness: float  # 0-100
    conscientiousness: float  # 0-100
    extraversion: float  # 0-100
    agreeableness: float  # 0-100
    neuroticism: float  # 0-100

class SchwartzValues(BaseModel):
    primary: str  # One of 10 Schwartz values
    secondary: str
    tertiary: str | None = None

class Psychographics(BaseModel):
    ocean: OceanScores
    vals_type: str  # "innovator", "thinker", "achiever", "experiencer", "believer", "striver", "maker", "survivor"
    schwartz_values: SchwartzValues
    
class ConsumerProfile(BaseModel):
    price_sensitivity: float  # 0-1 (0=insensitive, 1=very sensitive)
    brand_loyalty: float  # 0-1
    research_tendency: float  # 0-1 (how much they research before buying)
    impulse_tendency: float  # 0-1
    social_influence: float  # 0-1 (susceptibility to peer/social proof)
    risk_tolerance: float  # 0-1
    category_engagement: str  # "heavy", "moderate", "light", "non_user"
    decision_style: str  # "analytical", "emotional", "habitual", "impulsive"

class VoiceProfile(BaseModel):
    vocabulary_level: str  # "basic", "moderate", "advanced", "expert"
    verbosity: str  # "terse", "moderate", "verbose"
    hedging_tendency: float  # 0-1 (uses qualifiers like "maybe", "I think")
    emotional_expressiveness: float  # 0-1
    assertiveness: float  # 0-1
    humor_tendency: float  # 0-1
    communication_style: str  # "direct", "diplomatic", "analytical", "storytelling"

class Persona(BaseModel):
    id: str  # UUID
    name: str  # Realistic first name
    demographics: Demographics
    psychographics: Psychographics
    consumer: ConsumerProfile
    voice: VoiceProfile
    initial_opinion: str | None = None  # Pre-seeded opinion on the product concept
    opinion_valence: float | None = None  # -1 to 1 (negative to positive)
```

### demographics.py — Census-Anchored Sampling

Use hardcoded but realistic US demographic distributions (we'll hook up Census API later):

- Joint distributions for age × gender × income × education
- Regional distributions (state-level population weights)
- Urbanicity by region
- Occupation by education level
- Household type by age × gender

Key function:
```python
def sample_demographics(
    n: int,
    constraints: dict | None = None  # e.g. {"age_range": (25, 45), "income_min": 50000}
) -> list[Demographics]:
```

The constraints allow clients to specify their target market. Unconstrained dimensions are sampled from census distributions. Use numpy for weighted random sampling.

### psychographics.py — OCEAN + VALS + Schwartz

OCEAN scoring rules:
- Base distribution: Normal(mean=50, std=15) for each trait
- Age adjustments: Extraversion -0.3/decade after 30, Agreeableness +0.2/decade, Conscientiousness peaks 40-60
- Gender adjustments: Women +4 Agreeableness, +4 Neuroticism (small effect sizes)
- Apply stratified sampling: divide OCEAN space into quadrants, ensure coverage

VALS derivation from OCEAN + demographics:
- Innovators: High O, High C, High income, High education
- Thinkers: High O, High C, Moderate E, High education
- Achievers: High C, High E, Moderate-High income
- Experiencers: High O, High E, Low C, Younger
- Believers: Low O, High A, High C, Lower education
- Strivers: High E, Low C, Lower income
- Makers: Low E, High C, Lower income, practical occupations
- Survivors: Low across most traits, lowest income

Schwartz values: assign 2-3 from the 10 types, weighted by OCEAN profile:
- High O → Self-Direction, Stimulation
- High C → Achievement, Security
- High E → Power, Hedonism
- High A → Benevolence, Universalism, Conformity
- High N → Security, Tradition

### consumer_behavior.py — Purchase Behavior Derivation

Derive all ConsumerProfile fields from demographics + psychographics:

- price_sensitivity: inverse of income (normalized) × (1 - conscientiousness/200)
- brand_loyalty: low openness × high conscientiousness
- research_tendency: conscientiousness × 0.7 + neuroticism × 0.3
- impulse_tendency: inverse of conscientiousness × extraversion
- social_influence: agreeableness × 0.5 + extraversion × 0.3 + neuroticism × 0.2
- risk_tolerance: openness × 0.4 + extraversion × 0.3 - neuroticism × 0.3
- category_engagement: random weighted by age/income relevance
- decision_style: derived from dominant OCEAN trait

### voice.py — Communication Style

Derive VoiceProfile from demographics + psychographics:
- vocabulary_level: from education
- verbosity: from extraversion (high E = verbose)
- hedging_tendency: from neuroticism × agreeableness (high = more hedging)
- emotional_expressiveness: from extraversion × (1 - conscientiousness)
- assertiveness: from extraversion × (1 - agreeableness)
- humor_tendency: from openness × extraversion
- communication_style: from dominant trait combination

### diversity.py — Diversity Checker & Enforcer

```python
class DiversityChecker:
    def check(self, personas: list[Persona]) -> DiversityReport:
        """Check if persona group meets diversity requirements"""
    
    def enforce(self, personas: list[Persona], target: DiversityTarget) -> list[Persona]:
        """Re-sample personas that violate diversity constraints"""

class DiversityReport(BaseModel):
    opinion_entropy: float  # Shannon entropy of opinion valences
    personality_spread: dict[str, float]  # Std dev of each OCEAN trait
    passes: bool
    issues: list[str]
```

Diversity rules:
- At least 1 persona with agreeableness < 30 (contrarian)
- At least 1 persona with openness > 80 (enthusiast)  
- At least 1 persona with openness < 30 (skeptic)
- At least 1 persona with neuroticism > 70 (worrier)
- Opinion entropy > 1.5 bits
- Each OCEAN trait std dev > 15 across the group
- Max 5/8 personas with same opinion valence sign

### opinion_seeder.py — Pre-seed Independent Opinions

```python
class OpinionSeeder:
    def seed_opinions(
        self,
        personas: list[Persona],
        product_concept: str,
        category: str
    ) -> list[Persona]:
        """Generate independent initial opinions for each persona.
        Uses persona's psychographic profile to determine likely reaction.
        Does NOT use LLM — uses rules-based opinion generation."""
```

For MVP, use rules-based opinion seeding (no LLM dependency):
- opinion_valence from OCEAN: high O + low N → positive; low O + high N → negative
- category_engagement affects intensity: heavy users have stronger opinions
- price_sensitivity affects value perception
- Sprinkle randomness (±0.2) to prevent determinism

### generator.py — Main Orchestrator

```python
class PersonaGenerator:
    def generate(
        self,
        n: int = 8,
        constraints: dict | None = None,
        product_concept: str | None = None,
        category: str | None = None
    ) -> list[Persona]:
        """Full pipeline: demographics → psychographics → consumer → voice → diversity check → opinion seeding"""
```

## Requirements

- Python 3.11+
- pydantic >= 2.0
- numpy
- pytest for tests

## Testing

Write tests that:
1. Generate 8 personas with no constraints → verify diversity passes
2. Generate with tight constraints (e.g., women 25-35 in tech) → verify demographics match
3. Run generator 10 times → verify opinion valences span the full range
4. Verify OCEAN distributions match expected age/gender adjustments
5. Verify VALS assignment logic is consistent
6. Verify diversity enforcer re-samples when constraints are violated

## Important

- NO LLM calls in this module. Everything is rules-based + statistical sampling.
- The LLM integration comes later in the moderator/discussion modules.
- Use type hints everywhere. Modern Python style.
- Make it fast — generating 8 personas should take < 100ms.
