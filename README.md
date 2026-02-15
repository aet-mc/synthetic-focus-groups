# Synthetic Focus Groups

AI-powered market research that simulates realistic focus group discussions in minutes instead of weeks. Census-grounded personas, real group dynamics, quantified insights.

## Why This Exists

Traditional focus groups cost **$15-65K**, take **4-8 weeks**, and use **8-12 people**. We run them in **under 10 minutes** with AI personas grounded in real demographic data, producing structured insights at a fraction of the cost.

**The key insight:** We don't just ask 8 AI personas their opinions independently. We simulate actual group dynamics — conformity pressure, opinion leadership, cascade effects, the quiet person who speaks up in round 3. That's what makes focus groups valuable, and it's what every other "AI survey" tool misses.

## How It Works

```
Concept → Persona Generation → Moderated Discussion → Analysis → Report
              (census-grounded)    (multi-agent, 5 phases)   (themes, sentiment, scores)
```

1. **Persona Engine** — Generates diverse participants grounded in US Census demographics, OCEAN personality traits, consumer behavior profiles, and unique voice patterns. Enforces diversity (age, gender, income, personality) automatically.

2. **Discussion Simulator** — Multi-agent orchestration where each persona responds in character across 5 discussion phases (introduction, exploration, deep dive, debate, conclusion). Personas influence each other realistically based on agreeableness, extraversion, and opinion strength.

3. **Analysis Engine** — Extracts themes, sentiment trajectories, concept scores (purchase intent, appeal, uniqueness), and generates actionable recommendations. All analysis uses structured JSON extraction for reliability.

4. **Report Generator** — Produces a comprehensive HTML report with participant profiles, discussion transcript, theme analysis, sentiment charts, and a go/iterate/no-go recommendation.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run a focus group
cd src
python demo.py "Your product concept here" \
  --category "consumer electronics" \
  --provider google \
  --output report.html

# Run tests
pytest tests/ -x -q
```

### LLM Providers

| Provider | Flag | Notes |
|----------|------|-------|
| Google Gemini | `--provider google` | Free tier, recommended for testing |
| OpenAI | `--provider openai` | GPT-4o |
| Anthropic | `--provider anthropic` | Claude Sonnet |
| Groq | `--provider groq` | Fast, rate-limited free tier |
| DeepSeek | `--provider deepseek` | Budget option |

Set the corresponding API key as an environment variable (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.)

## Architecture

```
src/
├── persona_engine/          # Persona generation & diversity enforcement
│   ├── demographics.py      # Census-grounded demographic sampling
│   ├── psychographics.py    # OCEAN personality, values, attitudes
│   ├── consumer_behavior.py # Price sensitivity, brand loyalty, adoption style
│   ├── diversity.py         # Gender balance, opinion spread, personality mix
│   ├── generator.py         # Orchestrates persona creation
│   ├── opinion_seeder.py    # Initial opinion generation from persona traits
│   └── voice.py             # Unique communication style per persona
├── discussion/              # Multi-agent discussion simulation
│   ├── simulator.py         # 5-phase discussion orchestration
│   ├── participant.py       # Individual persona agent
│   ├── moderator.py         # AI moderator that guides discussion
│   ├── llm_client.py        # Multi-provider LLM client with rate limiting & circuit breaking
│   └── prompts.py           # All prompt templates
├── analysis/                # Post-discussion analysis
│   ├── analyzer.py          # Main analysis orchestrator
│   ├── theme_extractor.py   # Thematic coding & clustering
│   ├── concept_scorer.py    # Batch concept scoring (purchase intent, appeal, etc.)
│   ├── sentiment.py         # Sentiment trajectory analysis
│   └── segment_analyzer.py  # Demographic segment analysis
├── report/                  # HTML report generation
│   ├── generator.py         # Jinja2-based report builder
│   └── charts.py            # SVG chart generation
├── tests/                   # 44+ tests, all passing
└── demo.py                  # CLI entry point
```

## Key Features

- **Census-grounded personas** — Demographics sampled from real distributions, not random
- **OCEAN personality model** — Each persona has measurable personality traits that affect behavior
- **Group dynamics simulation** — Conformity pressure, opinion shifts, leadership emergence
- **Anchored opinions** — Personas resist groupthink based on their personality (low agreeableness = more independent)
- **Multi-provider LLM support** — Google, OpenAI, Anthropic, Groq, DeepSeek, Nvidia
- **Rate limiting & circuit breaking** — Production-ready LLM client with sliding window rate limiter and automatic circuit breaker
- **Structured JSON extraction** — All analysis uses `complete_json()` with automatic markdown fence stripping
- **Batch scoring** — Concept scoring in 1 LLM call instead of N, with automatic fallback
- **Comprehensive HTML reports** — Participant profiles, transcript, themes, sentiment, scores, recommendations
- **44+ unit tests** — Full test coverage of persona generation, diversity enforcement, and analysis

## Sample Output

A focus group on *"A wearable AI pin that replaces your smartphone"*:
- **8 participants** — ages 24-82, balanced gender, diverse occupations
- **61 messages** across 5 discussion phases
- **7 themes** extracted (privacy concerns, accessibility, screen dependence, etc.)
- **Recommendation: ITERATE** — 25% purchase intent, 29% excitement
- Full HTML report with transcript, charts, and actionable insights

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest src/tests/ -x -q

# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT — see [LICENSE](LICENSE)
