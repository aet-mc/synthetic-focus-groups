# Product Specification: Synthetic Focus Groups Platform

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CLIENT INTERFACE                     │
│  Study Setup → Persona Config → Live Session → Report│
└─────────────┬───────────────────────────┬───────────┘
              │                           │
    ┌─────────▼──────────┐     ┌─────────▼──────────┐
    │  PERSONA ENGINE     │     │  ANALYSIS ENGINE    │
    │                     │     │                     │
    │  Census Data        │     │  Sentiment Analysis │
    │  BLS Consumer Data  │     │  Theme Extraction   │
    │  Psychographic      │     │  Statistical Summary│
    │  Models (OCEAN)     │     │  Quote Selection    │
    │  Category Behavior  │     │  Confidence Scoring │
    │  Purchase History   │     │  Segment Comparison │
    │  Media Consumption  │     │  Recommendation Gen │
    └─────────┬──────────┘     └─────────▲──────────┘
              │                           │
    ┌─────────▼───────────────────────────┴──────────┐
    │            DISCUSSION SIMULATOR                  │
    │                                                  │
    │  Moderator Agent (funnel methodology)            │
    │       │                                          │
    │       ├── Persona Agent 1 (cautious buyer)       │
    │       ├── Persona Agent 2 (early adopter)        │
    │       ├── Persona Agent 3 (skeptic)              │
    │       ├── Persona Agent 4 (price-sensitive)      │
    │       ├── Persona Agent 5 (brand loyal)          │
    │       ├── Persona Agent 6 (impulse buyer)        │
    │       ├── Persona Agent 7 (researcher)           │
    │       └── Persona Agent 8 (social influencer)    │
    │                                                  │
    │  Group Dynamics Engine:                          │
    │    - Conformity pressure modeling                │
    │    - Opinion leader detection                    │
    │    - Cascade effects                             │
    │    - Disagreement escalation                     │
    │    - Social desirability bias                    │
    └──────────────────────────────────────────────────┘
```

---

## Component 1: Persona Engine

### Purpose
Generate realistic, grounded AI personas that behave like real consumers from specific demographic segments.

### Data Layers

#### Layer 1: Demographics (Census API — Free)
- Age, income, education, location, household composition, occupation
- Weighted to match client's target market distribution
- Source: US Census Bureau API (api.census.gov)
- Data: American Community Survey (ACS) 5-year estimates

#### Layer 2: Consumer Behavior (BLS — Free)
- Actual spending patterns by demographic: groceries, dining, electronics, health, entertainment
- Price sensitivity derived from income-to-spending ratios
- Category engagement level (heavy buyer vs. occasional)
- Source: Bureau of Labor Statistics Consumer Expenditure Surveys

#### Layer 3: Psychographics (OCEAN/Big Five Personality Model)
- **Openness** — Creativity, curiosity, willingness to try new things
- **Conscientiousness** — Organization, discipline, planning tendency
- **Extraversion** — Assertiveness, sociability, talkativeness
- **Agreeableness** — Cooperation, trust, conformity tendency
- **Neuroticism** — Emotional reactivity, anxiety, risk aversion

Each trait score (0-100) determines:
- How they express opinions (assertive vs. hedging)
- Whether they lead or follow in group discussion
- Risk tolerance in purchasing decisions
- How they process new information (analytical vs. emotional)
- Response to social pressure (conform vs. resist)

Distribution: Use population-level OCEAN distributions (Costa & McCrae norms) adjusted by demographics.

#### Layer 4: Category-Specific Context
- For each study, personas receive category knowledge
- Example: "How does a price-sensitive suburban mom with high conscientiousness think about organic baby food?"
- LLM synthesizes demographic + personality + category into behavioral model
- Includes: brand awareness, purchase frequency, decision factors, information sources

#### Layer 5: Communication Style
- Vocabulary complexity matches education level
- Response length varies by extraversion (high E = longer responses)
- Use of qualifiers matches confidence and personality:
  - Low confidence: "I think maybe..." "I'm not sure but..."
  - High confidence: "Absolutely" "100%" "No question"
- Some personas are articulate, some ramble, some are terse
- Filler words, interruptions, tangents — like real people

### Persona Profile Schema

```json
{
  "id": "persona_001",
  "name": "Sarah M.",
  "demographics": {
    "age": 34,
    "gender": "female",
    "income": 82000,
    "education": "bachelors",
    "location": "suburban Denver, CO",
    "household": "married, 2 children (ages 3, 6)",
    "occupation": "marketing coordinator"
  },
  "psychographics": {
    "openness": 72,
    "conscientiousness": 81,
    "extraversion": 55,
    "agreeableness": 68,
    "neuroticism": 45
  },
  "consumer_behavior": {
    "annual_grocery_spend": 9200,
    "dining_out_frequency": "2x/week",
    "price_sensitivity": 0.65,
    "brand_loyalty": 0.45,
    "research_before_purchase": true,
    "primary_info_sources": ["Instagram", "mom blogs", "Amazon reviews"],
    "impulse_buy_tendency": 0.3
  },
  "category_context": {
    "category": "organic baby food",
    "current_brands": ["Happy Baby", "Earth's Best"],
    "purchase_frequency": "weekly",
    "key_decision_factors": ["ingredients", "price", "convenience"],
    "awareness_of_concept": "low"
  },
  "communication_style": {
    "verbosity": "moderate",
    "vocabulary_level": "college",
    "qualifier_frequency": "medium",
    "storytelling_tendency": "high",
    "directness": "medium"
  },
  "group_dynamics": {
    "speak_probability_base": 0.55,
    "conformity_tendency": 0.65,
    "opinion_leadership": 0.35,
    "reaction_to_disagreement": "considers",
    "social_desirability_bias": 0.5
  }
}
```

---

## Component 2: Discussion Simulator

### The Moderator Agent

Implements professional focus group moderation methodology:

#### Funnel Discussion Guide Structure
1. **Warm-Up (5 min)** — Low-stakes, easy questions. Build rapport. "Tell us your name and one thing you bought this week that made you happy."
2. **Context Setting (10 min)** — Establish category relationship. "How do you typically decide what to buy in [category]?"
3. **Concept Introduction (5 min)** — Present the product/concept to the group
4. **Initial Reactions (15 min)** — Open-ended first impressions. Round-robin format.
5. **Deep Exploration (25 min)** — Probe specific aspects: features, pricing, messaging, packaging
6. **Projective Techniques (15 min)** — "If this product were a person, who would it be?" / "Complete this sentence: I would buy this if..."
7. **Purchase Intent (10 min)** — Direct questions: likelihood to buy, how much would you pay, who would you recommend this to
8. **Wrap-Up (5 min)** — "One thing you want the makers of this product to know"

#### Moderator Behaviors
- Follows the guide but adapts: if a thread is rich, follows up; if dead, moves on
- Draws out quiet personas: "Sarah, you've been quiet — what's your take?"
- Manages dominators: "Good point. Let's hear from someone who hasn't spoken yet."
- Plays devil's advocate: "Interesting that most of you like it. What could go wrong?"
- Probes shallow answers: "You said 'it's nice.' What specifically makes it nice for you?"
- Summarizes and reflects: "So it sounds like price is the main concern. Is that right?"

### Persona Agent Behavior Model

Each conversation turn, every persona agent:

1. **Hears** what the moderator and other personas said
2. **Processes** through personality filter:
   - Do they agree or disagree with what was said?
   - How strongly? (conviction score)
   - Are they feeling social pressure? (conformity check)
3. **Decides whether to speak:**
   - Base probability from extraversion + group position
   - Increased if directly addressed by moderator
   - Increased if they strongly disagree
   - Decreased if they just spoke recently
   - Decreased if dominant persona is holding the floor
4. **Generates response** considering:
   - Their persona profile (demographics, personality, category context)
   - What others said (agreement, disagreement, building on ideas)
   - Social pressure (may soften criticism if room is positive)
   - Communication style (vocabulary, length, qualifiers)
5. **Updates internal state:**
   - Opinion may shift based on persuasive arguments heard
   - Confidence may increase or decrease
   - Social comfort level evolves through the session

### Group Dynamics Engine

#### Conformity Pressure (Asch Effect)
```python
def calculate_conformity_pressure(persona, group_sentiment):
    """
    When 3+ personas agree, dissenting voices face pressure.
    High agreeableness = more likely to conform.
    """
    agreement_ratio = count_agreeing(group_sentiment) / total_speakers
    pressure = agreement_ratio * persona.agreeableness * CONFORMITY_WEIGHT
    
    # Strong conviction resists pressure
    resistance = persona.conviction * (1 - persona.agreeableness)
    
    return max(0, pressure - resistance)
```

#### Opinion Leadership
```python
def calculate_influence(persona, group_state):
    """
    High extraversion + perceived expertise = opinion leader.
    Their statements carry more weight for others.
    """
    influence = (
        persona.extraversion * 0.4 +
        persona.perceived_expertise * 0.4 +
        persona.speak_count_ratio * 0.2  # more speaking = more influence
    )
    return influence
```

#### Cascade Effects
When opinion leader makes a strong statement:
1. High-agreeableness personas shift toward leader's position
2. Low-agreeableness personas may push back harder (polarization)
3. Neutral personas wait to see which direction group moves

#### Social Desirability Bias
- Personas with high agreeableness soften negative feedback in group
- "I don't like it" becomes "It's not really for me, but I can see how others might"
- This bias is INTENTIONAL — it matches real focus group behavior

#### The Quiet Dissenter
- Low-extraversion + low-agreeableness + strong conviction = the person who barely speaks but has the most valuable insight
- Moderator agent specifically draws these personas out
- Their responses are flagged in analysis as potential hidden insights

### Implementation Pattern

```python
class DiscussionSimulator:
    def __init__(self, personas: List[PersonaAgent], moderator: ModeratorAgent):
        self.personas = personas
        self.moderator = moderator
        self.transcript = []
        self.turn_count = 0
        
    async def run_session(self, discussion_guide: DiscussionGuide):
        for section in discussion_guide.sections:
            # Moderator introduces the section
            mod_prompt = self.moderator.introduce_section(section)
            self.transcript.append({"speaker": "moderator", "text": mod_prompt})
            
            # Run discussion rounds for this section
            for round_num in range(section.max_rounds):
                responses = await self.run_round(mod_prompt, section)
                
                if not responses:  # no one wants to speak
                    break
                    
                # Moderator decides: follow up, redirect, or move on
                mod_action = self.moderator.evaluate_round(responses, section)
                
                if mod_action.type == "follow_up":
                    mod_prompt = mod_action.prompt
                    self.transcript.append({"speaker": "moderator", "text": mod_prompt})
                elif mod_action.type == "draw_out":
                    # Target a quiet persona
                    mod_prompt = mod_action.prompt
                    self.transcript.append({"speaker": "moderator", "text": mod_prompt})
                elif mod_action.type == "move_on":
                    break
                    
        return self.transcript
    
    async def run_round(self, prompt, section):
        responses = []
        # Determine speaking order (not always round-robin)
        for persona in self.get_speaking_order():
            should_speak = persona.decide_to_speak(
                prompt=prompt,
                previous_responses=responses,
                group_state=self.get_group_state()
            )
            if should_speak:
                response = await persona.generate_response(
                    prompt=prompt,
                    heard=responses,
                    group_state=self.get_group_state()
                )
                responses.append(response)
                self.transcript.append({
                    "speaker": persona.name,
                    "persona_id": persona.id,
                    "text": response.text,
                    "sentiment": response.sentiment,
                    "conviction": response.conviction
                })
                # Update group dynamics
                self.update_group_state(persona, response)
        return responses
```

---

## Component 3: Analysis Engine

### Quantitative Outputs

1. **Purchase Intent Distribution**
   - 5-point scale (Definitely would / Probably would / Might / Probably not / Definitely not)
   - Mean, standard deviation, confidence interval
   - Broken down by segment (age, income, personality type)

2. **Feature Importance Ranking**
   - Derived from discussion emphasis (how much each feature was discussed)
   - Weighted by sentiment (positive discussion vs. negative)
   - Statistical significance testing between features

3. **Net Promoter Score Equivalent**
   - Derived from advocacy language: "I'd tell my friends" = promoter
   - Detractor language: "I wouldn't recommend" / "I'd look elsewhere"
   - NPS score with confidence interval

4. **Sentiment Heat Map**
   - By topic area (product, price, packaging, brand, competition)
   - By persona segment
   - Over time (beginning vs. end of discussion — did sentiment shift?)

5. **Confidence Scoring**
   - How reliable are these results?
   - Based on: persona diversity, discussion depth, consensus level, consistency with known data

### Qualitative Outputs

1. **Top Themes (5-7)**
   - Extracted via clustering of discussion topics
   - Each theme with: summary, supporting quotes, prevalence, sentiment
   
2. **Moment of Truth**
   - The single most important exchange in the discussion
   - Where an insight emerged that the client didn't ask about
   - Highlighted in executive summary

3. **Unexpected Findings**
   - Things the client didn't ask about but should know
   - Flagged automatically when discussion diverges from guide topics

4. **Counter-Arguments**
   - What the skeptics said and why
   - Organized as objections + potential responses
   - Critical for messaging development

5. **Verbatim Quotes**
   - Best quotes selected for each theme
   - Attributed to persona with demographic context
   - Ready for client presentations

### Report Structure

```
Executive Summary (1 page)
├── Top 3 findings
├── Purchase intent summary
├── Key recommendation
│
Methodology (1 page)
├── Persona composition
├── Discussion guide overview
├── Confidence scoring explanation
│
Key Findings (5-10 pages)
├── Theme 1: [Title]
│   ├── Summary
│   ├── Supporting quotes
│   ├── Segment breakdown
│   └── Implications
├── Theme 2-5: [Same structure]
│
Purchase Intent Analysis (2 pages)
├── Overall distribution
├── Segment comparison
├── Price sensitivity
│
Group Dynamics Insights (1-2 pages)
├── Opinion leader analysis
├── Conformity patterns observed
├── Key disagreements and their resolution
│
Recommendations (1-2 pages)
├── Product/concept changes
├── Messaging suggestions
├── Target segment refinement
├── Suggested follow-up research
│
Appendices
├── Full transcript
├── Individual persona profiles
├── Statistical methodology
└── Raw data tables
```

---

## Component 4: Web Interface

### Study Setup Flow

1. **Describe Your Concept**
   - Text description of product/concept
   - Upload images, mockups, pricing info (optional)
   - Select study type: concept test, message test, pricing test, competitive eval

2. **Define Your Audience**
   - Quick presets: "US Women 25-45" / "Tech-savvy millennials" / "Budget-conscious families"
   - Advanced: Custom demographic criteria, psychographic filters
   - Number of personas: 8 (Explorer), 24 (Professional), 100+ (Enterprise)

3. **Customize Research Questions**
   - Auto-generated discussion guide based on study type
   - Client can add/modify questions
   - Projective technique selection

4. **Run Study**
   - Real-time progress: "Warm-up complete... Exploring initial reactions... Deep dive in progress..."
   - Optional: Live transcript view (watch the discussion unfold)
   - Estimated time: 5-15 minutes

5. **Review Report**
   - Interactive report in-browser
   - Download as PDF
   - Explore transcript with search/filter
   - Re-run with different audience (A/B testing)

### Tech Stack
- **Frontend:** Next.js / React (deployed on Cloudflare Pages)
- **Backend:** Python FastAPI (discussion simulation, analysis)
- **LLM:** Claude API (persona responses, analysis)
- **Data:** Census API, BLS API, local behavioral models
- **Database:** SQLite (study history) → PostgreSQL at scale
- **Payments:** Stripe
- **Hosting:** Cloudflare (Pages + Workers + D1)

---

## Development Phases

### Phase 1: Core Engine (Week 1-2)
- [ ] Persona generator with Census + OCEAN grounding
- [ ] Basic moderator agent with funnel methodology
- [ ] Multi-agent discussion loop (8 personas)
- [ ] Simple transcript output
- [ ] Test on 3 real product concepts

### Phase 2: Analysis + Report (Week 2-3)
- [ ] Theme extraction from transcripts
- [ ] Sentiment analysis per persona and aggregate
- [ ] Purchase intent calculation
- [ ] Quote selection algorithm
- [ ] PDF report generation
- [ ] Interactive web report

### Phase 3: Web Platform (Week 3-4)
- [ ] Study setup interface
- [ ] Audience configuration
- [ ] Real-time progress display
- [ ] Report viewer
- [ ] Stripe payment integration
- [ ] User accounts and study history

### Phase 4: Validation + Launch (Week 4-5)
- [ ] Run 3-5 parallel studies (synthetic + Prolific real humans)
- [ ] Calculate correlation metrics
- [ ] Publish validation data
- [ ] Product Hunt launch
- [ ] First 10 paying customers

### Phase 5: Iteration (Month 2-3)
- [ ] Group dynamics refinement based on user feedback
- [ ] Additional projective techniques
- [ ] Multi-language support
- [ ] API access for integrations
- [ ] A/B testing mode (same audience, different concepts)
