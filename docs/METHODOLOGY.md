# Methodology: How We Replicate Focus Group Research

## Overview

This document details the scientific and methodological foundation for synthetic focus groups. It covers how we replicate established qualitative research methods, the psychological models underlying persona behavior, and our validation framework.

---

## 1. Focus Group Methodology We Replicate

### The Funnel Approach (Krueger & Casey)

Standard focus group methodology uses a "funnel" discussion guide that moves from broad to narrow:

1. **Opening (Ice-Breaker):** Easy, non-threatening questions everyone can answer. Purpose: build rapport, establish that everyone's opinion matters.
   - Example: "Tell us your name and one purchase you made this week that you're happy about."

2. **Introduction:** Establish the topic without revealing the specific concept.
   - Example: "When you think about [category], what brands come to mind? How do you typically make decisions in this space?"

3. **Transition:** Bridge from general to specific.
   - Example: "What would make you switch from your current [category] product?"

4. **Key Questions (3-5):** The core of the research. Open-ended, probing.
   - Example: "Now I'm going to show you a new concept. Take a moment to look at it... What's your first reaction?"

5. **Ending:** Summarize, final thoughts, anything missed.
   - Example: "If you could say one thing to the makers of this product, what would it be?"

### Projective Techniques We Simulate

| Technique | How It Works | What It Reveals |
|-----------|-------------|-----------------|
| **Personification** | "If this brand were a person, describe them" | Brand personality perception |
| **Sentence Completion** | "I would buy this if..." | Purchase barriers and drivers |
| **Word Association** | "What's the first word that comes to mind?" | Immediate emotional reaction |
| **Obituary/Eulogy** | "This product just died. Write its obituary" | Perceived value and legacy |
| **Shopping Cart** | "Put this product in your cart or put it back. Why?" | Purchase intent with reasoning |
| **Price Thermometer** | "At what price is this too cheap? Too expensive? A bargain?" | Price sensitivity mapping (Van Westendorp) |

### Moderation Techniques We Simulate

| Technique | When Used | How Moderator Implements |
|-----------|----------|------------------------|
| **Round-Robin** | Warm-up, initial reactions | Address each persona by name in turn |
| **Laddering** | Shallow answers | "Why?" chain: "You said convenient — why is that important?" |
| **Devil's Advocate** | Too much agreement | "Interesting that everyone likes it. What could go wrong?" |
| **Redirect** | Dominant speaker | "Good points, Mike. Sarah, you've been thinking — what's your take?" |
| **Silence** | After big question | Let silence sit for 3-5 seconds. Introverts speak up |
| **Reflection** | Check understanding | "So it sounds like the main concern is price. Am I hearing that right?" |
| **Probe** | Surface-level response | "Tell me more about that" / "Can you give me an example?" |

---

## 2. Psychological Models

### Big Five (OCEAN) Personality Model

We use the Five-Factor Model (Costa & McCrae, 1992) as the foundation for persona behavior. Each trait is scored 0-100 and maps to specific discussion behaviors:

#### Openness to Experience (O)
- **High (70-100):** Curious about new concepts, willing to try unfamiliar products, generates creative responses, uses metaphors and analogies
- **Low (0-30):** Prefers familiar products, skeptical of novelty, pragmatic responses, wants proven track record
- **Impact on discussion:** High-O personas engage enthusiastically with new concepts; Low-O require more convincing

#### Conscientiousness (C)
- **High (70-100):** Analytical, considers pros/cons systematically, mentions research/reviews, budget-conscious, plans purchases
- **Low (0-30):** Impulsive, goes with gut feeling, doesn't research, spontaneous purchaser
- **Impact on discussion:** High-C personas give structured, detailed responses; Low-C give brief, emotional ones

#### Extraversion (E)
- **High (70-100):** Speaks frequently, speaks early, longer responses, comfortable disagreeing, uses emphatic language
- **Low (0-30):** Speaks when prompted, brief responses, waits to hear others first, hedges opinions
- **Impact on discussion:** High-E personas dominate airtime; Low-E need to be drawn out by moderator

#### Agreeableness (A)
- **High (70-100):** Seeks consensus, builds on others' ideas, softens criticism, conforms under pressure
- **Low (0-30):** Willing to disagree, challenges other opinions, resists conformity, blunt
- **Impact on discussion:** High-A personas create echo chambers; Low-A personas create productive tension

#### Neuroticism (N)
- **High (70-100):** Risk-averse, mentions potential problems, anxious about new products, wants guarantees/safety
- **Low (0-30):** Comfortable with risk, optimistic about new products, doesn't dwell on negatives
- **Impact on discussion:** High-N personas raise objections; Low-N personas are boosters

### Population Distribution

OCEAN traits follow roughly normal distributions in the general population:
- Mean: ~50 for all traits
- SD: ~15-20
- We sample from these distributions, adjusted by demographic correlations:
  - Women slightly higher A and N (meta-analyses)
  - Age correlates with higher A and C (maturity effect)
  - Education correlates with higher O
  - Income weakly correlates with higher C and E

### Behavioral Rules Engine

```
IF persona.extraversion > 70 AND question.type == "open_ended":
    speak_probability += 0.3
    
IF persona.agreeableness > 70 AND group_consensus == "positive":
    sentiment_shift += 0.2 toward positive
    
IF persona.conscientiousness > 70:
    response includes at least one comparison or tradeoff analysis
    
IF persona.neuroticism > 70 AND topic == "new_product":
    response includes at least one risk or concern
    
IF persona.openness < 30 AND concept == "novel":
    initial_reaction skews skeptical
```

---

## 3. Group Dynamics Models

### Asch Conformity Effect (1956)

**Finding:** When 3+ confederates give an obviously wrong answer, 37% of subjects conform.

**Our implementation:**
- When 3+ personas express the same sentiment, remaining personas face conformity pressure
- Pressure strength = (n_agreeing / total) × persona.agreeableness
- Personas with low agreeableness resist; high agreeableness conform
- Conformity manifests as: softened language, hedged disagreement, shifted opinion

### Moscovici's Minority Influence (1969)

**Finding:** A consistent minority can shift majority opinion over time, especially if they appear confident and principled.

**Our implementation:**
- If a persona with high conviction and low agreeableness consistently disagrees, other personas gradually consider their position
- Requires 2+ rounds of consistent dissent
- More effective when dissenter provides reasoning (not just "I disagree")

### Group Polarization (Stoner, 1961)

**Finding:** Group discussion tends to push attitudes toward more extreme versions of pre-existing tendencies.

**Our implementation:**
- Track group sentiment direction over the discussion
- Gradually amplify the majority direction
- Risk-averse groups become more cautious; risk-accepting groups become more bold
- Provides useful data: "The group started moderately positive and became enthusiastically positive by the end — this suggests word-of-mouth amplification potential"

### Social Desirability Bias (Crowne & Marlowe, 1960)

**Finding:** People modify their responses to present themselves favorably to others.

**Our implementation:**
- High agreeableness + group setting = softened criticism
- "I hate it" becomes "It's not really for me, but I can see the appeal"
- This bias is INTENTIONAL — it matches real focus group behavior
- Analysis engine accounts for this: raw sentiment vs. adjusted sentiment

### Dominant Speaker Effect

**Finding:** In unmoderated groups, 1-2 people dominate 60-70% of speaking time.

**Our implementation:**
- High-E personas speak more often and longer
- Without moderation, this creates realistic imbalance
- Moderator agent counteracts by redirecting to quiet personas
- Analysis tracks speaking distribution as a quality metric

---

## 4. Validation Framework

### Synthetic-Organic Parity (SOP) Methodology

Our credibility depends on proving synthetic results match real human results.

#### Validation Study Design

For each validation:
1. **Identical stimulus:** Same product concept, same questions
2. **Real panel:** 50-200 respondents via Prolific, matched to the same demographics
3. **Synthetic panel:** 50-200 AI personas with matching demographics
4. **Comparison metrics:**

| Metric | Measurement | Target |
|--------|------------|--------|
| **Purchase Intent Correlation** | Pearson r between real and synthetic distributions | r > 0.85 |
| **Theme Overlap** | % of real themes also found in synthetic | > 80% |
| **Sentiment Direction** | Agreement on positive/negative per topic | > 85% |
| **Feature Priority Match** | Rank correlation of feature importance | ρ > 0.80 |
| **Extreme Response Match** | Correlation of strong agree/disagree rates | r > 0.75 |

#### Validation Schedule
- **Pre-launch:** 3-5 parallel studies to establish baseline SOP
- **Monthly:** 1-2 validation studies to maintain accuracy claims
- **Per-category:** When entering a new category, run validation first
- **Published:** All validation data publicly available (transparency = trust)

### Known Limitations (Stated Upfront)

1. **Cannot replicate physical reactions** — No body language, facial expressions, product touch/taste
2. **Experiential questions are less accurate** — "Tell me about a time when..." generates stereotypical rather than authentic stories
3. **Novel/unprecedented products** — If nothing like it exists, personas lack behavioral grounding
4. **Cultural nuance** — Cross-cultural research requires culture-specific calibration
5. **Not a replacement for all qualitative research** — Best for concept testing, message testing, feature prioritization. Not for ethnography, usability testing, or deep psychological exploration

### Positioning: Augment, Don't Replace

The most credible positioning (and the one NielsenIQ recommends):
- Use synthetic BEFORE real research to screen concepts and refine stimuli
- Use synthetic for rapid iteration during development
- Use real humans for final validation and go/no-go decisions
- Synthetic reduces the NUMBER of real studies needed, not eliminates them

This positioning:
- Disarms skeptics ("we agree real humans matter")
- Creates a wedge into traditional research budgets
- Builds trust that leads to eventual primary use

---

## 5. Ethical Considerations

### Transparency
- All reports clearly labeled as synthetic research
- Methodology section in every report explains how personas were generated
- No claim of "talking to real people" — ever

### Bias Acknowledgment
- LLMs have known biases (political, cultural, demographic)
- We document known bias patterns and their mitigation
- Validation against real humans is the primary bias check

### Data Privacy
- No real individual's data is used to create specific personas
- All data sources are aggregate/population-level
- Client concepts are not used for model training
- Option for full data deletion after study

### Responsible Use
- Not suitable for: medical decisions, legal proceedings, regulatory compliance
- Suitable for: product development, marketing research, concept screening, message testing
- Clear terms of service defining appropriate use cases
