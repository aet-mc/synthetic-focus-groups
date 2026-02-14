# Persona Engine: Deep Research

## Table of Contents
1. [The Core Problem: Why Persona Quality Is Everything](#1-the-core-problem)
2. [Academic State of the Art](#2-academic-state-of-the-art)
3. [The #1 Failure Mode: Mean Regression & Homogeneity](#3-the-1-failure-mode)
4. [Psychological Models for Consumer Behavior](#4-psychological-models)
5. [Data Sources for Grounding](#5-data-sources)
6. [Persona Architecture Design](#6-persona-architecture)
7. [Consumer Decision-Making Models](#7-consumer-decision-making)
8. [Variance & Diversity Engineering](#8-variance--diversity)
9. [Known Limitations & Mitigations](#9-known-limitations)
10. [Implementation Recommendations](#10-implementation-recommendations)

---

## 1. The Core Problem

The persona engine is the single most important component. If personas are shallow, stereotypical, or homogeneous, the entire platform produces garbage. Every competitor fails here.

**The fundamental tension:** LLMs are trained to predict the most likely next token. The most likely response is, by definition, the average response. But real consumers are NOT average — they're diverse, contradictory, emotional, and surprising. A focus group with 8 average people produces no insights.

**What we need:** Personas that are individually coherent (this specific person makes sense) AND collectively diverse (the 8 people in the room have genuinely different perspectives, not slightly different phrasings of the same opinion).

---

## 2. Academic State of the Art

### Key Papers (2024-2026)

#### PolyPersona (Dec 2025) — Persona-Grounded LLM for Synthetic Survey Responses
- **Authors:** Dash, Karri, Vurity, Datla, Ahmad, Rafi, Tangudu
- **Key finding:** Persona-conditioned fine-tuning on compact models (TinyLlama 1.1B, Phi-2) achieves performance on par with 7B-8B baselines
- **Dataset:** 3,568 responses spanning 10 domains and 433 unique personas
- **Approach:** Dialogue-formatted data pipeline preserving persona cues, LoRA adapters, 4-bit quantization
- **Implication for us:** We don't need to fine-tune — prompting with rich persona context works if grounded properly. But their evaluation framework (BLEU, ROUGE, BERTScore + survey-specific metrics) is worth adopting

#### PersonaCite (Jan 2026) — VoC-Grounded Interviewable Agentic Synthetic AI Personas
- **Authors:** CHI 2026 submission, 14 industry experts
- **Key finding:** Personas that retrieve actual voice-of-customer artifacts during conversation and constrain responses to retrieved evidence dramatically outperform prompt-based roleplaying
- **Key innovation:** "Persona Provenance Cards" — documentation pattern for responsible AI persona use
- **Critical insight:** "LLM-based personas are often weakly grounded, inconsistent and prone to hallucinating plausible yet unverifiable user opinions"
- **Implication for us:** RAG-based persona grounding (retrieving real consumer review data, social media posts, forum discussions) could dramatically improve quality. Consider building a consumer opinion database

#### "LLM Generated Persona is a Promise with a Catch" (May 2025) — Li, Chen, Namkoong, Peng
- **THE MOST IMPORTANT PAPER FOR US**
- **Key finding:** The more "realistic" and detailed you make AI personas with LLM-generated backstories, the WORSE they perform. Census-style stripped-down profiles consistently outperform richly imagined ones
- **Evidence:** In U.S. election simulations (2016-2024), personas with LLM-generated narratives predicted clean Democratic sweeps across all years. Census-derived "Meta" personas were closest to actual results
- **Root cause:** LLM-generated persona details are systematically biased toward progressive, optimistic, educated, urban profiles. The more you let the LLM create, the more it regresses to its training data distribution, not real population distribution
- **1M persona dataset:** Open-sourced for further research
- **Critical implication for us:** 
  - DO ground in census demographic data
  - DO NOT let the LLM generate backstories freely
  - Use structured templates with real statistical distributions
  - Only use LLM for generating RESPONSES, not for generating PERSONAS

#### Mixture-of-Personas (MoP) (Apr 2025) — Hierarchical persona structure
- **Approach:** Hierarchical structure enabling effective representation of diversity and complexity in population-level behaviors
- **Key benefit:** Simultaneously mitigates biases in LLM-generated responses
- **Implication for us:** Consider a hierarchical persona structure — broad segments first, then individual variation within segments

#### Persona Hub (Tencent AI Lab, June 2024) — 1 Billion Personas
- **Approach:** Curated 1 billion diverse personas automatically from web data
- **Use case:** Training data synthesis, not market research
- **Relevance:** Shows that persona diversity at scale is achievable, but their personas are for data generation, not behavioral simulation

#### Gender Bias in Synthetic Personas (Oct 2025) — ScienceDirect
- **Finding:** LLM-generated personas encode and perform gendering — systematic gender stereotypes
- **Mitigation:** Critical prompting strategies can increase diversity and decrease bias
- **Implication for us:** Must actively test for and counteract demographic stereotyping in personas

---

## 3. The #1 Failure Mode: Mean Regression & Homogeneity

This is the biggest unsolved problem in synthetic persona research. Every paper flags it.

### What Happens

When you prompt an LLM: "You are Sarah, 34, suburban Denver, $82K income, marketing coordinator. What do you think of this organic snack?"

The LLM generates what it thinks is the MOST LIKELY response from this demographic — which is essentially the average of its training data for similar contexts. After 8 personas, you get 8 slightly different phrasings of the same mild enthusiasm.

**Real focus groups:** 2 people love it, 1 hates it, 3 are indifferent, 1 has a wildly creative suggestion, 1 is distracted by the packaging

**Naive LLM personas:** 7 think it's "interesting" with "room for improvement" and 1 is slightly more enthusiastic

### Why It Happens

1. **LLM training objective** — Predicts most probable next token. Modal response = average response
2. **Social desirability in training data** — LLMs are RLHF'd to be agreeable and helpful. This bleeds into persona responses
3. **Persona description homogenizes** — "suburban mom who cares about health" activates the same neural pathways regardless of other traits
4. **Lack of genuine internal state** — Real people have moods, recent experiences, specific associations. LLM personas don't

### Proven Mitigations (From Literature)

**1. Census-Anchored Demographics (Li et al. 2025)**
- Generate persona demographics from actual census joint distributions
- NOT "make up a realistic person" — instead sample from real data tables
- Example: If 23% of women 30-40 in suburban Denver make $70-90K with bachelor's degrees — sample exactly that proportion
- This prevents the LLM from overrepresenting its training data distribution

**2. Personality Extremes (Our Innovation)**
- Don't sample OCEAN from normal distribution with mean 50
- Deliberately include extreme profiles: very low agreeableness (the contrarian), very high neuroticism (the worrier), very low conscientiousness (the impulse buyer)
- Real focus group recruiters deliberately recruit for diversity of opinion — we should too
- Use stratified sampling: ensure at least 1 persona in each personality quadrant

**3. Pre-seeded Opinions (PolyPersona Approach)**
- Before the discussion, each persona privately forms an opinion
- This opinion is generated independently, without group context
- Opinions should span the full range: 1-2 strongly positive, 1-2 strongly negative, 3-4 mixed
- This prevents the convergence that happens when all personas hear the same prompt simultaneously

**4. Voice-of-Customer Grounding (PersonaCite Approach)**
- Instead of letting LLMs imagine what consumers think, retrieve ACTUAL consumer opinions
- Source: Product reviews, Reddit discussions, social media posts, forum threads
- Each persona's responses are influenced by real consumer language and concerns
- This creates genuine diversity because real consumers ARE diverse

**5. Temperature + Sampling Strategies**
- Higher temperature (0.9-1.1) for persona responses (more creative, less predictable)
- Lower temperature (0.3-0.5) for analysis (more reliable, more consistent)
- Top-p sampling with p=0.95 to allow occasional unexpected responses
- Consider per-persona temperature: high-openness personas get higher temp

**6. Contradiction Injection**
- Deliberately program 1-2 personas to disagree with the majority
- Not random — grounded in their personality (low agreeableness + high conviction)
- This forces the discussion to surface counter-arguments
- Real focus group recruiters do exactly this — they recruit "contrarians" on purpose

---

## 4. Psychological Models for Consumer Behavior

### Big Five (OCEAN) — Primary Model

The Five-Factor Model (Costa & McCrae, 1992) is the gold standard. 40+ years of validation data, cross-cultural consistency, well-established demographic correlations.

#### Age-Personality Correlations (Roberts et al. 2006, N=50,120)
Based on meta-analysis of 113 longitudinal samples:

| Trait | Age Trend | Consumer Behavior Impact |
|-------|----------|------------------------|
| **Extraversion** | Declines with age (especially after 50s) | Younger personas more vocal, more influenced by social proof |
| **Agreeableness** | Increases with age (gradual, lifelong) | Older personas more agreeable in groups, more brand loyal |
| **Conscientiousness** | Peaks in middle age (40-60), declines slightly after | Middle-aged most methodical about purchases, do most research |
| **Neuroticism** | Slight decline with age (inconsistent across studies) | Younger personas more anxious about new products |
| **Openness** | Stable 20s-50s, declines after mid-50s | Younger/middle personas most receptive to innovation |

#### Gender-Personality Correlations (Meta-analyses)
- Women: Slightly higher Agreeableness (+0.3 SD) and Neuroticism (+0.3 SD)
- Men: Slightly higher Extraversion (+0.15 SD)
- Openness and Conscientiousness: Minimal gender differences
- **CAUTION:** These are small effect sizes. Individual variation far exceeds group means. Do NOT make all female personas agreeable.

#### OCEAN → Consumer Behavior Mapping (Mnemonic.ai, 2025)

**Openness to Experience:**
- High O: Early adopters, tries new brands, attracted to novel/artisan products, willing to pay premium for unique
- Low O: Brand loyal, prefers familiar, suspicious of "new and improved," values tradition
- *Purchase trigger:* High O = novelty, uniqueness. Low O = proven track record

**Conscientiousness:**
- High C: Reads labels, compares prices, checks reviews, plans purchases, uses shopping lists
- Low C: Impulse buys, grabs what looks good, doesn't compare, convenience-driven
- *Purchase trigger:* High C = value/quality analysis. Low C = ease and instant gratification

**Extraversion:**
- High E: Influenced by social proof, shares purchases socially, brand-as-identity, trend-following
- Low E: Private about purchases, less influenced by trends, practical utility focus
- *Purchase trigger:* High E = social validation. Low E = personal utility

**Agreeableness:**
- High A: Influenced by group opinions, avoids conflict brands, prefers "feel-good" brands, ethical consumption
- Low A: Contrarian, willing to buy controversial brands, price > ethics, resistant to peer pressure
- *Purchase trigger:* High A = social approval. Low A = personal benefit

**Neuroticism:**
- High N: Risk-averse, fears buyer's remorse, wants guarantees/returns, over-researches
- Low N: Comfortable with risk, doesn't worry about mistakes, quick decisions
- *Purchase trigger:* High N = safety/guarantee. Low N = opportunity/upside

### VALS (Values and Lifestyles) — Secondary Model

Strategic Business Insights' VALS segments US consumers into 8 types based on primary motivation and resources:

| Type | Motivation | Resources | % of US | Behavior |
|------|-----------|-----------|---------|----------|
| Innovators | All three | High | 8% | Sophisticated, take-charge, success-oriented |
| Thinkers | Ideals | High | 11% | Mature, reflective, well-educated, informed |
| Achievers | Achievement | High | 13% | Goal-oriented, brand-conscious, conventional |
| Experiencers | Self-expression | High | 12% | Young, enthusiastic, impulsive, trend-following |
| Believers | Ideals | Low | 16% | Conservative, conventional, loyal to established brands |
| Strivers | Achievement | Low | 13% | Trendy, fun-loving, concerned with status |
| Makers | Self-expression | Low | 12% | Practical, self-sufficient, focused on functionality |
| Survivors | None dominant | Lowest | 14% | Cautious, risk-averse, brand-loyal, budget-driven |

**How to integrate:** Assign VALS type as a secondary psychographic layer. It adds motivational context that OCEAN alone doesn't capture. A high-E, high-O person could be an "Experiencer" (self-expression motivated) or an "Innovator" (multi-motivated) — the VALS type disambiguates their purchase drivers.

### Schwartz Value Survey — Consideration

Research (MDPI, 2019) found that Schwartz values outperformed Big Five in predicting consumer preferences in most product categories. The 10 value types:

1. Power (social status, dominance)
2. Achievement (personal success)
3. Hedonism (pleasure, sensuous gratification)
4. Stimulation (excitement, novelty)
5. Self-Direction (independent thought, action)
6. Universalism (tolerance, social justice, nature)
7. Benevolence (welfare of close others)
8. Tradition (cultural/religious customs)
9. Conformity (restraint from harming others)
10. Security (safety, stability)

**Recommendation:** Consider adding 2-3 dominant Schwartz values per persona for purchase motivation grounding. E.g., a persona with Security + Tradition values will approach a new tech product very differently than one with Stimulation + Self-Direction.

---

## 5. Data Sources for Grounding

### Tier 1: Free Government Data (Must Use)

#### US Census Bureau API
- **URL:** api.census.gov
- **Key datasets:**
  - American Community Survey (ACS) 5-year estimates
  - Decennial Census
  - Current Population Survey (CPS)
- **Variables available:** Age, sex, race, income, education, occupation, household type, geographic location, commute, internet access, language
- **Access:** Free API key, no rate limits for reasonable use
- **Python library:** `census` or `pytidycensus`
- **Format:** JSON responses with variable codes → need mapping tables
- **Critical use:** Joint distributions (e.g., probability of a 35-year-old woman in Denver making $80-90K with a bachelor's degree in marketing)

#### Bureau of Labor Statistics (BLS) Consumer Expenditure Survey
- **URL:** bls.gov/cex/
- **Key data:** Detailed spending by category broken down by:
  - Income quintile
  - Age of reference person
  - Family size
  - Region
  - Education
  - Race
  - Housing tenure (own/rent)
- **Categories:** Food, housing, transportation, healthcare, entertainment, personal care, education, apparel, cash contributions
- **Format:** Published tables (CSV/XLSX) + public-use microdata (SAS/CSV)
- **Critical use:** Ground persona spending behavior in real data. "A household making $80K in the Mountain West spends $X on groceries, $Y on dining out, $Z on healthcare"

#### BLS American Time Use Survey (ATUS)
- **URL:** bls.gov/tus/
- **Data:** How Americans spend their time by demographic
- **Use:** Media consumption, leisure activities, household work — informs persona lifestyle

### Tier 2: Free Academic/Research Data

#### IPIP (International Personality Item Pool)
- **URL:** ipip.ori.org
- **Data:** Free personality measurement items, norms, and scoring guides
- **Use:** OCEAN norms by age/gender for realistic personality distribution sampling

#### General Social Survey (GSS)
- **URL:** gss.norc.org
- **Data:** Attitudes, behaviors, demographics of US adults since 1972
- **Use:** Ground persona opinions on social issues, consumer confidence, trust in institutions

#### Pew Research Center
- **URL:** pewresearch.org/datasets
- **Data:** Public opinion surveys on politics, media, technology, religion, social trends
- **Use:** Attitudinal grounding — what do real people in each demographic actually think about current issues?

### Tier 3: Consumer Voice Data (For RAG Grounding)

#### Reddit (Public)
- Category-specific subreddits (r/BuyItForLife, r/frugal, r/technology, r/skincare, etc.)
- Real consumer opinions, unfiltered
- Use: Build a searchable corpus of real consumer statements by category
- Access: Reddit API or pushshift archives

#### Amazon/Product Reviews (Public)
- Millions of detailed product reviews with demographic hints
- Real purchase decisions with reasoning
- Datasets available on Kaggle and academic archives

#### Trustpilot/G2/Capterra Reviews
- Service and software reviews with more B2B perspective
- Public APIs available

### Tier 4: Paid Data (Later Stage)

#### Prolific
- Real human participants for validation studies
- $2-5 per response, highly targeted demographics
- Use: Validation benchmark, not primary data

#### Simmons/MRI Consumer Surveys
- Gold standard for consumer behavior data
- Extremely expensive ($50K+/year)
- Future consideration for enterprise tier

---

## 6. Persona Architecture Design

### The Three-Layer Approach

Based on the research, the optimal persona architecture has three distinct layers:

```
Layer 1: DEMOGRAPHIC SHELL (Census-anchored, deterministic)
├── Age, gender, income, education, location, occupation, household
├── Sampled from ACTUAL census joint distributions
├── NOT LLM-generated — pulled from real data tables
└── This is the "anchor" that prevents bias drift

Layer 2: PSYCHOGRAPHIC CORE (Model-driven, semi-random)
├── OCEAN scores (sampled from age/gender-adjusted distributions)
├── VALS type (derived from OCEAN + income + education)
├── Schwartz values (2-3 dominant values, sampled)
├── Consumer behavior tendencies (derived from psychographic scores)
└── Stratified sampling ensures diversity across the group

Layer 3: CONTEXTUAL SURFACE (LLM-generated, constrained)
├── Category-specific knowledge and experience
├── Communication style (derived from education + extraversion)
├── Recent relevant experiences (LLM fills in, but constrained by Layer 1+2)
├── Brand awareness/attitudes (grounded in market share data where available)
└── This is the ONLY layer where the LLM creates freely
```

### Critical Design Rules (From Literature)

1. **Never let the LLM generate demographics.** Always sample from census data. (Li et al. 2025)
2. **Pre-seed opinions before group discussion.** Each persona privately decides their stance. (PolyPersona)
3. **Include personality extremes.** At least 1 persona with < 25th percentile on agreeableness. At least 1 with > 90th percentile on openness. Etc.
4. **Assign speaking roles deliberately.** 1-2 dominant speakers (high E), 2-3 moderate, 2-3 quiet (low E). The moderator must draw out the quiet ones.
5. **Ground in real consumer voice where possible.** Retrieve real reviews/opinions and use them to shape persona language. (PersonaCite)
6. **Track internal state across the discussion.** Personas should change their mind based on what they hear — but the amount of change should be governed by their personality (high A = more change, low A = less change).

### Persona Generation Pipeline

```
Input: Client's target demographic criteria
  ↓
Step 1: CENSUS SAMPLING
  - Query Census API for joint distribution matching criteria
  - Sample N personas weighted by real population distribution
  - Include deliberate diversity (not all from the modal demographic)
  ↓
Step 2: PERSONALITY ASSIGNMENT
  - For each persona, sample OCEAN from age/gender-adjusted norms
  - Apply stratified sampling: ensure coverage of personality space
  - Derive VALS type and Schwartz values from OCEAN + demographics
  ↓
Step 3: CONSUMER BEHAVIOR MODELING
  - Look up BLS spending patterns for this demographic cell
  - Derive price sensitivity, brand loyalty, research tendency from OCEAN
  - Assign category engagement level (heavy/moderate/light user)
  ↓
Step 4: OPINION PRE-SEEDING
  - Present the product concept to each persona INDIVIDUALLY (no group context)
  - LLM generates initial reaction anchored in persona profile
  - Apply diversity check: if > 70% are positive, re-generate 1-2 as negative
  ↓
Step 5: VOICE GROUNDING (Optional, improves quality)
  - Search consumer review corpus for similar demographic + category
  - Attach 3-5 real consumer quotes as "things people like you have said"
  - These serve as RAG context during discussion
  ↓
Step 6: COMMUNICATION STYLE DERIVATION
  - Education → vocabulary level
  - Extraversion → response length, frequency
  - Agreeableness → hedging language, qualifier use
  - Conscientiousness → structure and detail level
  - Compile into a "voice card" used in every LLM call
  ↓
Output: Complete persona profile ready for discussion simulation
```

---

## 7. Consumer Decision-Making Models

Understanding HOW consumers decide is critical for realistic discussion simulation.

### Elaboration Likelihood Model (Petty & Cacioppo, 1986)

Two routes to persuasion:
- **Central route:** High involvement → careful evaluation of arguments, features, evidence
- **Peripheral route:** Low involvement → influenced by cues (brand, celebrity, design, social proof)

**Persona application:**
- High Conscientiousness + high involvement category → Central route processing → detailed analytical responses
- Low Conscientiousness + low involvement category → Peripheral route → "it looks nice" / "my friend has one"
- This determines the DEPTH of a persona's engagement with the product concept

### Consumer Decision Journey (McKinsey)

Not a linear funnel but a loop:
1. **Initial consideration set** — brands/products already known
2. **Active evaluation** — research, comparison, review reading
3. **Moment of purchase** — final trigger
4. **Post-purchase experience** — satisfaction, sharing, loyalty loop

**Persona application:**
- Where is this persona in the journey? (Unaware → Aware → Considering → Decided)
- High-C personas are more likely to be in "active evaluation" mode
- Low-C personas might skip straight to "moment of purchase" (impulse)

### Loss Aversion & Prospect Theory (Kahneman & Tversky)

People feel losses 2x more intensely than equivalent gains.

**Persona application:**
- High Neuroticism personas are MORE loss-averse
- They focus on "what could go wrong" more than "what I'd gain"
- Frame product negatively → High-N personas react strongly
- Frame product as preventing a loss → High-N personas are more interested

---

## 8. Variance & Diversity Engineering

### The Diversity Checklist

For every focus group of 8 personas, verify:

- [ ] At least 2 personality "types" strongly represented (e.g., analytical + impulsive)
- [ ] At least 1 persona with very low agreeableness (the contrarian)
- [ ] At least 1 persona with very high neuroticism (the worrier)  
- [ ] At least 1 persona with very high openness (the enthusiast)
- [ ] At least 1 persona with very low openness (the skeptic)
- [ ] Pre-seeded opinions span the full range (not all positive or all negative)
- [ ] Income distribution includes at least 2 tiers
- [ ] Age range spans at least 15 years
- [ ] Category engagement varies (heavy user + light user + non-user)
- [ ] Communication styles vary (verbose + terse + analytical + emotional)

### Forced Diversity Mechanisms

1. **Opinion quota:** Maximum 5 of 8 personas can have the same initial sentiment (positive/negative/neutral)
2. **Devil's advocate assignment:** 1 persona is explicitly assigned to challenge the majority view
3. **Non-user inclusion:** At least 1 persona who doesn't currently use this product category — they bring outside perspective
4. **Outlier injection:** 1 persona with an unusual combination (e.g., high income + low brand loyalty, or young + traditional values)

### Measuring Diversity

After generating personas, calculate:
- **Opinion entropy:** Shannon entropy of pre-seeded opinions. Target: > 1.5 bits (out of max ~2.1 for 5-point scale)
- **Personality spread:** Standard deviation of each OCEAN trait across personas. Target: > 15 for each trait
- **Response uniqueness:** After first round of discussion, cosine similarity between each pair of responses. Target: average < 0.6

---

## 9. Known Limitations & Mitigations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **LLMs skew progressive/optimistic** | Personas underrepresent conservative, skeptical, or negative viewpoints | Census-anchored demographics + forced diversity + opinion pre-seeding |
| **Cannot simulate physical reactions** | No body language, taste, touch, facial expressions | Position as complementary to physical testing, not replacement |
| **Experiential questions are weak** | "Tell me about a time when..." generates stereotypical stories, not authentic memories | Use experiential questions sparingly; focus on attitudinal and intentional questions |
| **Cultural nuance is hard** | Cross-cultural personas may flatten to Western stereotypes | Per-culture calibration required; start with US market, expand carefully |
| **LLMs are agreeable by nature** | RLHF training makes personas too polite and accommodating | Explicit "be honest, not polite" instructions; low-agreeableness personas; devil's advocate role |
| **Reproducibility varies** | Same prompt → slightly different results each run | Use seed values where possible; run multiple iterations and report ranges |
| **Novel/unprecedented products** | No training data for truly new categories | Acknowledge explicitly in report; use analogous category behavior as proxy |
| **Joint demographic distributions are complex** | Hard to sample realistic combinations (not just marginals) | Use census microdata (PUMS) which provides actual household-level records |

---

## 10. Implementation Recommendations

### Phase 1 (MVP — Week 1)

1. **Hardcode 8 diverse persona profiles** using manually researched census data
2. **Implement OCEAN scoring** with stratified sampling (ensure personality diversity)
3. **Pre-seed opinions** via independent LLM calls (1 per persona, no group context)
4. **Force diversity:** Manually verify the 8 personas span the checklist above
5. **Use high temperature** (0.9) for persona responses, low (0.3) for analysis
6. **Test:** Run the same study 3 times, measure response variance. If too similar → increase personality extremes

### Phase 2 (Census Integration — Week 2)

1. **Hook up Census API** — American Community Survey 5-year estimates
2. **Build joint distribution sampler** — given client criteria, sample realistic demographic combinations
3. **Integrate BLS spending data** — automatically look up spending patterns per persona
4. **Implement OCEAN → consumer behavior mapping** — personality directly influences purchase style

### Phase 3 (RAG Grounding — Week 3-4)

1. **Build consumer voice corpus** — scrape/index Reddit + review data by category
2. **Implement retrieval** — for each persona + category, retrieve 3-5 relevant real consumer quotes
3. **Use retrieved quotes as context** in persona LLM calls (not as direct output, but as behavioral influence)
4. **Compare quality** — studies with RAG grounding vs. without, measure diversity and realism

### Phase 4 (Validation — Week 4-5)

1. **Run parallel Prolific studies** — same concept, same demographics, real humans
2. **Measure SOP** — Synthetic-Organic Parity across multiple metrics
3. **Iterate on persona engine** until SOP > 80% on key metrics
4. **Publish validation data** — this IS the product's credibility

---

## References & Key Papers

1. Li, Chen, Namkoong, Peng (2025). "LLM Generated Persona is a Promise with a Catch." — **MUST READ. The grounding paper.**
2. Dash et al. (2025). "PolyPersona: Persona-Grounded LLM for Synthetic Survey Responses." arXiv:2512.14562
3. PersonaCite (2026). "VoC-Grounded Interviewable Agentic Synthetic AI Personas." CHI 2026.
4. Chan et al. (2024). "Scaling Synthetic Data Creation with 1,000,000,000 Personas." Tencent AI Lab.
5. Costa & McCrae (1992). "Revised NEO Personality Inventory (NEO-PI-R)."
6. Roberts, Walton, & Viechtbauer (2006). "Patterns of mean-level change in personality traits across the life course."
7. Lucas & Donnellan (2009). "Age Differences in the Big Five Across the Life Span." PMC2562318.
8. Petty & Cacioppo (1986). "Elaboration Likelihood Model."
9. Schwartz (1992). "Universals in the content and structure of values."
10. NielsenIQ (2024). "The Rise of Synthetic Respondents in Market Research."
11. Qualtrics (2025). "2025 Market Research Trends Report."
12. Saucery.ai (2025). "The Science Behind AI Personas: Research Accuracy."
13. MoP (2025). "Mixture-of-Personas Language Models for Population Simulation." arXiv:2504.05019
