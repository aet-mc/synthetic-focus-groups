from __future__ import annotations

THEME_CODING_PROMPT = """
Analyze this focus group transcript.

For each participant message, assign 1-3 topic codes.
Return strict JSON as:
[
  {{"message_index": 0, "codes": ["price", "trust"]}}
]

Transcript:
{transcript}
""".strip()


THEME_CLUSTERING_PROMPT = """
Given these topic codes from a focus group, cluster them into {max_themes} major themes.
Return strict JSON as:
[
  {{
    "name": "Price Sensitivity",
    "description": "One to two sentence theme description.",
    "codes": ["price", "value", "cost"]
  }}
]

Codes and counts:
{codes}
""".strip()


SENTIMENT_BATCH_PROMPT = """
Score each of these focus group responses from -1 (very negative) to 1 (very positive).
Return strict JSON array of floats matching input order.

Responses:
{responses}
""".strip()


CONCEPT_SCORE_PROMPT = """
Score this participant's reaction to the product concept on 6 INDEPENDENT metrics (1-5 scale).

Product Concept: {concept_description}

Each metric measures something DIFFERENT — do NOT give the same score across all metrics:
- purchase_intent: Would they BUY this? (1=no way, 5=eager to buy)
- overall_appeal: Do they find the IDEA appealing, even if they wouldn't buy? (1=repulsed, 5=love it)
- uniqueness: How NOVEL is this vs alternatives? Even haters can recognize novelty. (1=copycat, 5=never seen before)
- relevance: How relevant to THEIR life/needs? (1=irrelevant, 5=solves their problem)
- believability: Do they believe it CAN work? (1=impossible, 5=fully convinced)
- value_perception: Is the price fair? (1=ripoff, 5=bargain)

Return strict JSON object with those 6 keys only. Use decimals (e.g., 3.5).

Persona summary:
{persona}

Participant statements:
{statements}
""".strip()


CONCEPT_SCORE_BATCH_PROMPT = """
You are scoring focus group participants' reactions to a product concept.

Product Concept: {concept_description}

Score each participant on these 6 INDEPENDENT metrics (1-5 scale). Each metric measures something DIFFERENT — score them separately, not as one overall rating.

METRIC DEFINITIONS (score each independently):
1. purchase_intent: Would they personally BUY this? Based on explicit statements about buying, trying, or paying.
   (1=explicitly refused, 2=unlikely, 3=maybe, 4=probably, 5=eager to buy)

2. overall_appeal: Do they find the IDEA interesting or attractive, regardless of whether they'd buy it?
   (1=repulsed, 2=uninterested, 3=somewhat interesting, 4=quite appealing, 5=love the concept)

3. uniqueness: How novel/different is this compared to what exists? This is OBJECTIVE — even someone who hates a product can recognize it's unlike anything else.
   (1=copycat/exists already, 2=minor twist, 3=somewhat different, 4=very novel, 5=never seen anything like it)

4. relevance: How relevant is this to THEIR specific life, needs, or situation?
   (1=zero relevance, 2=tangential, 3=somewhat relevant, 4=quite relevant, 5=solves a real problem they have)

5. believability: Do they believe the product can DELIVER on its promises? Based on trust signals, skepticism, or credulity expressed.
   (1=impossible/scam, 2=highly doubtful, 3=plausible, 4=likely works, 5=fully convinced)

6. value_perception: Is the price fair for what's offered? Based on comments about pricing, cost, worth.
   (1=ripoff, 2=overpriced, 3=fair, 4=good deal, 5=bargain)

CRITICAL RULES:
- Metrics are INDEPENDENT. A participant can score 1 on purchase_intent but 5 on uniqueness (e.g., "I'd never buy this but it's unlike anything I've seen").
- A participant can score 4 on appeal but 2 on value (e.g., "love the idea but way too expensive").
- Do NOT give the same score across all metrics for a participant. If you find yourself doing that, re-read their statements and look for nuance.
- Base scores on what they SAID, not assumed overall sentiment.

Return strict JSON keyed by participant ID.
Example where someone dislikes a product but recognizes its novelty:
{{"p1": {{"purchase_intent": 1.5, "overall_appeal": 2.5, "uniqueness": 4.5, "relevance": 2.0, "believability": 3.0, "value_perception": 2.0}}}}

Participants and their statements:
{participants_block}
""".strip()


EXECUTIVE_SUMMARY_PROMPT = """
Write a concise 3-5 sentence executive summary of these focus group findings.
Use plain language and prioritize the strongest takeaway.

Concept scores:
{concept_scores}

Top themes:
{themes}

Sentiment timeline:
{sentiment}

Discussion config:
{config}
""".strip()


RECOMMENDATION_PROMPT = """
Based on these focus group results, provide a clear GO/ITERATE/NO-GO recommendation and a confidence level.
Return strict JSON:
{{"recommendation": "...", "confidence_level": "high|medium|low"}}

Concept scores:
{concept_scores}

Themes:
{themes}

Top concerns:
{concerns}
""".strip()
