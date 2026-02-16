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
Based on this participant's statements throughout the discussion, score their reaction on these 1-5 metrics:
- purchase_intent
- overall_appeal
- uniqueness
- relevance
- believability
- value_perception

Return strict JSON object with those keys only.

Persona summary:
{persona}

Participant statements:
{statements}
""".strip()


CONCEPT_SCORE_BATCH_PROMPT = """
You are scoring focus group participants' reactions to a product concept.

For each participant, rate their reaction on these 6 metrics using a 1-5 scale:
- purchase_intent: How likely are they to buy? (1=definitely not, 2=probably not, 3=might or might not, 4=probably would, 5=definitely would)
- overall_appeal: How appealing do they find the concept? (1=not at all, 2=slightly, 3=somewhat, 4=very, 5=extremely)
- uniqueness: How unique/differentiated vs alternatives? (1=not unique, 2=slightly, 3=somewhat, 4=very, 5=completely unique)
- relevance: How relevant to their life/needs? (1=not relevant, 2=slightly, 3=somewhat, 4=very, 5=extremely relevant)
- believability: How believable are the claims? (1=not believable, 2=slightly, 3=somewhat, 4=very, 5=completely believable)
- value_perception: How good is the value for money? (1=terrible, 2=poor, 3=fair, 4=good, 5=excellent)

IMPORTANT: Base scores strictly on what each participant SAID in their statements. If they expressed enthusiasm, interest, or intent to purchase, score high (4-5). If they were lukewarm or mixed, score middle (3). If they were critical or dismissive, score low (1-2). Do NOT default to middle scores — differentiate based on actual sentiment.

Return strict JSON object keyed by participant ID, each value being an object with the 6 metric scores (use decimals like 3.5).
Example: {{"p1": {{"purchase_intent": 3.5, "overall_appeal": 4.0, "uniqueness": 3.0, "relevance": 4.5, "believability": 3.5, "value_perception": 3.0}}, "p2": ...}}

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
