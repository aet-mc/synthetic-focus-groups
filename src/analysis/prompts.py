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
For each participant below, score their reaction to the product concept on these 1-5 metrics:
- purchase_intent
- overall_appeal
- uniqueness
- relevance
- believability
- value_perception

Return strict JSON object keyed by participant ID, each value being an object with the 6 metric scores.
Example: {{"p1": {{"purchase_intent": 3.5, "overall_appeal": 4.0, ...}}, "p2": ...}}

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
