from __future__ import annotations

import json
from collections import Counter

from discussion.llm_client import MockLLMClient
from discussion.models import DiscussionTranscript, MessageRole
from persona_engine.models import Persona

from .models import Theme
from .prompts import THEME_CLUSTERING_PROMPT, THEME_CODING_PROMPT


class ThemeExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self._persona_map: dict[str, Persona] = {}

    def set_personas(self, personas: list[Persona]) -> None:
        self._persona_map = {persona.id: persona for persona in personas}

    async def extract_themes(self, transcript: DiscussionTranscript, max_themes: int = 7) -> list[Theme]:
        participant_records = [
            (index, message)
            for index, message in enumerate(transcript.messages)
            if message.role == MessageRole.PARTICIPANT
        ]
        if not participant_records:
            return []

        if isinstance(self.llm, MockLLMClient):
            coded_messages = self._mock_code_messages(participant_records)
            clustered = self._mock_cluster_codes(coded_messages, max_themes=max_themes)
        else:
            coded_messages = await self._code_messages_with_llm(participant_records)
            clustered = await self._cluster_codes_with_llm(coded_messages, max_themes=max_themes)
            if not coded_messages or not clustered:
                coded_messages = self._mock_code_messages(participant_records)
                clustered = self._mock_cluster_codes(coded_messages, max_themes=max_themes)

        return self._build_themes(
            transcript=transcript,
            participant_records=participant_records,
            coded_messages=coded_messages,
            clustered=clustered,
        )

    async def _code_messages_with_llm(
        self, participant_records: list[tuple[int, object]]
    ) -> list[dict[str, object]]:
        transcript_text = "\n".join(
            f"[{index}] ({message.phase.value}) {message.speaker_name}: {message.content}"
            for index, message in participant_records
        )
        prompt = THEME_CODING_PROMPT.format(transcript=transcript_text)
        raw = await self.llm.complete_json(
            system_prompt="You are a qualitative researcher. Return JSON only.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=1500,
        )

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []
            normalized: list[dict[str, object]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                message_index = int(item.get("message_index"))
                codes = [str(code).strip().lower() for code in item.get("codes", []) if str(code).strip()]
                if not codes:
                    continue
                normalized.append({"message_index": message_index, "codes": list(dict.fromkeys(codes))[:3]})
            return normalized
        except (ValueError, TypeError, json.JSONDecodeError):
            return []

    async def _cluster_codes_with_llm(
        self, coded_messages: list[dict[str, object]], max_themes: int
    ) -> list[dict[str, object]]:
        all_codes = [
            str(code)
            for record in coded_messages
            for code in record.get("codes", [])
            if str(code).strip()
        ]
        code_counts = Counter(all_codes)
        codes_block = "\n".join(f"- {code}: {count}" for code, count in code_counts.most_common())
        prompt = THEME_CLUSTERING_PROMPT.format(max_themes=max_themes, codes=codes_block)

        raw = await self.llm.complete_json(
            system_prompt="You are a thematic analysis expert. Return JSON only.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=1200,
        )

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []
            normalized: list[dict[str, object]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                description = str(item.get("description", "")).strip()
                codes = [str(code).strip().lower() for code in item.get("codes", []) if str(code).strip()]
                if not name or not description or not codes:
                    continue
                normalized.append({"name": name, "description": description, "codes": list(dict.fromkeys(codes))})
            return normalized[:max_themes]
        except (ValueError, TypeError, json.JSONDecodeError):
            return []

    def _mock_code_messages(self, participant_records: list[tuple[int, object]]) -> list[dict[str, object]]:
        keyword_map = {
            "price": "price_sensitivity",
            "cost": "price_sensitivity",
            "value": "value_for_money",
            "setup": "ease_of_use",
            "simple": "ease_of_use",
            "easy": "ease_of_use",
            "feature": "feature_depth",
            "innov": "novelty",
            "new": "novelty",
            "trust": "trust_and_proof",
            "proof": "trust_and_proof",
            "reliab": "trust_and_proof",
            "buy": "purchase_intent",
            "purchase": "purchase_intent",
            "worry": "risk_concerns",
            "risk": "risk_concerns",
            "concern": "risk_concerns",
            "compare": "competitive_evaluation",
            "alternat": "competitive_evaluation",
        }

        coded: list[dict[str, object]] = []
        for index, message in participant_records:
            lowered = message.content.lower()
            codes: list[str] = []
            for token, code in keyword_map.items():
                if token in lowered and code not in codes:
                    codes.append(code)
            persona = self._persona_map.get(message.speaker_id)
            valence = persona.opinion_valence if persona and persona.opinion_valence is not None else 0.0
            codes.append("positive_reaction" if valence >= 0.2 else "skepticism" if valence <= -0.2 else "mixed_reaction")

            if not codes:
                codes = ["general_feedback"]

            coded.append({"message_index": index, "codes": codes[:3]})

        return coded

    @staticmethod
    def _mock_cluster_codes(coded_messages: list[dict[str, object]], max_themes: int) -> list[dict[str, object]]:
        theme_templates = {
            "Value and Pricing": {
                "description": "Participants assessed cost, value, and willingness to pay.",
                "codes": {"price_sensitivity", "value_for_money", "purchase_intent"},
            },
            "Usability and Fit": {
                "description": "Feedback focused on ease of use and practical day-to-day fit.",
                "codes": {"ease_of_use", "feature_depth", "general_feedback"},
            },
            "Trust and Risk": {
                "description": "Participants discussed credibility, reliability, and downside risks.",
                "codes": {"trust_and_proof", "risk_concerns", "skepticism"},
            },
            "Differentiation": {
                "description": "Comments compared the concept against alternatives and novelty expectations.",
                "codes": {"novelty", "competitive_evaluation", "mixed_reaction"},
            },
            "Positive Momentum": {
                "description": "Statements reflected enthusiasm and optimism about trying the concept.",
                "codes": {"positive_reaction", "purchase_intent", "novelty"},
            },
        }

        all_codes = [
            str(code)
            for record in coded_messages
            for code in record.get("codes", [])
            if str(code).strip()
        ]
        code_counts = Counter(all_codes)

        ranked: list[tuple[str, int]] = []
        for theme_name, template in theme_templates.items():
            overlap = sum(code_counts.get(code, 0) for code in template["codes"])
            ranked.append((theme_name, overlap))

        ranked.sort(key=lambda pair: pair[1], reverse=True)

        clustered: list[dict[str, object]] = []
        for theme_name, overlap in ranked:
            if overlap == 0:
                continue
            template = theme_templates[theme_name]
            clustered.append(
                {
                    "name": theme_name,
                    "description": template["description"],
                    "codes": sorted(template["codes"]),
                }
            )
            if len(clustered) >= max_themes:
                break

        if not clustered:
            clustered.append(
                {
                    "name": "General Feedback",
                    "description": "Participants shared broad reactions to the concept.",
                    "codes": ["general_feedback", "mixed_reaction"],
                }
            )

        return clustered

    def _build_themes(
        self,
        transcript: DiscussionTranscript,
        participant_records: list[tuple[int, object]],
        coded_messages: list[dict[str, object]],
        clustered: list[dict[str, object]],
    ) -> list[Theme]:
        all_participant_ids = {message.speaker_id for _, message in participant_records}
        message_lookup = {index: message for index, message in participant_records}

        code_map: dict[int, list[str]] = {}
        for record in coded_messages:
            message_index = int(record["message_index"])
            codes = [str(code) for code in record.get("codes", [])]
            code_map[message_index] = codes

        themes: list[Theme] = []
        for cluster in clustered:
            cluster_codes = {str(code) for code in cluster.get("codes", [])}
            matched_indices = [
                index
                for index, codes in code_map.items()
                if cluster_codes.intersection(codes)
            ]
            if not matched_indices:
                continue

            matched_messages = [message_lookup[index] for index in matched_indices if index in message_lookup]
            participant_ids = sorted({message.speaker_id for message in matched_messages})
            prevalence = len(participant_ids) / max(1, len(all_participant_ids))

            sentiments: list[float] = []
            for message in matched_messages:
                if message.sentiment is not None:
                    sentiments.append(float(message.sentiment))
                    continue
                persona = self._persona_map.get(message.speaker_id)
                if persona and persona.opinion_valence is not None:
                    sentiments.append(float(persona.opinion_valence))
                else:
                    sentiments.append(0.0)
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            phase_distribution: dict[str, int] = {}
            for message in matched_messages:
                key = message.phase.value
                phase_distribution[key] = phase_distribution.get(key, 0) + 1

            quote_candidates = sorted(
                {message.content.strip() for message in matched_messages if message.content.strip()},
                key=lambda quote: len(quote),
                reverse=True,
            )
            supporting_quotes = quote_candidates[:5] if quote_candidates else []

            themes.append(
                Theme(
                    name=str(cluster.get("name", "Theme")).strip(),
                    description=str(cluster.get("description", "")).strip(),
                    prevalence=round(prevalence, 4),
                    sentiment=round(avg_sentiment, 4),
                    supporting_quotes=supporting_quotes,
                    participant_ids=participant_ids,
                    phase_distribution=phase_distribution,
                )
            )

        themes.sort(key=lambda theme: theme.prevalence, reverse=True)
        return themes
