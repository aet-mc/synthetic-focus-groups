from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


class LLMClient:
    """Async wrapper for LLM API calls."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4-20250514") -> None:
        self.model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 300,
    ) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for LLMClient")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()


@dataclass
class MockLLMClient(LLMClient):
    """Deterministic mock LLM for tests and offline simulation."""

    model: str = "mock/model"

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 300,
    ) -> str:
        digest = hashlib.sha256(f"{system_prompt}||{user_prompt}".encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16)

        if "moderating a market research focus group" in user_prompt:
            return self._mock_question(user_prompt, idx)
        if "Determine whether this participant shifted their opinion" in user_prompt:
            return self._mock_shift(user_prompt)
        return self._mock_response(system_prompt, user_prompt, idx)

    def _mock_question(self, prompt: str, idx: int) -> str:
        phase = "general"
        for candidate in ("warmup", "exploration", "deep_dive", "reaction", "synthesis"):
            if f"Current phase: {candidate}" in prompt:
                phase = candidate
                break

        quiet_names = ""
        marker = "Quiet participants to draw out:"
        if marker in prompt:
            quiet_names = prompt.split(marker, 1)[1].splitlines()[0].strip()

        phase_templates = {
            "warmup": [
                "To get started, what does buying in this category usually look like for you?",
                "Before we dive in, tell us about your recent experience with products like this.",
            ],
            "exploration": [
                "What is the first thing that comes to mind when you hear this concept?",
                "What expectations or concerns show up for you when you hear this idea?",
            ],
            "deep_dive": [
                "Which feature, pricing detail, or practical concern would make or break this for you?",
                "What would you need to trust this enough to choose it over alternatives?",
            ],
            "reaction": [
                "After seeing this, what is your immediate reaction and what would you do next?",
                "Based on this material, would you seriously consider trying it? Why or why not?",
            ],
            "synthesis": [
                "If this were available tomorrow, would you buy it? What is the main reason?",
                "Final take: who is this for, and would you personally purchase it?",
            ],
        }

        question = phase_templates.get(phase, phase_templates["exploration"])[idx % 2]
        if quiet_names and quiet_names != "None":
            first_name = quiet_names.split(",")[0].strip()
            if first_name:
                question = f"{question} {first_name}, I especially want your perspective."
        return question

    def _mock_response(self, system_prompt: str, user_prompt: str, idx: int) -> str:
        response_variants = [
            "I can see the upside, but I would still compare options before deciding.",
            "This sounds useful in theory, though I would need clearer proof it works in real life.",
            "I like the direction, especially if pricing stays reasonable and setup is simple.",
            "I am not fully convinced yet; reliability and value would decide it for me.",
            "I would consider trying it once, but long-term use depends on consistency.",
            "Part of me likes it, but I also worry about hidden tradeoffs after purchase.",
        ]
        base = response_variants[idx % len(response_variants)]

        # Add a light interaction cue so responses feel conversational.
        if "Discussion context:" in user_prompt and "Moderator:" in user_prompt:
            if idx % 3 == 0:
                return f"I agree with what was said earlier. {base}"
            if idx % 3 == 1:
                return f"I see it differently from some of the group. {base}"
        return base

    def _mock_shift(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ["now i would buy", "changed my mind", "more positive now"]):
            return '{"changed_mind": true, "new_valence": 0.4}'
        if any(token in lowered for token in ["now i would avoid", "less interested now", "more negative now"]):
            return '{"changed_mind": true, "new_valence": -0.4}'
        return '{"changed_mind": false, "new_valence": null}'
