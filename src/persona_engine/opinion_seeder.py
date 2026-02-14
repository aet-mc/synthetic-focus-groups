from __future__ import annotations

import numpy as np

from .models import Persona


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _opinion_text(valence: float, concept: str, category: str) -> str:
    if valence > 0.5:
        return f"I like this {category} concept. {concept} feels genuinely useful and worth trying."
    if valence > 0.15:
        return f"I am cautiously positive about this {category} idea. {concept} has potential if priced right."
    if valence < -0.5:
        return f"I do not see strong value in this {category} concept. {concept} feels risky or unnecessary."
    if valence < -0.15:
        return f"I am skeptical about this {category} idea. I would need better proof before considering {concept}."
    return f"I am neutral on this {category} concept. {concept} has pros and cons for me."


class OpinionSeeder:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def seed_opinions(
        self,
        personas: list[Persona],
        product_concept: str,
        category: str,
    ) -> list[Persona]:
        for persona in personas:
            o = persona.psychographics.ocean.openness
            n = persona.psychographics.ocean.neuroticism

            base = ((o - 50) / 50.0) * 0.55 + ((50 - n) / 50.0) * 0.45

            engagement_multiplier = {
                "heavy": 1.2,
                "moderate": 1.0,
                "light": 0.8,
                "non_user": 0.6,
            }[persona.consumer.category_engagement]

            price_penalty = (persona.consumer.price_sensitivity - 0.5) * 0.45
            noise = self.rng.uniform(-0.2, 0.2)

            valence = _clamp((base - price_penalty) * engagement_multiplier + noise, -1.0, 1.0)
            persona.opinion_valence = float(valence)
            persona.initial_opinion = _opinion_text(valence, product_concept, category)

        return personas
