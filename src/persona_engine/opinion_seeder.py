from __future__ import annotations

import numpy as np

from .models import Persona


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _opinion_text(valence: float, concept: str, category: str, persona=None) -> str:
    """Generate a grounded initial opinion using persona traits for variation."""
    # Base stance from valence
    if valence > 0.5:
        stance = "enthusiastic"
        core = f"{concept} feels genuinely useful and worth trying"
    elif valence > 0.15:
        stance = "cautiously positive"
        core = f"{concept} has potential if executed well"
    elif valence < -0.5:
        stance = "strongly skeptical"
        core = f"{concept} feels risky or unnecessary"
    elif valence < -0.15:
        stance = "hesitant"
        core = f"I would need better proof before considering {concept}"
    else:
        stance = "neutral"
        core = f"{concept} has both pros and cons for me"

    if persona is None:
        return f"I am {stance} about this {category} concept. {core}."

    # Build persona-specific reasoning
    reasons = []
    ocean = persona.psychographics.ocean

    if ocean.openness >= 70:
        reasons.append("I'm usually open to trying new things in this space")
    elif ocean.openness <= 30:
        reasons.append("I generally stick with what I know works")

    if ocean.neuroticism >= 70:
        reasons.append("I worry about what could go wrong")
    elif ocean.neuroticism <= 30:
        reasons.append("I don't stress much about trying something new")

    if persona.consumer.price_sensitivity >= 0.7:
        reasons.append("price is always top of mind for me")
    elif persona.consumer.price_sensitivity <= 0.3:
        reasons.append("I'm willing to pay more for the right solution")

    if persona.consumer.research_tendency >= 0.7:
        reasons.append("I'd research this thoroughly before committing")
    elif persona.consumer.research_tendency <= 0.3:
        reasons.append("I tend to go with my gut on purchases like this")

    if persona.consumer.brand_loyalty >= 0.7:
        reasons.append("I'm loyal to brands I already trust in this category")

    # Pick 2-3 reasons for variety
    selected = reasons[:3] if reasons else ["I have mixed feelings about it"]
    reason_text = ", and ".join(selected[:2])
    if len(selected) > 2:
        reason_text += f". Also, {selected[2]}"

    demo = persona.demographics
    context = ""
    if demo.age >= 60:
        context = f" As someone in my {demo.age // 10 * 10}s, I've seen a lot of products come and go."
    elif demo.age <= 30:
        context = f" At {demo.age}, I'm always looking for what's next."

    return f"I'm {stance} about this {category} concept. {core} — {reason_text}.{context}"


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
            persona.initial_opinion = _opinion_text(valence, product_concept, category, persona=persona)

        return personas
