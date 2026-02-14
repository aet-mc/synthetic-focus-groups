from __future__ import annotations

import numpy as np

from .models import ConsumerProfile, Demographics, Psychographics


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _income_inverse(income: int) -> float:
    normalized = (income - 20_000) / 180_000
    return _clamp01(1.0 - normalized)


def _decision_style(psy: Psychographics) -> str:
    o = psy.ocean.openness
    c = psy.ocean.conscientiousness
    e = psy.ocean.extraversion
    a = psy.ocean.agreeableness
    n = psy.ocean.neuroticism

    dominant = max(
        {
            "openness": o,
            "conscientiousness": c,
            "extraversion": e,
            "agreeableness": a,
            "neuroticism": n,
        }.items(),
        key=lambda item: item[1],
    )[0]

    if dominant == "conscientiousness":
        return "analytical"
    if dominant == "neuroticism":
        return "emotional"
    if dominant == "agreeableness":
        return "habitual"
    if dominant == "extraversion" and c < 45:
        return "impulsive"
    if dominant == "openness" and c >= 55:
        return "analytical"
    if e >= 65 and c < 50:
        return "impulsive"
    return "habitual"


def _category_engagement(demo: Demographics, rng: np.random.Generator) -> str:
    age_factor = 1.0 if demo.age <= 40 else 0.75 if demo.age <= 60 else 0.55
    income_factor = 1.0 if demo.income >= 90_000 else 0.85 if demo.income >= 50_000 else 0.65
    heavy = 0.22 * age_factor * income_factor
    moderate = 0.44
    light = 0.26 + (0.1 if demo.age > 60 else 0)
    non_user = max(0.04, 1.0 - heavy - moderate - light)

    probs = np.array([heavy, moderate, light, non_user], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(["heavy", "moderate", "light", "non_user"], p=probs))


def derive_consumer_profile(
    demographics: Demographics,
    psychographics: Psychographics,
    rng: np.random.Generator | None = None,
) -> ConsumerProfile:
    rng = rng or np.random.default_rng()

    o = psychographics.ocean.openness
    c = psychographics.ocean.conscientiousness
    e = psychographics.ocean.extraversion
    a = psychographics.ocean.agreeableness
    n = psychographics.ocean.neuroticism

    price_sensitivity = _clamp01(_income_inverse(demographics.income) * (1 - c / 200.0))
    brand_loyalty = _clamp01(((100 - o) / 100.0) * 0.55 + (c / 100.0) * 0.45)
    research_tendency = _clamp01((c / 100.0) * 0.7 + (n / 100.0) * 0.3)
    impulse_tendency = _clamp01((1 - c / 100.0) * (e / 100.0))
    social_influence = _clamp01((a / 100.0) * 0.5 + (e / 100.0) * 0.3 + (n / 100.0) * 0.2)
    risk_tolerance = _clamp01((o / 100.0) * 0.4 + (e / 100.0) * 0.3 - (n / 100.0) * 0.3)
    category_engagement = _category_engagement(demographics, rng)
    decision_style = _decision_style(psychographics)

    return ConsumerProfile(
        price_sensitivity=price_sensitivity,
        brand_loyalty=brand_loyalty,
        research_tendency=research_tendency,
        impulse_tendency=impulse_tendency,
        social_influence=social_influence,
        risk_tolerance=risk_tolerance,
        category_engagement=category_engagement,
        decision_style=decision_style,
    )
