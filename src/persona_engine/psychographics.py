from __future__ import annotations

from collections import defaultdict

import numpy as np

from .models import Demographics, OceanScores, Psychographics, SchwartzValues

VALS_TYPES = {
    "innovator",
    "thinker",
    "achiever",
    "experiencer",
    "believer",
    "striver",
    "maker",
    "survivor",
}
SCHWARTZ_TYPES = [
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
]


def _clip(v: float) -> float:
    return float(np.clip(v, 0, 100))


def _age_adjusted_scores(base: np.ndarray, demo: Demographics) -> np.ndarray:
    openness, conscientiousness, extraversion, agreeableness, neuroticism = base

    decades_after_30 = max(demo.age - 30, 0) / 10.0
    extraversion -= 0.3 * decades_after_30
    agreeableness += 0.2 * decades_after_30

    if 40 <= demo.age <= 60:
        conscientiousness += 6
    elif 30 <= demo.age < 40 or 60 < demo.age <= 70:
        conscientiousness += 3
    else:
        conscientiousness -= 2

    if demo.gender == "female":
        agreeableness += 4
        neuroticism += 4

    return np.array([
        _clip(openness),
        _clip(conscientiousness),
        _clip(extraversion),
        _clip(agreeableness),
        _clip(neuroticism),
    ])


def derive_vals_type(ocean: OceanScores, demo: Demographics) -> str:
    o = ocean.openness
    c = ocean.conscientiousness
    e = ocean.extraversion
    a = ocean.agreeableness
    n = ocean.neuroticism

    high_income = demo.income >= 120_000
    low_income = demo.income < 55_000
    high_edu = demo.education in {"bachelors", "masters", "doctorate"}
    lower_edu = demo.education in {"high_school", "some_college"}
    young = demo.age <= 34

    if o > 65 and c > 65 and high_income and high_edu:
        return "innovator"
    if o > 60 and c > 60 and 40 <= e <= 70 and high_edu:
        return "thinker"
    if c > 65 and e > 60 and demo.income >= 70_000:
        return "achiever"
    if o > 65 and e > 65 and c < 45 and young:
        return "experiencer"
    if o < 40 and a > 55 and c > 55 and lower_edu:
        return "believer"
    if e > 60 and c < 45 and low_income:
        return "striver"
    if e < 45 and c > 55 and low_income:
        return "maker"
    if np.mean([o, c, e, a]) < 45 and demo.income < 45_000:
        return "survivor"

    # Fallback nearest profile
    scores = {
        "innovator": (o + c) / 2 + (15 if high_income else 0) + (10 if high_edu else 0),
        "thinker": (o + c) / 2 - abs(e - 55),
        "achiever": (c + e) / 2 + (8 if demo.income >= 70_000 else 0),
        "experiencer": (o + e - c) / 2 + (8 if young else 0),
        "believer": ((100 - o) + a + c) / 3 + (8 if lower_edu else 0),
        "striver": (e + (100 - c)) / 2 + (8 if low_income else 0),
        "maker": ((100 - e) + c) / 2 + (8 if low_income else 0),
        "survivor": ((100 - o) + (100 - c) + (100 - e) + (100 - a) + n) / 5 + (12 if demo.income < 45_000 else 0),
    }
    return max(scores.items(), key=lambda x: x[1])[0]


def derive_schwartz_values(ocean: OceanScores, rng: np.random.Generator) -> SchwartzValues:
    weights = defaultdict(float)

    weights["self_direction"] += ocean.openness * 1.0
    weights["stimulation"] += ocean.openness * 0.9
    weights["achievement"] += ocean.conscientiousness * 0.8
    weights["security"] += ocean.conscientiousness * 0.6
    weights["power"] += ocean.extraversion * 0.7
    weights["hedonism"] += ocean.extraversion * 0.6
    weights["benevolence"] += ocean.agreeableness * 0.8
    weights["universalism"] += ocean.agreeableness * 0.7
    weights["conformity"] += ocean.agreeableness * 0.5
    weights["security"] += ocean.neuroticism * 0.8
    weights["tradition"] += ocean.neuroticism * 0.6

    ordered = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)]
    primary = ordered[0]
    secondary = ordered[1]
    tertiary = ordered[2] if rng.random() < 0.55 else None
    return SchwartzValues(primary=primary, secondary=secondary, tertiary=tertiary)


def generate_psychographics(
    demographics: list[Demographics],
    rng: np.random.Generator | None = None,
) -> list[Psychographics]:
    rng = rng or np.random.default_rng()
    if not demographics:
        return []

    # Stratify by O/E quadrants to enforce spread.
    quadrant_targets = [(35, 35), (35, 70), (70, 35), (70, 70)]
    psychographics: list[Psychographics] = []

    for idx, demo in enumerate(demographics):
        base = rng.normal(50, 15, size=5)
        adjusted = _age_adjusted_scores(base, demo)
        openness, conscientiousness, extraversion, agreeableness, neuroticism = adjusted

        target_o, target_e = quadrant_targets[idx % len(quadrant_targets)]
        openness = _clip((openness * 0.75) + (target_o * 0.25))
        extraversion = _clip((extraversion * 0.75) + (target_e * 0.25))

        ocean = OceanScores(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
        )
        vals_type = derive_vals_type(ocean, demo)
        schwartz_values = derive_schwartz_values(ocean, rng)

        psychographics.append(
            Psychographics(ocean=ocean, vals_type=vals_type, schwartz_values=schwartz_values)
        )

    if len(psychographics) >= 4:
        psychographics[0].ocean.agreeableness = _clip(min(psychographics[0].ocean.agreeableness, 25))
        psychographics[1].ocean.openness = _clip(max(psychographics[1].ocean.openness, 85))
        psychographics[2].ocean.openness = _clip(min(psychographics[2].ocean.openness, 25))
        psychographics[3].ocean.neuroticism = _clip(max(psychographics[3].ocean.neuroticism, 75))

    trait_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    for trait_name in trait_names:
        values = np.array([getattr(p.ocean, trait_name) for p in psychographics], dtype=float)
        std = float(np.std(values))
        if std < 15 and len(psychographics) > 1:
            ramp = np.linspace(-18, 18, len(psychographics))
            boost = min(1.0, (15 - std) / 10 + 0.2)
            for i, persona in enumerate(psychographics):
                shifted = getattr(persona.ocean, trait_name) + ramp[i] * boost
                setattr(persona.ocean, trait_name, _clip(shifted))

    return psychographics
