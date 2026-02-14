from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
from pydantic import BaseModel

from .models import DiversityTarget, Persona


class DiversityReport(BaseModel):
    opinion_entropy: float
    personality_spread: dict[str, float]
    passes: bool
    issues: list[str]


def _sign_bucket(valence: float | None) -> str:
    if valence is None:
        return "none"
    if valence > 0.1:
        return "positive"
    if valence < -0.1:
        return "negative"
    return "neutral"


def _entropy(values: list[float | None]) -> float:
    usable = [v for v in values if v is not None]
    if not usable:
        return 0.0

    bins = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    hist = np.histogram(usable, bins=bins)[0]
    probs = hist[hist > 0] / np.sum(hist)
    return float(-np.sum(probs * np.log2(probs)))


class DiversityChecker:
    def __init__(self, resample_fn: Callable[[int], list[Persona]] | None = None):
        self._resample_fn = resample_fn

    def check(self, personas: list[Persona], target: DiversityTarget | None = None) -> DiversityReport:
        target = target or DiversityTarget()
        issues: list[str] = []

        if not personas:
            return DiversityReport(
                opinion_entropy=0.0,
                personality_spread={
                    "openness": 0.0,
                    "conscientiousness": 0.0,
                    "extraversion": 0.0,
                    "agreeableness": 0.0,
                    "neuroticism": 0.0,
                },
                passes=False,
                issues=["No personas provided"],
            )

        openness = np.array([p.psychographics.ocean.openness for p in personas], dtype=float)
        conscientiousness = np.array([p.psychographics.ocean.conscientiousness for p in personas], dtype=float)
        extraversion = np.array([p.psychographics.ocean.extraversion for p in personas], dtype=float)
        agreeableness = np.array([p.psychographics.ocean.agreeableness for p in personas], dtype=float)
        neuroticism = np.array([p.psychographics.ocean.neuroticism for p in personas], dtype=float)

        personality_spread = {
            "openness": float(np.std(openness)),
            "conscientiousness": float(np.std(conscientiousness)),
            "extraversion": float(np.std(extraversion)),
            "agreeableness": float(np.std(agreeableness)),
            "neuroticism": float(np.std(neuroticism)),
        }

        if target.require_contrarian and not np.any(agreeableness < 30):
            issues.append("Missing contrarian (agreeableness < 30)")
        if target.require_enthusiast and not np.any(openness > 80):
            issues.append("Missing enthusiast (openness > 80)")
        if target.require_skeptic and not np.any(openness < 30):
            issues.append("Missing skeptic (openness < 30)")
        if target.require_worrier and not np.any(neuroticism > 70):
            issues.append("Missing worrier (neuroticism > 70)")

        opinion_entropy = _entropy([p.opinion_valence for p in personas])
        if opinion_entropy < target.min_opinion_entropy:
            issues.append(f"Opinion entropy too low ({opinion_entropy:.2f} < {target.min_opinion_entropy})")

        for trait, spread in personality_spread.items():
            if spread < target.min_trait_std:
                issues.append(f"Trait spread too low for {trait} ({spread:.2f} < {target.min_trait_std})")

        signs = Counter(_sign_bucket(p.opinion_valence) for p in personas)
        if max(signs.get("positive", 0), signs.get("negative", 0)) > target.max_same_sign:
            issues.append("Too many personas share the same opinion sign")

        return DiversityReport(
            opinion_entropy=opinion_entropy,
            personality_spread=personality_spread,
            passes=len(issues) == 0,
            issues=issues,
        )

    def enforce(self, personas: list[Persona], target: DiversityTarget) -> list[Persona]:
        report = self.check(personas, target)
        if report.passes:
            return personas

        if self._resample_fn is None:
            # Fall back to light-touch adjustment if no external resampler is set.
            adjusted = list(personas)
            if adjusted:
                adjusted[0].psychographics.ocean.agreeableness = min(adjusted[0].psychographics.ocean.agreeableness, 25)
                adjusted[0].psychographics.ocean.openness = max(adjusted[0].psychographics.ocean.openness, 82)
                adjusted[0].psychographics.ocean.neuroticism = max(adjusted[0].psychographics.ocean.neuroticism, 72)
            return adjusted

        for _ in range(30):
            candidates = self._resample_fn(len(personas))
            if self.check(candidates, target).passes:
                return candidates

        repaired = list(personas)
        if len(repaired) >= 4:
            repaired[0].psychographics.ocean.agreeableness = min(repaired[0].psychographics.ocean.agreeableness, 25)
            repaired[1].psychographics.ocean.openness = max(repaired[1].psychographics.ocean.openness, 85)
            repaired[2].psychographics.ocean.openness = min(repaired[2].psychographics.ocean.openness, 25)
            repaired[3].psychographics.ocean.neuroticism = max(repaired[3].psychographics.ocean.neuroticism, 75)

        trait_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for trait_name in trait_names:
            vals = np.array([getattr(p.psychographics.ocean, trait_name) for p in repaired], dtype=float)
            if float(np.std(vals)) < target.min_trait_std and len(repaired) > 1:
                ramp = np.linspace(-20, 20, len(repaired))
                for i, p in enumerate(repaired):
                    new_val = float(np.clip(getattr(p.psychographics.ocean, trait_name) + ramp[i], 0, 100))
                    setattr(p.psychographics.ocean, trait_name, new_val)

        seeded_valences = [-0.9, -0.6, -0.3, -0.05, 0.2, 0.5, 0.75, 0.95]
        for i, p in enumerate(repaired):
            p.opinion_valence = seeded_valences[i % len(seeded_valences)]

        return repaired
