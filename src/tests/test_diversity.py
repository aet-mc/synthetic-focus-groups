from __future__ import annotations

from persona_engine.diversity import DiversityChecker
from persona_engine.generator import PersonaGenerator
from persona_engine.models import DiversityTarget


def test_diversity_enforcer_resamples_when_violated() -> None:
    generator = PersonaGenerator(seed=7)
    homogeneous = generator.generate(
        n=8,
        constraints={"age_range": (30, 31), "gender": "male", "income_min": 40_000, "income_max": 42_000},
        product_concept="Basic reusable bottle",
        category="consumer goods",
    )

    # Force diversity failure.
    for persona in homogeneous:
        persona.psychographics.ocean.agreeableness = 60
        persona.psychographics.ocean.openness = 45
        persona.psychographics.ocean.neuroticism = 45
        persona.opinion_valence = 0.4

    checker = DiversityChecker(
        resample_fn=lambda n: PersonaGenerator(seed=101 + n).generate(
            n=n,
            product_concept="Advanced smart bottle",
            category="consumer goods",
        )
    )
    target = DiversityTarget()

    before_report = checker.check(homogeneous, target)
    assert not before_report.passes

    enforced = checker.enforce(homogeneous, target)
    after_report = checker.check(enforced, target)

    assert after_report.passes, after_report.issues
    assert {p.id for p in enforced} != {p.id for p in homogeneous}
