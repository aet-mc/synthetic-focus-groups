from __future__ import annotations

import pytest

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


def test_diversity_check_flags_gender_homogeneity() -> None:
    personas = PersonaGenerator(seed=11).generate(
        n=8,
        constraints={"gender": "male"},
        product_concept="Compact travel umbrella",
        category="consumer goods",
    )

    report = DiversityChecker().check(personas, DiversityTarget())

    assert not report.passes
    assert any("Gender diversity too low" in issue for issue in report.issues)


def test_diversity_check_flags_gender_majority() -> None:
    personas = PersonaGenerator(seed=21).generate(
        n=8,
        product_concept="Portable espresso maker",
        category="consumer goods",
    )
    for persona in personas[:5]:
        persona.demographics.gender = "male"
    for persona in personas[5:]:
        persona.demographics.gender = "female"

    report = DiversityChecker().check(personas, DiversityTarget())

    assert not report.passes
    assert any("Gender imbalance too high" in issue for issue in report.issues)


def test_diversity_validate_raises_for_gender_majority() -> None:
    personas = PersonaGenerator(seed=22).generate(
        n=8,
        product_concept="Portable espresso maker",
        category="consumer goods",
    )
    for persona in personas[:5]:
        persona.demographics.gender = "male"
    for persona in personas[5:]:
        persona.demographics.gender = "female"

    target = DiversityTarget(
        min_opinion_entropy=0.0,
        min_trait_std=0.0,
        max_same_sign=8,
        require_contrarian=False,
        require_enthusiast=False,
        require_skeptic=False,
        require_worrier=False,
    )

    with pytest.raises(ValueError, match="Gender imbalance too high"):
        DiversityChecker().validate(personas, target)
