from __future__ import annotations

import numpy as np

from persona_engine.models import Demographics, Location, OceanScores
from persona_engine.psychographics import derive_vals_type, generate_psychographics


def _demo(age: int, gender: str, income: int = 90_000, education: str = "bachelors") -> Demographics:
    return Demographics(
        age=age,
        gender=gender,
        income=income,
        education=education,
        occupation="software engineer",
        location=Location(state="CA", metro_area="San Francisco", urbanicity="urban"),
        household_type="single",
        race_ethnicity="white",
    )


def test_ocean_age_gender_adjustments_directionally_hold() -> None:
    rng = np.random.default_rng(123)

    young_males = [_demo(25, "male") for _ in range(300)]
    older_males = [_demo(65, "male") for _ in range(300)]
    females = [_demo(35, "female") for _ in range(300)]
    males = [_demo(35, "male") for _ in range(300)]

    psy_young = generate_psychographics(young_males, rng=rng)
    psy_old = generate_psychographics(older_males, rng=rng)
    psy_f = generate_psychographics(females, rng=rng)
    psy_m = generate_psychographics(males, rng=rng)

    mean_e_young = np.mean([p.ocean.extraversion for p in psy_young])
    mean_e_old = np.mean([p.ocean.extraversion for p in psy_old])
    assert mean_e_old < mean_e_young

    mean_a_f = np.mean([p.ocean.agreeableness for p in psy_f])
    mean_a_m = np.mean([p.ocean.agreeableness for p in psy_m])
    mean_n_f = np.mean([p.ocean.neuroticism for p in psy_f])
    mean_n_m = np.mean([p.ocean.neuroticism for p in psy_m])
    assert mean_a_f > mean_a_m
    assert mean_n_f > mean_n_m


def test_vals_assignment_consistency() -> None:
    demo_innovator = _demo(40, "female", income=160_000, education="masters")
    ocean_innovator = OceanScores(
        openness=85,
        conscientiousness=82,
        extraversion=70,
        agreeableness=55,
        neuroticism=40,
    )
    assert derive_vals_type(ocean_innovator, demo_innovator) == "innovator"

    demo_survivor = _demo(58, "male", income=30_000, education="high_school")
    ocean_survivor = OceanScores(
        openness=20,
        conscientiousness=25,
        extraversion=22,
        agreeableness=30,
        neuroticism=72,
    )
    assert derive_vals_type(ocean_survivor, demo_survivor) == "survivor"
