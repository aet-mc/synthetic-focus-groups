from __future__ import annotations

from persona_engine.demographics import sample_demographics


def test_sample_demographics_constraints_respected() -> None:
    demos = sample_demographics(
        40,
        constraints={
            "age_range": (30, 40),
            "income_min": 60_000,
            "gender": "female",
            "occupation_sector": "tech",
        },
    )

    assert len(demos) == 40
    assert all(30 <= d.age <= 40 for d in demos)
    assert all(d.income >= 60_000 for d in demos)
    assert all(d.gender == "female" for d in demos)
    assert all(
        d.occupation
        in {
            "software engineer",
            "data scientist",
            "product manager",
            "engineering manager",
            "principal engineer",
            "business analyst",
            "technician",
        }
        for d in demos
    )
