from __future__ import annotations

from persona_engine.diversity import DiversityChecker
from persona_engine.generator import PersonaGenerator


def test_generate_8_personas_diversity_passes() -> None:
    generator = PersonaGenerator(seed=42)
    personas = generator.generate(n=8, product_concept="AI meal planner", category="app")
    assert len(personas) == 8

    report = DiversityChecker().check(personas)
    assert report.passes, report.issues


def test_generate_with_tight_constraints() -> None:
    generator = PersonaGenerator(seed=101)
    personas = generator.generate(
        n=8,
        constraints={
            "gender": "female",
            "age_range": (25, 35),
            "occupation_sector": "tech",
        },
        product_concept="AI meeting assistant",
        category="software",
    )

    assert len(personas) == 8
    assert all(p.demographics.gender == "female" for p in personas)
    assert all(25 <= p.demographics.age <= 35 for p in personas)
    assert all(
        p.demographics.occupation
        in {
            "software engineer",
            "data scientist",
            "product manager",
            "engineering manager",
            "principal engineer",
            "business analyst",
            "technician",
        }
        for p in personas
    )


def test_opinion_valences_span_range_across_runs() -> None:
    lows: list[float] = []
    highs: list[float] = []
    for seed in range(10):
        generator = PersonaGenerator(seed=seed)
        personas = generator.generate(
            n=8,
            product_concept="Self-heating lunchbox",
            category="kitchen gadget",
        )
        valences = [p.opinion_valence for p in personas if p.opinion_valence is not None]
        lows.append(min(valences))
        highs.append(max(valences))

    assert min(lows) < -0.45
    assert max(highs) > 0.45
