from __future__ import annotations

from persona_engine.generator import PersonaGenerator
from persona_engine.opinion_seeder import OpinionSeeder


def test_opinion_seeder_sets_valence_and_text() -> None:
    personas = PersonaGenerator(seed=11).generate(n=8)
    seeded = OpinionSeeder().seed_opinions(personas, "AI wardrobe planner", "mobile app")

    assert all(p.opinion_valence is not None for p in seeded)
    assert all(-1 <= p.opinion_valence <= 1 for p in seeded if p.opinion_valence is not None)
    assert all(p.initial_opinion is not None and len(p.initial_opinion) > 10 for p in seeded)
