from __future__ import annotations

from uuid import uuid4

import numpy as np

from .consumer_behavior import derive_consumer_profile
from .demographics import sample_demographics
from .diversity import DiversityChecker
from .models import DiversityTarget, Persona
from .opinion_seeder import OpinionSeeder
from .psychographics import generate_psychographics
from .voice import derive_voice_profile

MALE_NAMES = [
    "James",
    "Michael",
    "David",
    "Daniel",
    "Matthew",
    "Anthony",
    "Joshua",
    "Christopher",
]
FEMALE_NAMES = [
    "Emma",
    "Olivia",
    "Sophia",
    "Mia",
    "Ava",
    "Charlotte",
    "Amelia",
    "Isabella",
]
NEUTRAL_NAMES = ["Taylor", "Jordan", "Alex", "Casey", "Riley", "Morgan", "Quinn", "Avery"]


class PersonaGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.opinion_seeder = OpinionSeeder(self.rng)

    def _pick_name(self, gender: str) -> str:
        if gender == "male":
            return str(self.rng.choice(MALE_NAMES))
        if gender == "female":
            return str(self.rng.choice(FEMALE_NAMES))
        return str(self.rng.choice(NEUTRAL_NAMES))

    def _build_once(self, n: int, constraints: dict | None = None) -> list[Persona]:
        demos = sample_demographics(n=n, constraints=constraints, rng=self.rng)
        psychos = generate_psychographics(demos, rng=self.rng)

        personas: list[Persona] = []
        for demo, psycho in zip(demos, psychos):
            consumer = derive_consumer_profile(demo, psycho, rng=self.rng)
            voice = derive_voice_profile(demo, psycho)
            personas.append(
                Persona(
                    id=str(uuid4()),
                    name=self._pick_name(demo.gender),
                    demographics=demo,
                    psychographics=psycho,
                    consumer=consumer,
                    voice=voice,
                )
            )
        return personas

    def generate(
        self,
        n: int = 8,
        constraints: dict | None = None,
        product_concept: str | None = None,
        category: str | None = None,
    ) -> list[Persona]:
        target = DiversityTarget()

        def resample_fn(group_size: int) -> list[Persona]:
            generated = self._build_once(group_size, constraints=constraints)
            if product_concept and category:
                return self.opinion_seeder.seed_opinions(generated, product_concept, category)
            return generated

        checker = DiversityChecker(resample_fn=resample_fn)

        personas = self._build_once(n, constraints=constraints)

        if product_concept and category:
            personas = self.opinion_seeder.seed_opinions(personas, product_concept, category)

        report = checker.check(personas, target)
        if not report.passes:
            personas = checker.enforce(personas, target)

        if (product_concept and category) and any(p.opinion_valence is None for p in personas):
            personas = self.opinion_seeder.seed_opinions(personas, product_concept, category)

        return personas
