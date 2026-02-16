from __future__ import annotations

from uuid import uuid4

import numpy as np

from .consumer_behavior import derive_consumer_profile
from .demographics import sample_demographics, sample_unique_names
from .diversity import DiversityChecker
from .models import DiversityTarget, Persona
from .opinion_seeder import OpinionSeeder
from .psychographics import generate_psychographics
from .voice import derive_voice_profile


class PersonaGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.opinion_seeder = OpinionSeeder(self.rng)

    def _build_once(self, n: int, constraints: dict | None = None) -> list[Persona]:
        demos = sample_demographics(n=n, constraints=constraints, rng=self.rng)
        psychos = generate_psychographics(demos, rng=self.rng)
        names = sample_unique_names([d.gender for d in demos], self.rng)

        personas: list[Persona] = []
        for demo, psycho, name in zip(demos, psychos, names):
            consumer = derive_consumer_profile(demo, psycho, rng=self.rng)
            voice = derive_voice_profile(demo, psycho)
            personas.append(
                Persona(
                    id=str(uuid4()),
                    name=name,
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

        # Enforce demographic diversity for large pools
        self._enforce_large_pool_diversity(personas)

        return personas

    def _enforce_large_pool_diversity(self, personas: list[Persona]) -> None:
        """Ensure age bracket and geographic diversity for large pools."""
        n = len(personas)
        if n <= 12:
            return

        # For n > 12: ensure at least 3 distinct age brackets
        age_brackets = {self._age_bracket(p.demographics.age) for p in personas}
        if len(age_brackets) < 3 and n > 12:
            all_brackets = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            missing = [b for b in all_brackets if b not in age_brackets]
            bracket_ages = {
                "18-24": 21, "25-34": 30, "35-44": 40,
                "45-54": 50, "55-64": 60, "65+": 70,
            }
            for i, bracket in enumerate(missing):
                if len(age_brackets) >= 3:
                    break
                if i < len(personas):
                    personas[-(i + 1)].demographics.age = bracket_ages[bracket]
                    age_brackets.add(bracket)

        # For n > 16: ensure at least 4 distinct geographic regions
        if n > 16:
            from .demographics import STATE_REGION
            regions = {STATE_REGION.get(p.demographics.location.state, "unknown") for p in personas}
            all_regions = ["west", "south", "northeast", "midwest"]
            region_states = {"west": "CA", "south": "TX", "northeast": "NY", "midwest": "IL"}
            missing_regions = [r for r in all_regions if r not in regions]
            for i, region in enumerate(missing_regions):
                if len(regions) >= 4:
                    break
                if i < len(personas):
                    personas[-(i + 1)].demographics.location.state = region_states[region]
                    regions.add(region)

    @staticmethod
    def _age_bracket(age: int) -> str:
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        if age < 65:
            return "55-64"
        return "65+"
