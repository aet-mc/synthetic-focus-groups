from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import Demographics, Location


@dataclass
class WeightedChoice:
    values: tuple[Any, ...]
    weights: tuple[float, ...]


AGE_BUCKETS = WeightedChoice(
    values=((18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 85)),
    weights=(0.12, 0.18, 0.17, 0.16, 0.18, 0.19),
)
GENDER = WeightedChoice(values=("male", "female", "non-binary"), weights=(0.485, 0.495, 0.02))
INCOME_BY_AGE = {
    "young": WeightedChoice(
        values=((20000, 39999), (40000, 69999), (70000, 109999), (110000, 180000)),
        weights=(0.36, 0.38, 0.2, 0.06),
    ),
    "mid": WeightedChoice(
        values=((25000, 44999), (45000, 79999), (80000, 129999), (130000, 220000)),
        weights=(0.24, 0.36, 0.28, 0.12),
    ),
    "older": WeightedChoice(
        values=((22000, 44999), (45000, 74999), (75000, 119999), (120000, 200000)),
        weights=(0.3, 0.37, 0.24, 0.09),
    ),
}
EDUCATION_BY_AGE = {
    "young": WeightedChoice(
        values=("high_school", "some_college", "bachelors", "masters", "doctorate"),
        weights=(0.24, 0.37, 0.29, 0.08, 0.02),
    ),
    "mid": WeightedChoice(
        values=("high_school", "some_college", "bachelors", "masters", "doctorate"),
        weights=(0.2, 0.3, 0.31, 0.15, 0.04),
    ),
    "older": WeightedChoice(
        values=("high_school", "some_college", "bachelors", "masters", "doctorate"),
        weights=(0.3, 0.31, 0.25, 0.1, 0.04),
    ),
}
STATE_WEIGHTS = WeightedChoice(
    values=("CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "WA", "AZ"),
    weights=(0.12, 0.09, 0.075, 0.06, 0.04, 0.04, 0.035, 0.032, 0.03, 0.028, 0.026, 0.024),
)
STATE_REGION = {
    "CA": "west",
    "WA": "west",
    "AZ": "west",
    "TX": "south",
    "FL": "south",
    "GA": "south",
    "NC": "south",
    "NY": "northeast",
    "PA": "northeast",
    "IL": "midwest",
    "OH": "midwest",
    "MI": "midwest",
}
URBANICITY_BY_REGION = {
    "west": WeightedChoice(values=("urban", "suburban", "rural"), weights=(0.38, 0.46, 0.16)),
    "south": WeightedChoice(values=("urban", "suburban", "rural"), weights=(0.29, 0.44, 0.27)),
    "northeast": WeightedChoice(values=("urban", "suburban", "rural"), weights=(0.35, 0.51, 0.14)),
    "midwest": WeightedChoice(values=("urban", "suburban", "rural"), weights=(0.27, 0.45, 0.28)),
}
METRO_BY_STATE = {
    "CA": ("Los Angeles", "San Francisco", "San Diego"),
    "TX": ("Dallas", "Houston", "Austin"),
    "FL": ("Miami", "Tampa", "Orlando"),
    "NY": ("New York", "Buffalo", "Rochester"),
    "PA": ("Philadelphia", "Pittsburgh", None),
    "IL": ("Chicago", None),
    "OH": ("Columbus", "Cleveland", "Cincinnati"),
    "GA": ("Atlanta", None),
    "NC": ("Charlotte", "Raleigh", None),
    "MI": ("Detroit", "Grand Rapids", None),
    "WA": ("Seattle", None),
    "AZ": ("Phoenix", "Tucson", None),
}
RACE_ETHNICITY = WeightedChoice(
    values=("white", "black", "hispanic_latino", "asian", "native_american", "multiracial"),
    weights=(0.58, 0.12, 0.19, 0.07, 0.01, 0.03),
)
HOUSEHOLD_BY_AGE = {
    "young": WeightedChoice(
        values=("single", "married_no_kids", "married_with_kids", "single_parent"),
        weights=(0.56, 0.2, 0.14, 0.1),
    ),
    "mid": WeightedChoice(
        values=("single", "married_no_kids", "married_with_kids", "single_parent"),
        weights=(0.31, 0.23, 0.33, 0.13),
    ),
    "older": WeightedChoice(
        values=("single", "married_no_kids", "married_with_kids", "single_parent"),
        weights=(0.34, 0.39, 0.19, 0.08),
    ),
}
OCCUPATION_BY_EDU = {
    "high_school": ("retail associate", "warehouse worker", "driver", "administrative assistant"),
    "some_college": ("sales representative", "customer success specialist", "technician", "office coordinator"),
    "bachelors": ("software engineer", "teacher", "marketing manager", "business analyst", "nurse"),
    "masters": ("product manager", "financial analyst", "data scientist", "engineering manager"),
    "doctorate": ("research scientist", "physician", "professor", "principal engineer"),
}
TECH_OCCUPATIONS = {
    "software engineer",
    "data scientist",
    "product manager",
    "engineering manager",
    "principal engineer",
    "business analyst",
    "technician",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _weighted_pick(rng: np.random.Generator, choice: WeightedChoice):
    return rng.choice(choice.values, p=np.array(choice.weights) / np.sum(choice.weights))


def _age_band(age: int) -> str:
    if age <= 34:
        return "young"
    if age <= 59:
        return "mid"
    return "older"


def _sample_one(rng: np.random.Generator) -> Demographics:
    age_min, age_max = _weighted_pick(rng, AGE_BUCKETS)
    age = int(rng.integers(age_min, age_max + 1))
    band = _age_band(age)
    gender = _weighted_pick(rng, GENDER)
    income_low, income_high = _weighted_pick(rng, INCOME_BY_AGE[band])
    income = int(rng.integers(income_low, income_high + 1))
    education = _weighted_pick(rng, EDUCATION_BY_AGE[band])
    state = _weighted_pick(rng, STATE_WEIGHTS)
    region = STATE_REGION[state]
    urbanicity = _weighted_pick(rng, URBANICITY_BY_REGION[region])
    metros = METRO_BY_STATE[state]
    if urbanicity == "rural":
        metro_area = None
    else:
        metro_area = rng.choice(metros)
    household_type = _weighted_pick(rng, HOUSEHOLD_BY_AGE[band])
    if gender == "male" and household_type == "single_parent" and rng.random() < 0.2:
        household_type = "single"
    occupation = rng.choice(OCCUPATION_BY_EDU[education])
    if education in {"bachelors", "masters"} and rng.random() < 0.18:
        occupation = "software engineer"
    race_ethnicity = _weighted_pick(rng, RACE_ETHNICITY)
    return Demographics(
        age=age,
        gender=gender,
        income=income,
        education=education,
        occupation=occupation,
        location=Location(state=state, metro_area=metro_area, urbanicity=urbanicity),
        household_type=household_type,
        race_ethnicity=race_ethnicity,
    )


def _matches_constraints(demo: Demographics, constraints: dict[str, Any] | None) -> bool:
    if not constraints:
        return True

    age_range = constraints.get("age_range")
    if age_range and not (age_range[0] <= demo.age <= age_range[1]):
        return False
    income_min = constraints.get("income_min")
    if income_min is not None and demo.income < income_min:
        return False
    income_max = constraints.get("income_max")
    if income_max is not None and demo.income > income_max:
        return False

    gender = constraints.get("gender")
    if gender is not None:
        allowed = {gender} if isinstance(gender, str) else set(gender)
        if demo.gender not in allowed:
            return False

    education = constraints.get("education")
    if education is not None:
        allowed = {education} if isinstance(education, str) else set(education)
        if demo.education not in allowed:
            return False

    states = constraints.get("states")
    if states is not None:
        allowed = {states} if isinstance(states, str) else set(states)
        if demo.location.state not in allowed:
            return False

    urbanicity = constraints.get("urbanicity")
    if urbanicity is not None:
        allowed = {urbanicity} if isinstance(urbanicity, str) else set(urbanicity)
        if demo.location.urbanicity not in allowed:
            return False

    if constraints.get("occupation_contains") and constraints["occupation_contains"].lower() not in demo.occupation.lower():
        return False

    occupation_sector = constraints.get("occupation_sector")
    if occupation_sector == "tech" and demo.occupation not in TECH_OCCUPATIONS:
        return False

    household_type = constraints.get("household_type")
    if household_type is not None:
        allowed = {household_type} if isinstance(household_type, str) else set(household_type)
        if demo.household_type not in allowed:
            return False

    race_ethnicity = constraints.get("race_ethnicity")
    if race_ethnicity is not None:
        allowed = {race_ethnicity} if isinstance(race_ethnicity, str) else set(race_ethnicity)
        if demo.race_ethnicity not in allowed:
            return False

    return True


def sample_demographics(
    n: int,
    constraints: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Demographics]:
    rng = rng or np.random.default_rng()
    if n <= 0:
        return []

    samples: list[Demographics] = []
    attempts = 0
    max_attempts = max(5000, n * 1000)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        candidate = _sample_one(rng)

        # Constraint-guided nudges to maintain speed for tight filters.
        if constraints:
            if constraints.get("gender") and isinstance(constraints["gender"], str):
                candidate.gender = constraints["gender"]
            if constraints.get("age_range"):
                lo, hi = constraints["age_range"]
                if candidate.age < lo or candidate.age > hi:
                    candidate.age = int(rng.integers(lo, hi + 1))
            if constraints.get("occupation_sector") == "tech" and candidate.occupation not in TECH_OCCUPATIONS:
                candidate.occupation = rng.choice(sorted(TECH_OCCUPATIONS))
            if constraints.get("income_min") is not None and candidate.income < constraints["income_min"]:
                candidate.income = int(_clamp(constraints["income_min"], 0, 500000))
            if constraints.get("income_max") is not None and candidate.income > constraints["income_max"]:
                candidate.income = int(_clamp(constraints["income_max"], 0, 500000))

        if _matches_constraints(candidate, constraints):
            samples.append(candidate)

    if len(samples) < n:
        raise ValueError("Unable to satisfy demographic constraints with current distributions")
    return samples
