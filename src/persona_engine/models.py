from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Location(BaseModel):
    state: str
    metro_area: str | None = None
    urbanicity: str


class Demographics(BaseModel):
    age: int = Field(ge=18, le=85)
    gender: str
    income: int = Field(ge=0)
    education: str
    occupation: str
    location: Location
    household_type: str
    race_ethnicity: str


class OceanScores(BaseModel):
    openness: float = Field(ge=0, le=100)
    conscientiousness: float = Field(ge=0, le=100)
    extraversion: float = Field(ge=0, le=100)
    agreeableness: float = Field(ge=0, le=100)
    neuroticism: float = Field(ge=0, le=100)


class SchwartzValues(BaseModel):
    primary: str
    secondary: str
    tertiary: str | None = None


class Psychographics(BaseModel):
    ocean: OceanScores
    vals_type: str
    schwartz_values: SchwartzValues


class ConsumerProfile(BaseModel):
    price_sensitivity: float = Field(ge=0, le=1)
    brand_loyalty: float = Field(ge=0, le=1)
    research_tendency: float = Field(ge=0, le=1)
    impulse_tendency: float = Field(ge=0, le=1)
    social_influence: float = Field(ge=0, le=1)
    risk_tolerance: float = Field(ge=0, le=1)
    category_engagement: str
    decision_style: str


class VoiceProfile(BaseModel):
    vocabulary_level: str
    verbosity: str
    hedging_tendency: float = Field(ge=0, le=1)
    emotional_expressiveness: float = Field(ge=0, le=1)
    assertiveness: float = Field(ge=0, le=1)
    humor_tendency: float = Field(ge=0, le=1)
    communication_style: str


class Persona(BaseModel):
    id: str
    name: str
    demographics: Demographics
    psychographics: Psychographics
    consumer: ConsumerProfile
    voice: VoiceProfile
    initial_opinion: str | None = None
    opinion_valence: float | None = None

    @field_validator("opinion_valence")
    @classmethod
    def validate_valence(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < -1 or value > 1:
            raise ValueError("opinion_valence must be in [-1, 1]")
        return value


class DiversityTarget(BaseModel):
    min_opinion_entropy: float = 1.5
    min_trait_std: float = 15.0
    max_same_sign: int = 5
    require_contrarian: bool = True
    require_enthusiast: bool = True
    require_skeptic: bool = True
    require_worrier: bool = True
