from __future__ import annotations

import numpy as np

from .models import Demographics, Psychographics, VoiceProfile

VOCAB_BY_EDU = {
    "high_school": "basic",
    "some_college": "moderate",
    "bachelors": "advanced",
    "masters": "advanced",
    "doctorate": "expert",
}


def _clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _verbosity(extraversion: float) -> str:
    if extraversion >= 67:
        return "verbose"
    if extraversion <= 38:
        return "terse"
    return "moderate"


def _style(psychographics: Psychographics) -> str:
    o = psychographics.ocean.openness
    c = psychographics.ocean.conscientiousness
    e = psychographics.ocean.extraversion
    a = psychographics.ocean.agreeableness
    n = psychographics.ocean.neuroticism

    if c >= 65 and o >= 55:
        return "analytical"
    if a >= 65 or (a >= 58 and n >= 58):
        return "diplomatic"
    if e >= 65 and o >= 55:
        return "storytelling"
    return "direct"


def derive_voice_profile(demo: Demographics, psychographics: Psychographics) -> VoiceProfile:
    o = psychographics.ocean.openness / 100.0
    c = psychographics.ocean.conscientiousness / 100.0
    e = psychographics.ocean.extraversion / 100.0
    a = psychographics.ocean.agreeableness / 100.0
    n = psychographics.ocean.neuroticism / 100.0

    return VoiceProfile(
        vocabulary_level=VOCAB_BY_EDU[demo.education],
        verbosity=_verbosity(psychographics.ocean.extraversion),
        hedging_tendency=_clamp01(n * a),
        emotional_expressiveness=_clamp01(e * (1 - c)),
        assertiveness=_clamp01(e * (1 - a)),
        humor_tendency=_clamp01(o * e),
        communication_style=_style(psychographics),
    )
