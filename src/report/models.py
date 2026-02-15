from __future__ import annotations

from pydantic import BaseModel


class MetricDisplay(BaseModel):
    name: str
    score: float
    formatted_score: str
    interpretation: str
    status: str


class ThemeDisplay(BaseModel):
    name: str
    description: str
    prevalence_text: str
    sentiment_label: str
    quotes: list[dict[str, str]]
