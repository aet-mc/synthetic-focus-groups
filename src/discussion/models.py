from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class DiscussionPhase(StrEnum):
    WARMUP = "warmup"
    EXPLORATION = "exploration"
    DEEP_DIVE = "deep_dive"
    REACTION = "reaction"
    SYNTHESIS = "synthesis"


class MessageRole(StrEnum):
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    SYSTEM = "system"


class DiscussionMessage(BaseModel):
    role: MessageRole
    speaker_id: str
    speaker_name: str
    content: str
    phase: DiscussionPhase
    turn_number: int
    replied_to: str | None = None
    sentiment: float | None = None
    changed_mind: bool = False


class DiscussionConfig(BaseModel):
    product_concept: str
    category: str
    stimulus_material: str | None = None
    num_personas: int = 8

    @field_validator("num_personas")
    @classmethod
    def _validate_num_personas(cls, v: int) -> int:
        if v < 4 or v > 48:
            raise ValueError(f"num_personas must be between 4 and 48, got {v}")
        return v

    phases: list[DiscussionPhase] = Field(
        default_factory=lambda: [
            DiscussionPhase.WARMUP,
            DiscussionPhase.EXPLORATION,
            DiscussionPhase.DEEP_DIVE,
            DiscussionPhase.REACTION,
            DiscussionPhase.SYNTHESIS,
        ]
    )
    questions_per_phase: int = 2
    max_responses_per_question: int = 5
    temperature: float = 0.9
    model: str | None = None
    seed: int | None = 42


class DiscussionTranscript(BaseModel):
    config: DiscussionConfig
    messages: list[DiscussionMessage] = Field(default_factory=list)
    personas: list = Field(default_factory=list)
