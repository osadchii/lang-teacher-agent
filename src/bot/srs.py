"""Spaced-repetition scheduling helpers for flashcard reviews."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.db.flashcards import DEFAULT_EASINESS_FACTOR


MIN_EASINESS_FACTOR = 1.3
MAX_INTERVAL_DAYS = 365


@dataclass(slots=True)
class ReviewSchedule:
    """Calculated review data for a flashcard after receiving a score."""

    next_review_at: datetime
    easiness_factor: float
    interval: int
    repetition: int


def calculate_next_schedule(
    *,
    score: int,
    current_easiness: float,
    current_interval: int,
    current_repetition: int,
    now: datetime | None = None,
) -> ReviewSchedule:
    """Return the next review schedule using a simplified SM-2 algorithm."""
    if now is None:
        now = datetime.now(timezone.utc)

    quality = max(0, min(5, score))
    easiness_factor = current_easiness or DEFAULT_EASINESS_FACTOR
    repetition = current_repetition or 0
    interval = max(0, current_interval or 0)

    if quality < 3:
        repetition = 0
        interval = 1
        easiness_factor = max(MIN_EASINESS_FACTOR, easiness_factor - 0.2)
    else:
        repetition += 1
        if repetition == 1:
            interval = 1
        elif repetition == 2:
            interval = 6
        else:
            interval = max(1, round(interval * easiness_factor))
        easiness_factor += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        if easiness_factor < MIN_EASINESS_FACTOR:
            easiness_factor = MIN_EASINESS_FACTOR

    if interval > MAX_INTERVAL_DAYS:
        interval = MAX_INTERVAL_DAYS

    next_review_at = now + timedelta(days=interval)

    return ReviewSchedule(
        next_review_at=next_review_at,
        easiness_factor=easiness_factor,
        interval=interval,
        repetition=repetition,
    )
