from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.bot.srs import calculate_next_schedule


def test_successful_review_increases_interval() -> None:
    now = datetime.now(timezone.utc)
    schedule = calculate_next_schedule(
        score=5,
        current_easiness=2.5,
        current_interval=2,
        current_repetition=2,
        now=now,
    )

    assert schedule.repetition == 3
    assert schedule.interval >= 5  # interval grows with easiness factor
    assert schedule.next_review_at >= now + timedelta(days=5)
    assert schedule.easiness_factor > 2.5


def test_failed_review_resets_progress() -> None:
    now = datetime.now(timezone.utc)
    schedule = calculate_next_schedule(
        score=1,
        current_easiness=2.2,
        current_interval=10,
        current_repetition=4,
        now=now,
    )

    assert schedule.repetition == 0
    assert schedule.interval == 1
    assert schedule.next_review_at == now + timedelta(days=1)
    assert schedule.easiness_factor >= 1.3
