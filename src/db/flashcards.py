"""Helpers for working with flashcard persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from . import Flashcard, FlashcardReview, UserFlashcard


DEFAULT_EASINESS_FACTOR = 2.5


@dataclass(slots=True)
class FlashcardPayload:
    """Definition of a flashcard that may be persisted or re-used."""

    source_text: str
    target_text: str
    example: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    tags: Optional[str] = None

    def normalized(self) -> "FlashcardPayload":
        """Return a payload with leading/trailing whitespace stripped."""
        return FlashcardPayload(
            source_text=self.source_text.strip(),
            target_text=self.target_text.strip(),
            example=self.example.strip() if isinstance(self.example, str) else self.example,
            source_lang=self.source_lang.strip() if isinstance(self.source_lang, str) else self.source_lang,
            target_lang=self.target_lang.strip() if isinstance(self.target_lang, str) else self.target_lang,
            tags=self.tags.strip() if isinstance(self.tags, str) else self.tags,
        )


async def get_or_create_flashcard(
    session: AsyncSession, payload: FlashcardPayload
) -> tuple[Flashcard, bool]:
    """Fetch a shared flashcard or create it when missing."""
    normalized = payload.normalized()

    stmt = select(Flashcard).where(
        Flashcard.source_text == normalized.source_text,
        Flashcard.target_text == normalized.target_text,
        Flashcard.source_lang.is_(normalized.source_lang) if normalized.source_lang is None else Flashcard.source_lang == normalized.source_lang,
        Flashcard.target_lang.is_(normalized.target_lang) if normalized.target_lang is None else Flashcard.target_lang == normalized.target_lang,
    )
    result = await session.execute(stmt)
    flashcard = result.scalars().first()

    if flashcard is not None:
        # Update missing optional details if the newly provided payload is richer.
        has_changes = False
        if normalized.example and not flashcard.example:
            flashcard.example = normalized.example
            has_changes = True
        if normalized.tags and not flashcard.tags:
            flashcard.tags = normalized.tags
            has_changes = True
        if has_changes:
            await session.flush()
        return flashcard, False

    flashcard = Flashcard(
        source_text=normalized.source_text,
        target_text=normalized.target_text,
        example=normalized.example,
        source_lang=normalized.source_lang,
        target_lang=normalized.target_lang,
        tags=normalized.tags,
    )
    session.add(flashcard)
    await session.flush()
    return flashcard, True


async def ensure_user_flashcard(
    session: AsyncSession,
    chat_id: int,
    flashcard: Flashcard,
    now: Optional[datetime] = None,
) -> tuple[UserFlashcard, bool, bool]:
    """Attach a shared flashcard to a user, activating it if previously disabled."""
    if now is None:
        now = datetime.now(timezone.utc)

    stmt = select(UserFlashcard).where(
        UserFlashcard.chat_id == chat_id,
        UserFlashcard.flashcard_id == flashcard.id,
    )
    result = await session.execute(stmt)
    user_flashcard = result.scalars().first()

    if user_flashcard is not None:
        was_inactive = not user_flashcard.is_active
        if not user_flashcard.is_active:
            user_flashcard.is_active = True
        current_due = user_flashcard.next_review_at
        if current_due is None:
            user_flashcard.next_review_at = now
        else:
            if current_due.tzinfo is None:
                current_due = current_due.replace(tzinfo=timezone.utc)
            user_flashcard.next_review_at = min(current_due, now)
        await session.flush()
        return user_flashcard, False, was_inactive

    user_flashcard = UserFlashcard(
        chat_id=chat_id,
        flashcard_id=flashcard.id,
        easiness_factor=DEFAULT_EASINESS_FACTOR,
        interval=0,
        repetition=0,
        next_review_at=now,
        last_score=None,
        is_active=True,
    )
    session.add(user_flashcard)
    await session.flush()
    return user_flashcard, True, False


async def get_next_flashcard_for_user(
    session: AsyncSession,
    chat_id: int,
    now: Optional[datetime] = None,
    include_future: bool = True,
) -> Optional[UserFlashcard]:
    """Return the next flashcard a user should study."""
    if now is None:
        now = datetime.now(timezone.utc)

    base_query = (
        select(UserFlashcard)
        .options(selectinload(UserFlashcard.flashcard))
        .where(UserFlashcard.chat_id == chat_id, UserFlashcard.is_active.is_(True))
        .order_by(UserFlashcard.next_review_at, UserFlashcard.id)
    )

    due_stmt = base_query.where(UserFlashcard.next_review_at <= now).limit(1)
    due_result = await session.execute(due_stmt)
    flashcard = due_result.scalars().first()
    if flashcard:
        return flashcard

    if not include_future:
        return None

    future_stmt = base_query.limit(1)
    future_result = await session.execute(future_stmt)
    return future_result.scalars().first()


async def record_flashcard_review(
    session: AsyncSession,
    user_flashcard: UserFlashcard,
    score: int,
    next_review_at: datetime,
    easiness_factor: float,
    interval: int,
    repetition: int,
    now: Optional[datetime] = None,
) -> None:
    """Persist a spaced-repetition review outcome for the given user flashcard."""
    if now is None:
        now = datetime.now(timezone.utc)

    user_flashcard.last_score = score
    user_flashcard.next_review_at = next_review_at
    user_flashcard.easiness_factor = easiness_factor
    user_flashcard.interval = interval
    user_flashcard.repetition = repetition
    user_flashcard.updated_at = now

    session.add(
        FlashcardReview(
            user_flashcard_id=user_flashcard.id,
            score=score,
            reviewed_at=now,
        )
    )
    await session.flush()


async def reactivate_flashcards(
    session: AsyncSession,
    user_flashcards: Sequence[UserFlashcard],
    now: Optional[datetime] = None,
) -> None:
    """Bulk-reactivate flashcards that should be made available to the user again."""
    if now is None:
        now = datetime.now(timezone.utc)

    changed = False
    for record in user_flashcards:
        if not record.is_active:
            record.is_active = True
            changed = True
        if record.next_review_at > now:
            record.next_review_at = now
            changed = True
    if changed:
        await session.flush()
