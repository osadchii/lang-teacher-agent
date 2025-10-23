from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from . import User


@dataclass(slots=True)
class UserStatistics:
    """Aggregated metrics describing a Telegram user's activity."""

    chat_id: int
    first_name: Optional[str]
    last_name: Optional[str]
    created_at: datetime
    questions_asked: int
    flashcards_added: int
    flashcards_reviewed: int
    flashcards_mastered: int


async def upsert_user(
    session: AsyncSession,
    chat_id: int,
    first_name: Optional[str],
    last_name: Optional[str],
) -> User:
    """Create or update a user record based on the latest Telegram payload."""
    user = await session.get(User, chat_id)

    if user is None:
        now = datetime.now(timezone.utc)
        user = User(
            chat_id=chat_id,
            first_name=first_name,
            last_name=last_name,
            created_at=now,
            updated_at=now,
        )
        session.add(user)
        return user

    has_changes = False

    if user.first_name != first_name:
        user.first_name = first_name
        has_changes = True

    if user.last_name != last_name:
        user.last_name = last_name
        has_changes = True

    if has_changes:
        user.updated_at = datetime.now(timezone.utc)
        await session.flush()

    return user


async def increment_user_statistics(
    session: AsyncSession,
    chat_id: int,
    *,
    questions: int = 0,
    added: int = 0,
    reviewed: int = 0,
    mastered: int = 0,
) -> None:
    """Increment one or more user statistics counters."""
    values = {}
    if questions:
        values["questions_asked"] = User.questions_asked + questions
    if added:
        values["flashcards_added"] = User.flashcards_added + added
    if reviewed:
        values["flashcards_reviewed"] = User.flashcards_reviewed + reviewed
    if mastered:
        values["flashcards_mastered"] = User.flashcards_mastered + mastered

    if not values:
        return

    values["updated_at"] = datetime.now(timezone.utc)

    stmt = (
        update(User)
        .where(User.chat_id == chat_id)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    await session.execute(stmt)


async def get_user_statistics(session: AsyncSession, chat_id: int) -> Optional[UserStatistics]:
    """Return consolidated statistics for a user, if present."""
    user = await session.get(User, chat_id)
    if user is None:
        return None
    return UserStatistics(
        chat_id=user.chat_id,
        first_name=user.first_name,
        last_name=user.last_name,
        created_at=user.created_at,
        questions_asked=user.questions_asked,
        flashcards_added=user.flashcards_added,
        flashcards_reviewed=user.flashcards_reviewed,
        flashcards_mastered=user.flashcards_mastered,
    )
