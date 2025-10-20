from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from . import User


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
