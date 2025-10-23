from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from src.bot.agent import GreekTeacherAgent
from src.db import Flashcard, UserFlashcard
from src.db.users import upsert_user
from src.main import SYSTEM_PROMPT


class _StubResponses:
    async def create(self, *args, **kwargs):  # pragma: no cover - network not expected here
        raise AssertionError("Unexpected OpenAI call in common words test.")


class _StubClient:
    def __init__(self) -> None:
        self.responses = _StubResponses()


@pytest.mark.asyncio
async def test_common_words_import_is_idempotent(session_factory) -> None:
    agent = GreekTeacherAgent(
        _StubClient(),
        "test-model",
        SYSTEM_PROMPT,
        session_factory=session_factory,
    )
    total_words = len(agent._load_common_flashcards())
    chat_id = 101
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        async with session.begin():
            await upsert_user(session, chat_id, "First", "User")
            (
                created_first,
                added_first,
                already_had_first,
                reactivated_first,
            ) = await agent._ensure_common_flashcards(
                session,
                chat_id,
                now=now,
            )

    assert created_first == total_words
    assert added_first == total_words
    assert already_had_first == 0
    assert reactivated_first == 0

    reactivation_sample = 5
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(UserFlashcard)
                .where(UserFlashcard.chat_id == chat_id)
                .order_by(UserFlashcard.id)
                .limit(reactivation_sample)
            )
            for record in result.scalars():
                record.is_active = False

    async with session_factory() as session:
        async with session.begin():
            (
                created_second,
                added_second,
                already_had_second,
                reactivated_second,
            ) = await agent._ensure_common_flashcards(
                session,
                chat_id,
                now=now,
            )

    assert created_second == 0
    assert added_second == 0
    assert already_had_second == total_words
    assert reactivated_second == reactivation_sample

    async with session_factory() as session:
        async with session.begin():
            (
                created_third,
                added_third,
                already_had_third,
                reactivated_third,
            ) = await agent._ensure_common_flashcards(
                session,
                chat_id,
                now=now,
            )

    assert created_third == 0
    assert added_third == 0
    assert already_had_third == total_words
    assert reactivated_third == 0

    second_chat_id = 202
    async with session_factory() as session:
        async with session.begin():
            await upsert_user(session, second_chat_id, "Second", "User")
            (
                created_fourth,
                added_fourth,
                already_had_fourth,
                reactivated_fourth,
            ) = await agent._ensure_common_flashcards(
                session,
                second_chat_id,
                now=now,
            )

    assert created_fourth == 0
    assert added_fourth == total_words
    assert already_had_fourth == 0
    assert reactivated_fourth == 0

    async with session_factory() as session:
        total_flashcards = await session.scalar(select(func.count()).select_from(Flashcard))
        total_links = await session.scalar(select(func.count()).select_from(UserFlashcard))

    assert total_flashcards == total_words
    assert total_links == total_words * 2
