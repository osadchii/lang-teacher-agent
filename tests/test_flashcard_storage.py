from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.db.flashcards import (
    FlashcardPayload,
    ensure_user_flashcard,
    get_next_flashcard_for_user,
    get_or_create_flashcard,
    get_user_flashcard_by_source_text,
)


@pytest.mark.asyncio
async def test_get_or_create_flashcard_reuses_existing(session_factory) -> None:
    payload = FlashcardPayload(
        source_text="λόγος",
        target_text="слово",
        example="Ο λόγος είναι σημαντικός.",
    )

    async with session_factory() as session:
        async with session.begin():
            _, created_first = await get_or_create_flashcard(session, payload)
        async with session.begin():
            flashcard, created_second = await get_or_create_flashcard(session, payload)

    assert created_first is True
    assert created_second is False
    assert flashcard.source_text == "λόγος"
    assert flashcard.target_text == "слово"


@pytest.mark.asyncio
async def test_ensure_user_flashcard_handles_reactivation(session_factory) -> None:
    payload = FlashcardPayload(source_text="μαθαίνω", target_text="учить")
    chat_id = 101
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        async with session.begin():
            flashcard, _ = await get_or_create_flashcard(session, payload)
            user_flashcard, created, reactivated = await ensure_user_flashcard(
                session, chat_id, flashcard, now=now
            )
            user_flashcard.is_active = False
        async with session.begin():
            second_flashcard, created_second, reactivated_second = await ensure_user_flashcard(
                session, chat_id, flashcard, now=now
            )

    assert created is True
    assert reactivated is False
    assert created_second is False
    assert reactivated_second is True
    assert second_flashcard.is_active is True


@pytest.mark.asyncio
async def test_get_next_flashcard_for_user_prefers_due(session_factory) -> None:
    payload_due = FlashcardPayload(source_text="σπίτι", target_text="дом")
    payload_future = FlashcardPayload(source_text="θάλασσα", target_text="море")
    chat_id = 202
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        async with session.begin():
            flashcard_due, _ = await get_or_create_flashcard(session, payload_due)
            flashcard_future, _ = await get_or_create_flashcard(session, payload_future)
            user_due, _, _ = await ensure_user_flashcard(session, chat_id, flashcard_due, now=now)
            user_due.next_review_at = now - timedelta(minutes=5)
            await ensure_user_flashcard(session, chat_id, flashcard_future, now=now)

        result_due = await get_next_flashcard_for_user(session, chat_id, now=now)
        assert result_due is not None
        assert result_due.flashcard.source_text == "σπίτι"

        user_due.next_review_at = now + timedelta(days=1)
        await session.flush()

        result_future = await get_next_flashcard_for_user(session, chat_id, now=now)
        assert result_future is not None
        assert result_future.flashcard.source_text == "θάλασσα"


@pytest.mark.asyncio
async def test_get_user_flashcard_by_source_text_finds_existing(session_factory) -> None:
    payload = FlashcardPayload(
        source_text="καλημέρα",
        target_text="доброе утро",
        source_lang="Greek",
        target_lang="Russian",
    )
    chat_id = 303
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        async with session.begin():
            flashcard, _ = await get_or_create_flashcard(session, payload)
            await ensure_user_flashcard(session, chat_id, flashcard, now=now)

        async with session.begin():
            found = await get_user_flashcard_by_source_text(
                session, chat_id, "καλημέρα", "Greek"
            )

    assert found is not None
    assert found.flashcard.source_text == "καλημέρα"
    assert found.flashcard.target_text == "доброе утро"
    assert found.chat_id == chat_id


@pytest.mark.asyncio
async def test_get_user_flashcard_by_source_text_returns_none_for_different_user(
    session_factory,
) -> None:
    payload = FlashcardPayload(
        source_text="ευχαριστώ",
        target_text="спасибо",
        source_lang="Greek",
        target_lang="Russian",
    )
    chat_id_one = 404
    chat_id_two = 405
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        async with session.begin():
            flashcard, _ = await get_or_create_flashcard(session, payload)
            await ensure_user_flashcard(session, chat_id_one, flashcard, now=now)

        async with session.begin():
            found_for_user_one = await get_user_flashcard_by_source_text(
                session, chat_id_one, "ευχαριστώ", "Greek"
            )
            found_for_user_two = await get_user_flashcard_by_source_text(
                session, chat_id_two, "ευχαριστώ", "Greek"
            )

    assert found_for_user_one is not None
    assert found_for_user_one.chat_id == chat_id_one
    assert found_for_user_two is None


@pytest.mark.asyncio
async def test_get_user_flashcard_by_source_text_ignores_translation(
    session_factory,
) -> None:
    """Test that same Greek word with different translation is still detected."""
    payload_first = FlashcardPayload(
        source_text="φίλος",
        target_text="друг",
        source_lang="Greek",
        target_lang="Russian",
    )
    chat_id = 506

    async with session_factory() as session:
        async with session.begin():
            flashcard, _ = await get_or_create_flashcard(session, payload_first)
            await ensure_user_flashcard(
                session, chat_id, flashcard, now=datetime.now(timezone.utc)
            )

        async with session.begin():
            # Try to find with same source but different target shouldn't matter
            found = await get_user_flashcard_by_source_text(
                session, chat_id, "φίλος", "Greek"
            )

    assert found is not None
    assert found.flashcard.source_text == "φίλος"
    assert found.flashcard.target_text == "друг"  # Original translation
