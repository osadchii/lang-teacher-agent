from __future__ import annotations

import types

import pytest

from src.bot.flashcard_workflow import FlashcardWorkflow
from src.db.flashcards import get_next_flashcard_for_user


class _StubResponses:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0
        self.last_args: tuple | None = None
        self.last_kwargs: dict | None = None

    async def create(self, *args, **kwargs):
        self.calls += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return types.SimpleNamespace(output_text=self._payload)


class _StubClient:
    def __init__(self, payload: str) -> None:
        self.responses = _StubResponses(payload)


FLASHCARD_JSON = """
{
  "should_add": true,
  "flashcards": [
    {
      "source_text": "λόγος",
      "target_text": "слово",
      "example": "Ο λόγος είναι σημαντικός.",
      "example_translation": "Слово важно."
    }
  ]
}
"""

MULTI_FLASHCARD_JSON = """
{
  "should_add": true,
  "flashcards": [
    {
      "source_text": "ὁ ἀριστεύων",
      "target_text": "прекраснейший",
      "example": "ὁ ἀριστεύων ἀνήρ.",
      "example_translation": "Самый прекрасный мужчина."
    },
    {
      "source_text": "υπέροχος",
      "target_text": "прекрасный",
      "example": "Είναι υπέροχος φίλος.",
      "example_translation": "Он прекрасный друг."
    },
    {
      "source_text": "ὁ υπέρτατος",
      "target_text": "самый лучший",
      "example": "ὁ υπέρτατος στόχος μας.",
      "example_translation": "Наша высшая цель."
    }
  ]
}
"""


@pytest.mark.asyncio
async def test_workflow_persists_and_reuses_flashcards(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(77, "Добавь слово λόγος в карточки")

    assert result.handled is True
    assert result.errors == []
    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.status == "created"
    assert summary.example.endswith("Слово важно.")
    assert client.responses.calls == 1

    async with session_factory() as session:
        stored = await get_next_flashcard_for_user(session, 77)
        assert stored is not None
        assert stored.flashcard.source_text == "λόγος"
        assert stored.flashcard.target_text == "слово"

    repeat_result = await workflow.handle(77, "Добавь это слово ещё раз в карточки")
    assert repeat_result.handled is True
    assert repeat_result.summaries[0].status == "existing"
    assert client.responses.calls == 2


@pytest.mark.asyncio
async def test_workflow_skips_irrelevant_messages(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(88, "Привет, как дела?")

    assert result.handled is False
    assert client.responses.calls == 0


@pytest.mark.asyncio
async def test_workflow_limits_to_primary_flashcard_when_request_is_generic(session_factory) -> None:
    client = _StubClient(MULTI_FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(201, "добавь в карточки")

    assert result.handled is True
    assert len(result.summaries) == 1
    assert result.summaries[0].source_text == "ὁ ἀριστεύων"

    from src.db import UserFlashcard  # type: ignore import-cycle
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    async with session_factory() as session:
        stmt = select(UserFlashcard).options(selectinload(UserFlashcard.flashcard)).where(
            UserFlashcard.chat_id == 201
        )
        stored = await session.execute(stmt)
        cards = stored.scalars().all()
        assert len(cards) == 1
        assert cards[0].flashcard.source_text == "ὁ ἀριστεύων"


@pytest.mark.asyncio
async def test_workflow_keeps_multiple_when_terms_are_listed(session_factory) -> None:
    client = _StubClient(MULTI_FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    message = "добавь ὁ ἀριστεύων и υπέροχος"
    result = await workflow.handle(202, message)

    assert result.handled is True
    assert len(result.summaries) == 2
    sources = {summary.source_text for summary in result.summaries}
    assert sources == {"ὁ ἀριστεύων", "υπέροχος"}

    from src.db import UserFlashcard  # type: ignore import-cycle
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    async with session_factory() as session:
        stmt = select(UserFlashcard).options(selectinload(UserFlashcard.flashcard)).where(
            UserFlashcard.chat_id == 202
        )
        stored = await session.execute(stmt)
        cards = stored.scalars().all()
        assert len(cards) == 2
        assert {card.flashcard.source_text for card in cards} == {"ὁ ἀριστεύων", "υπέροχος"}


@pytest.mark.asyncio
async def test_workflow_keeps_multiple_when_user_requests_all_variants(session_factory) -> None:
    client = _StubClient(MULTI_FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(203, "добавь все варианты в карточки")

    assert result.handled is True
    assert len(result.summaries) == 3

    from src.db import UserFlashcard  # type: ignore import-cycle
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    async with session_factory() as session:
        stmt = select(UserFlashcard).options(selectinload(UserFlashcard.flashcard)).where(
            UserFlashcard.chat_id == 203
        )
        stored = await session.execute(stmt)
        cards = stored.scalars().all()
        assert len(cards) == 3
        assert {card.flashcard.source_text for card in cards} == {
            "ὁ ἀριστεύων",
            "υπέροχος",
            "ὁ υπέρτατος",
        }



@pytest.mark.asyncio
async def test_workflow_includes_context_for_extraction(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    context = (
        "Предыдущее сообщение пользователя: Как будет «больница»?\n"
        "Ответ преподавателя: По-гречески «больница» — το νοσοκομείο (to nosokomío)."
    )

    await workflow.handle(99, "Добавь в карточки", context=context)

    assert client.responses.calls == 1
    assert client.responses.last_kwargs is not None
    user_prompt = client.responses.last_kwargs["input"][1]["content"]
    assert "Добавь в карточки" in user_prompt
    assert "Контекст для создания карточек" in user_prompt
    assert "το νοσοκομείο" in user_prompt


def test_system_prompt_discourages_extra_variants(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    prompt = workflow._system_prompt
    assert "Only produce flashcards for the exact terms explicitly requested" in prompt
    assert "Do not add extra grammatical variants" in prompt


@pytest.mark.asyncio
async def test_workflow_prevents_duplicate_by_source_text(session_factory) -> None:
    """Test that workflow prevents adding same Greek word even with different translation."""
    # First card: λόγος -> слово
    client_first = _StubClient(FLASHCARD_JSON)
    workflow_first = FlashcardWorkflow(
        client=client_first,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result_first = await workflow_first.handle(777, "Добавь слово λόγος в карточки")
    assert result_first.handled is True
    assert len(result_first.summaries) == 1
    assert result_first.summaries[0].status == "created"
    assert result_first.summaries[0].source_text == "λόγος"
    assert result_first.summaries[0].target_text == "слово"

    # Second attempt: λόγος with different translation -> речь
    different_translation_json = """
    {
      "should_add": true,
      "flashcards": [
        {
          "source_text": "λόγος",
          "target_text": "речь",
          "example": "Λόγος τίμησης.",
          "example_translation": "Речь чести."
        }
      ]
    }
    """
    client_second = _StubClient(different_translation_json)
    workflow_second = FlashcardWorkflow(
        client=client_second,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result_second = await workflow_second.handle(777, "Добавь λόγος со значением речь")
    assert result_second.handled is True
    assert len(result_second.summaries) == 1
    # Should return existing card with original translation
    assert result_second.summaries[0].status == "existing"
    assert result_second.summaries[0].source_text == "λόγος"
    assert result_second.summaries[0].target_text == "слово"  # Original translation

    # Verify only one flashcard exists for this user
    async with session_factory() as session:
        from src.db import UserFlashcard
        from sqlalchemy import select

        stmt = select(UserFlashcard).where(UserFlashcard.chat_id == 777)
        result = await session.execute(stmt)
        user_cards = result.scalars().all()
        assert len(user_cards) == 1
